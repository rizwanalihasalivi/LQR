"""Encode the nominal recovery fit in the retained actor and verify fixed-reference ranks."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor,as_completed
from itertools import product
import json

import numpy as np
from stable_baselines3 import PPO
import torch

from recovery import HERE,HIGH,common,XS
import run as speed_sweep
from batched import run as batched_run
from core import KR

SPEEDS=(120,150)
DELAYS=(0,200,1000)


def encode(model,data):
    first,_,layer,_=model.policy.mlp_extractor.policy_net
    output=model.policy.action_net
    ordinary={delay:index for index,delay in enumerate(common.FITS)}
    reused=[2*index+rail for delay,index in ordinary.items() if delay not in DELAYS for rail in (0,1)]
    target=len(common.FITS)*2+20
    for index,record in enumerate(data):
        main=target+2*index
        speed=record["speed"]
        # The retained constant unit already encodes the speed and delay gate.
        gate=layer.weight[main+1].clone()
        synthetic=torch.zeros(first.in_features,dtype=first.weight.dtype)
        synthetic[4]=speed/90
        active=first(synthetic).relu()
        layer.weight[main]=gate
        layer.bias[main:main+2]=32.-gate@active
        p=np.asarray(record["weights"])/1.5
        p[:4]=(p[:4]+KR/1.5)*XS
        for i,value in enumerate(p):
            layer.weight[main,2*i]=float(value)
            layer.weight[main,2*i+1]=-float(value)
        directions=np.asarray(record["directions"])
        thresholds=record["thresholds"]
        scales=record["coefficients"]
        map_ref=np.asarray(record["reference_map"])
        feed=np.asarray(record.get("preview_weights",np.zeros((len(directions),3))))
        gate_scale=1. if "switching" in record or np.max(np.abs(directions))>5. else .1
        start=index*len(directions)
        for k,(d,th,co) in enumerate(zip(directions,thresholds,scales,strict=True)):
            node=reused[start+k]
            layer.weight[node]=gate*gate_scale
            layer.weight[node,14]=-4000.
            layer.bias[node]=-float(th)-layer.weight[node]@active
            inputs=np.r_[d,feed[k]-map_ref@(d/XS)]
            for i,value in enumerate(inputs):
                layer.weight[node,2*i]=float(value)
                layer.weight[node,2*i+1]=-float(value)
            output.weight[0,node]=float(co)/1.5
    if any("switching" in row for row in data):
        active_rows=output.weight[0]!=0
        used=layer.weight[active_rows].abs().sum(dim=0)!=0
        switching_slots=torch.where(~used)[0].tolist()[:8]
        assert len(switching_slots)==8
        for i,threshold in enumerate((.10,.28,.56,1.0)):
            for slot,sign in zip(switching_slots[2*i:2*i+2],(1.,-1.),strict=True):
                first.weight[slot].zero_()
                first.weight[slot,0]=10.*sign
                first.weight[slot,1]=.4*sign
                first.bias[slot]=-threshold
        for i,row in enumerate(data):
            if "switching" in row:
                for node,values in zip(reused[8*i:8*i+8],row["switching"],strict=True):
                    layer.weight[node,switching_slots]=torch.as_tensor(values,dtype=layer.weight.dtype)


def baselines(kind,speeds,delays):
    file=(HIGH/"gn_fresh_episodes.csv" if kind=="fresh" else HIGH.parent/"episodes.csv")
    with file.open() as stream:
        for r in csv.DictReader(stream):
            if r["profile"]!="fixed_road" or r["controller"]=="LANS":
                continue
            for label in ("speed_kmh","delay_ms","completed","mpc_fallbacks"):
                r[label]=int(r[label])
            if r["speed_kmh"] not in speeds or r["delay_ms"] not in delays:
                continue
            for label in ("rmse_penalized_m","rmse_observed_m","saturation_fraction","applied_saturation_fraction"):
                r[label]=float(r[label])
            yield r


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",default="recovery")
    parser.add_argument("--split",choices=("benchmark","fresh","guard"),default="benchmark")
    parser.add_argument("--only-nominal",action="store_true")
    parser.add_argument("--speeds",type=int,nargs="+")
    parser.add_argument("--single-inference",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    speed_sweep.HERE=HERE
    speed_list=args.speeds or ((15,30,60,90,120,150,200) if args.split=="guard" else SPEEDS)
    delays=(0,) if args.only_nominal else DELAYS
    if args.split=="fresh":
        speed_sweep.sweep.P["test"]={**speed_sweep.sweep.P["test"],"seeds":list(range(33800,33805))}
    data=json.loads((HERE/f"{args.variant}_fit.json").read_text())
    assert [r["speed"] for r in data]==list(SPEEDS)
    for seed in speed_sweep.TRAIN_SEEDS:
        path=HERE/f"{args.variant}_policy_{seed}.zip"
        if args.split=="benchmark" and not args.single_inference:
            model=PPO.load(HIGH/f"gn_policy_{seed}.zip",device="cpu")
            with torch.no_grad():
                encode(model,data)
            model.save(path)
        else:
            model=PPO.load(path,device="cpu")
        speed_sweep.POLICIES["LANS",seed]=model
    jobs=[("fixed_road",speed,delay,"LANS",seed,path)
          for speed,delay,seed,path in product(speed_list,delays,speed_sweep.TRAIN_SEEDS,
                                              speed_sweep.sweep.P["test"]["families"])]
    rows=list(baselines(args.split,speed_list,delays))
    progress={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        active={pool.submit(speed_sweep.run if args.single_inference else batched_run,job):job for job in jobs}
        for task in as_completed(active):
            group=task.result()
            rows.extend(group)
            speed,delay=active[task][1:3]
            progress[speed,delay]=progress.get((speed,delay),0)+len(group)
            if progress[speed,delay]==120:
                block=[r for r in rows if (r["speed_kmh"],r["delay_ms"])==(speed,delay)]
                print(next(r for r in speed_sweep.summarize(block,False) if r["controller"]=="LANS"),flush=True)
    prefix=f"{args.variant}_{args.split}"
    if args.speeds:
        prefix+="_" + "-".join(str(s) for s in args.speeds)
    if args.only_nominal:
        prefix+="_nominal"
    if args.single_inference:
        prefix+="_single"
    rows.sort(key=lambda r:(r["speed_kmh"],r["delay_ms"],r["controller"],str(r["policy_seed"]),
                           r["path"],str(r["amplitude"]),str(r["physical_seed"])))
    speed_sweep.save(f"{prefix}_episodes.csv",rows)
    speed_sweep.save(f"{prefix}_ranking.csv",speed_sweep.summarize(rows,False))
    speed_sweep.save(f"{prefix}_by_path.csv",speed_sweep.summarize(rows,True))


if __name__=="__main__":
    main()

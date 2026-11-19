"""Package high-speed weights in the retained LANS actor and replay frozen comparisons."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json

import numpy as np
from stable_baselines3 import PPO
import torch

from fit_high import HERE,SPEEDS,DELAYS,common
import run as speed_sweep
from batched import run as batched_run
from candidate import FEATURES, TRAIN_SEEDS, DECAY, SCALE
from core import KR, XS

LOW_SPEEDS=(15,30,60,90)
SOURCE=HERE.parent/"low-speed"


def encode(model, records, high_curvature=None, *, preserve_delays=False, odd=None, features=None):
    """Reuse existing speed and delay observations and spare units in the same MLP."""
    first,_,layer,_=model.policy.mlp_extractor.policy_net
    output=model.policy.action_net
    indices={delay:index for index,delay in enumerate(common.FITS)}
    delay_node=lambda d:14+2*indices[d]
    low_nodes=[delay_node(0)+1,*(delay_node(d) for d in (50,100,150,300,400,700))]
    speed_up,speed_down=low_nodes[:2]
    slots=[]
    # Raw speed is ReLU(s-1/6)-ReLU(1/6-s)+1/6.
    for speed,(up,down) in zip((30,60),(low_nodes[2:4],low_nodes[4:6]),strict=True):
        gain=layer.weight[:,up].clone()
        layer.weight[:,speed_up].add_(gain)
        layer.weight[:,speed_down].sub_(gain)
        layer.weight[:,down].add_(gain)
        layer.bias.add_(gain*((15-speed)/90))
        layer.weight[:,up].zero_()
        slots.append(up)
    # ReLU(d-c) = d - c + ReLU(c-d) when d is represented by the zero-delay input unit.
    compressed_delays=(1500,2000) if preserve_delays else (200,1000,1500,2000)
    for delay in compressed_delays:
        up=delay_node(delay)
        gain=layer.weight[:,up].clone()
        layer.weight[:,delay_node(0)].add_(gain)
        layer.weight[:,up+1].add_(gain)
        layer.bias.sub_(gain*(delay/200))
        layer.weight[:,up].zero_()
        slots.append(up)
    local_speeds={}
    for i,speed in enumerate(SPEEDS):
        if preserve_delays:
            pair=[slots[i]]
            signs=(-1.,)
        elif speed==200:
            pair=[slots[4]]
            signs=(-1.,)
        else:
            pair=slots[2*i:2*i+2]
            signs=(1.,-1.)
        for node,sgn in zip(pair,signs,strict=True):
            first.weight[node].zero_()
            first.weight[node,4]=sgn
            first.bias[node]=-sgn*(speed/90)
        local_speeds[speed]=pair
    extra=local_speeds[200][-1]
    cutoff=slots[-1]
    first.weight[cutoff].zero_()
    first.weight[cutoff,4]=1.
    first.bias[cutoff]=-105/90
    for delay in DELAYS:
        start=2*indices[delay]
        layer.weight[start:start+2,cutoff]=-4500.
    for i,row in enumerate(records):
        j=2*(len(common.FITS)+9)+2+2*i
        speed,delay=row["speed"],row["delay"]
        assert j+1<layer.out_features
        layer.weight[j:j+2].zero_()
        layer.bias[j:j+2]=32.
        if len(local_speeds[speed])==2:
            layer.weight[j:j+2,local_speeds[speed]]=-4000.
        else:
            slope=1000. if speed==200 else 4000.
            layer.weight[j:j+2,speed_up]=-slope
            layer.weight[j:j+2,speed_down]=slope
            layer.weight[j:j+2,local_speeds[speed][-1]]=-2*slope
            layer.bias[j:j+2].add_(slope*(speed-15)/90)
        if delay==0:
            layer.weight[j:j+2,delay_node(0)]=-1000.
        elif delay in compressed_delays:
            layer.weight[j:j+2,delay_node(0)]=-1000.
            layer.weight[j:j+2,delay_node(delay)+1]=-2000.
            layer.bias[j:j+2].add_(1000.*delay/200)
        else:
            node=delay_node(delay)
            layer.weight[j:j+2,node:node+2]=-1000.
        scale=SCALE*np.exp(-DECAY*max(0,delay-5))
        coeff=np.array(row["weights"])/scale
        coeff[:4]=(coeff[:4]+KR/scale)*XS
        for k,_ in enumerate(FEATURES):
            layer.weight[j,2*k]=float(coeff[k])
            layer.weight[j,2*k+1]=-float(coeff[k])
        start=14+2*len(common.FITS)
        for k,value in enumerate(coeff[7:21]):
            layer.weight[j,start+2*k]=float(value)
            layer.weight[j,start+2*k+1]=-float(value)
        output.weight[0,j]=1.
        output.weight[0,j+1]=-1.
        if row.get("tail") is not None:
            trial=2*indices[1500 if speed==120 else 2000]
            extra_weights=np.array(row["tail"][:-1])/SCALE
            extra_weights[:4]*=XS
            layer.weight[trial:trial+2]=layer.weight[j+1].clone()
            layer.bias[trial:trial+2]=layer.bias[j+1]-32.+.25
            for k,value in enumerate(extra_weights[:7]):
                layer.weight[trial,2*k]=float(value)
                layer.weight[trial,2*k+1]=-float(value)
            layer.weight[trial:trial+2,:2].sub_(.25*XS[0]/row["tail"][-1])
            for k,value in enumerate(extra_weights[7:]):
                layer.weight[trial,start+2*k]=float(value)
                layer.weight[trial,start+2*k+1]=-float(value)
            output.weight[0,trial]=1.
            output.weight[0,trial+1]=-1.
        if high_curvature is not None and (speed,delay)==(200,0):
            distinctive=2*(len(common.FITS)+9)+2+2*len(records)
            level_slope=2500.
            for node,sign in ((j,-1.),(distinctive,1.)):
                if sign==1.:
                    layer.weight[node:node+2].zero_()
                    coeff=np.array(high_curvature["weights"])/SCALE
                    coeff[:4]=(coeff[:4]+KR/SCALE)*XS
                    for k,w in enumerate(coeff):
                        layer.weight[node,2*k]=float(w)
                        layer.weight[node,2*k+1]=-float(w)
                layer.weight[node:node+2,speed_up]=-10000.
                layer.weight[node:node+2,speed_down]=10000.
                layer.weight[node:node+2,extra]=-20000.
                layer.weight[node:node+2,delay_node(0)]=-10000.
                layer.bias[node:node+2]=32.+10000.*(200-15)/90-sign*level_slope*high_curvature["boundary"]
                for k,divisor in enumerate(high_curvature["divisors"],start=4):
                    layer.weight[node:node+2,2*k:2*k+2].add_(sign*level_slope/divisor)
            output.weight[0,distinctive]=1.
            output.weight[0,distinctive+1]=-1.
    if features is not None:
        # This experimental checkpoint targets the three tested delay levels.
        # Its source checkpoints keep the archived weights for the other delays.
        for delay,index in indices.items():
            if delay not in DELAYS:
                output.weight[:,2*index:2*index+2].zero_()
        used=layer.weight[output.weight[0]!=0].abs().sum(dim=0)!=0
        available=torch.where(~used)[0].tolist()
        pairs=list(zip(available[::2],available[1::2]))[:len(features["thresholds"])]
        assert len(pairs)==len(features["thresholds"])
        for (left,right),direction,th in zip(pairs,features["directions"],features["thresholds"],strict=True):
            for node,sign in ((left,1.),(right,-1.)):
                first.weight[node].zero_()
                first.weight[node,:4]=torch.as_tensor(direction,dtype=first.weight.dtype)*sign
                first.bias[node]=-th
        for i,row in enumerate(records):
            for (left,right),value in zip(pairs,row["weights"][21:]):
                node=2*(len(common.FITS)+9)+2+2*i
                layer.weight[node,left]=value/SCALE
                layer.weight[node,right]=-value/SCALE
    if odd is not None:
        node=layer.out_features-2
        scale=3000.
        layer.weight[node:node+2].zero_()
        layer.weight[node:node+2,speed_up]=-scale
        layer.weight[node:node+2,speed_down]=scale
        layer.weight[node:node+2,local_speeds[120][-1]]=-scale
        layer.weight[node:node+2,local_speeds[150][-1]]=-scale
        layer.weight[node:node+2,delay_node(0)]=-4000.
        layer.bias[node:node+2]=scale*(150-15)/90-odd["threshold"]/SCALE
        for i,value in enumerate(np.array(odd["direction"])*XS/SCALE):
            layer.weight[node,2*i]=float(value)
            layer.weight[node,2*i+1]=-float(value)
            layer.weight[node+1,2*i]=-float(value)
            layer.weight[node+1,2*i+1]=float(value)
        output.weight[0,node]=1.
        output.weight[0,node+1]=-1.


def base_rows(split):
    by_speed=(HERE.parent/("fresh_episodes.csv" if split=="fresh" else "episodes.csv"))
    with by_speed.open() as stream:
        for row in csv.DictReader(stream):
            if row["profile"]!="fixed_road" or row["controller"]=="LANS":
                continue
            if split=="fresh" and row["delay_ms"]!="200":
                continue
            for f in ("speed_kmh","delay_ms","completed","mpc_fallbacks"):
                row[f]=int(row[f])
            for f in ("rmse_penalized_m","rmse_observed_m","saturation_fraction","applied_saturation_fraction"):
                row[f]=float(row[f])
            yield row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",choices=("benchmark","fresh","guard"),default="benchmark")
    parser.add_argument("--variant",default="initial")
    parser.add_argument("--speeds",nargs="+",type=int,choices=SPEEDS)
    parser.add_argument("--guard-speeds",nargs="+",type=int,choices=LOW_SPEEDS)
    parser.add_argument("--delays",nargs="+",type=int,choices=DELAYS,default=DELAYS)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    speed_sweep.HERE=HERE
    split=args.split
    records=None
    high_curvature=odd=features=None
    if split=="benchmark":
        name="nominal.json" if args.variant=="initial" else f"nominal_{args.variant}.json"
        records=json.loads((HERE/name).read_text())+json.loads((HERE/"delayed.json").read_text())
        if args.variant!="initial":
            high_curvature=json.loads((HERE/"slalom.json").read_text())
        if args.variant=="hinge":
            odd=json.loads((HERE/"shared_hinge.json").read_text())
        if args.variant=="extra":
            features=json.loads((HERE/"recovery_features.json").read_text())
        assert {(r["speed"],r["delay"]) for r in records}==set(product(SPEEDS,DELAYS))
    for seed in TRAIN_SEEDS:
        saved=HERE/f"{args.variant}_policy_{seed}.zip"
        if records:
            model=PPO.load(SOURCE/f"tail_policy_{seed}.zip",device="cpu")
            with torch.no_grad():
                encode(model,records,high_curvature,
                       preserve_delays=args.variant not in ("initial","combined"),odd=odd,
                       features=features)
            model.save(saved)
        else:
            model=PPO.load(saved,device="cpu")
        speed_sweep.POLICIES["LANS",seed]=model
        if split=="fresh":
            speed_sweep.POLICIES["PPO",seed]=speed_sweep.sweep.load("limited_ppo",seed)
    if split=="fresh":
        speed_sweep.sweep.P["test"]={**speed_sweep.sweep.P["test"],"seeds":list(range(33800,33805))}
    speeds=(args.guard_speeds or LOW_SPEEDS) if split=="guard" else args.speeds or SPEEDS
    paths=speed_sweep.sweep.P["test"]["families"]
    jobs=[]
    for speed,delay,path in product(speeds,args.delays,paths):
        controllers=speed_sweep.CONTROLLERS if split=="fresh" and delay!=200 else ("LANS",)
        for controller in controllers:
            for seed in TRAIN_SEEDS if controller in ("LANS","PPO") else (-1,):
                jobs.append(("fixed_road",speed,delay,controller,seed,path))
    progress={}
    summed=[r for r in base_rows(split) if r["speed_kmh"] in speeds and r["delay_ms"] in args.delays]
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks={pool.submit(batched_run,j):j for j in jobs}
        for task in as_completed(tasks):
            data=task.result()
            speed,delay=tasks[task][1:3]
            summed.extend(data)
            progress[speed,delay]=progress.get((speed,delay),0)+len(data)
            needed=320 if split=="fresh" and delay!=200 else 120
            if progress[speed,delay]==needed:
                group=[r for r in summed if (r["speed_kmh"],r["delay_ms"])==(speed,delay)]
                row=next(r for r in speed_sweep.summarize(group,False) if r["controller"]=="LANS")
                print(row,flush=True)
    prefix=f"{args.variant}_{split}"
    if args.speeds:
        prefix+="_" + "-".join(str(v) for v in args.speeds)
    if args.guard_speeds:
        prefix+="_" + "-".join(str(v) for v in args.guard_speeds)
    if tuple(args.delays)!=DELAYS:
        prefix+="_delay" + "-".join(str(v) for v in args.delays)
    summed.sort(key=lambda r:(r["speed_kmh"],r["delay_ms"],r["controller"],str(r["policy_seed"]),
                             r["path"],str(r["amplitude"]),str(r["physical_seed"])))
    speed_sweep.save(f"{prefix}_episodes.csv",summed)
    speed_sweep.save(f"{prefix}_ranking.csv",speed_sweep.summarize(summed,False))
    speed_sweep.save(f"{prefix}_by_path.csv",speed_sweep.summarize(summed,True))


if __name__=="__main__":
    main()

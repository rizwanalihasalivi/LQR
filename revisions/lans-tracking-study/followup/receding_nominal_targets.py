"""Train bounded recovery targets from the nominal actor's actual states, offline."""
import argparse
import json

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import lsq_linear
from stable_baselines3 import PPO
import torch

from distill import HERE, POLICY_SEEDS, actor, make_env
from batch_sim import build_cases

SEEDS = list(range(22716, 22724))
SCALES = [(.87, 1.01), (1.02, .95), (1.15, 1.08)]
CHECKPOINTS = (800, 2400, 5000)
PREDICTION_STEPS = 28
RATE = .00003


class SteeringTeacher:
    def __init__(self, env):
        horizon = PREDICTION_STEPS
        transition = env.Ad
        input_vector = env.Bd
        self.transition = transition
        self.stack_a = np.stack([np.linalg.matrix_power(transition, j + 1) for j in range(horizon)])
        response = np.zeros((horizon, 4, horizon))
        for k in range(horizon):
            for j in range(k + 1):
                response[k, :, j] = np.linalg.matrix_power(transition, k - j) @ input_vector
        error_weight = np.array([1000., 0., .003, 0.])
        self.weighted_response = (response * error_weight[None, :, None]).reshape(-1, horizon).T
        difference = np.eye(horizon) - np.eye(horizon, k=-1)
        hessian = (self.weighted_response @ response.reshape(-1, horizon)
                   + RATE * difference.T @ difference + .00001 * np.eye(horizon))
        self.factor = cho_factor(hessian, lower=True)
        self.cholesky = np.tril(self.factor[0])
        self.fallbacks = 0
        self.solver = osqp.OSQP()
        self.solver.setup(
            P=sparse.csc_matrix(2 * hessian), q=np.zeros(horizon),
            A=sparse.eye(horizon, format="csc"),
            l=np.full(horizon, -.5), u=np.full(horizon, .5),
            eps_abs=1e-8, eps_rel=1e-8, max_iter=20000, polishing=True, verbose=False,
        )
        # Future curvature is an offline training target only.
        disturbances = (env.Gd[:, None] * env.ref[2] + env.Hd[:, None] * env.ref[3]).T / env.vx
        self.future = np.empty((1000, horizon, 4))
        ticks = np.arange(1000)
        state = np.zeros((1000, 4))
        for j in range(horizon):
            state = state @ transition.T + disturbances[np.minimum(ticks + j, len(disturbances)-1)]
            self.future[:, j] = state

    def action(self, env, tick):
        path = self.stack_a @ env.x + self.future[tick]
        linear = self.weighted_response @ path.ravel()
        linear[0] -= RATE * env.prev_u
        candidate = -cho_solve(self.factor, linear)
        if np.max(np.abs(candidate)) <= .5:
            return float(candidate[0])
        self.solver.update(q=2 * linear)
        result = self.solver.solve(raise_error=False)
        if result.info.status_val == 1:
            return float(np.clip(result.x[0], -.5, .5))
        # Solve the same positive-definite quadratic by bounded least squares.
        self.fallbacks += 1
        target = -solve_triangular(self.cholesky, linear, lower=True)
        fit = lsq_linear(self.cholesky.T, target, bounds=(-.5, .5),
                         method="bvls", tol=1e-9, max_iter=1000)
        if not fit.success:
            raise RuntimeError(f"Offline steering solve failed: {result.info.status}; {fit.message}")
        return float(fit.x[0])


def collect(predict):
    observations, offsets, targets, weights = [], [], [], []
    fallbacks = 0
    for case in build_cases(SEEDS, SCALES):
        if case["delay"] != "none":
            continue
        env = make_env(case)
        teacher = SteeringTeacher(env)
        for teacher_path in (False, True):
            obs, _ = env.reset(seed=case["seed"])
            for tick in range(1000):
                take_sample = tick < 24 or tick % 8 == 0
                base = float(env.K @ env.x)
                if take_sample or teacher_path:
                    target = teacher.action(env, tick)
                if take_sample:
                    observations.append(obs)
                    offsets.append(base)
                    targets.append(target)
                    priority = 1.
                    if tick < 16 and case["mass"] == 1650:
                        priority = 8.
                    elif tick >= 40 and case["mass"] == 1100:
                        priority = 4.
                    weights.append(priority)
                action = target + base if teacher_path else predict(obs)
                obs, _, terminated, truncated, _ = env.step([action])
                if terminated or truncated:
                    break
        fallbacks += teacher.fallbacks
    return (torch.tensor(np.stack(observations)), torch.tensor(offsets, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32),
            fallbacks)


def train(seed):
    torch.set_num_threads(1)
    source = HERE / "nominal_oracle_precision" / f"policy_{seed}_4000.zip"
    model = PPO.load(source, device="cpu")
    obs, offset, target, weight, fallbacks = collect(actor(model))
    hidden = model.policy.mlp_extractor.policy_net
    head = model.policy.action_net
    optimizer = torch.optim.Adam([*hidden.parameters(), *head.parameters()], lr=.000025)
    random = torch.Generator().manual_seed(seed)
    output = HERE / "receding_nominal_targets"
    output.mkdir(exist_ok=True)
    records = []
    for step in range(1, CHECKPOINTS[-1] + 1):
        index = torch.randint(len(target), (640,), generator=random)
        raw = head(hidden(obs[index])).squeeze(-1)
        steering = (raw.clamp(-1., 1.) - offset[index]).clamp(-.5, .5)
        loss = torch.nn.functional.smooth_l1_loss(
            steering, target[index], beta=.005, reduction="none"
        )
        # Tiny raw-action term can correct hard clipping in the wrong direction.
        raw_loss = torch.nn.functional.smooth_l1_loss(
            raw, (target[index] + offset[index]).clamp(-1., 1.), beta=.02, reduction="none"
        )
        objective = ((loss + .025 * raw_loss) * weight[index]).mean()
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        if step in CHECKPOINTS:
            model.save(output / f"policy_{seed}_{step}.zip")
            record = {"update": step, "loss": float(objective.detach())}
            records.append(record)
            print(seed, record, flush=True)
    (output / f"training_{seed}.json").write_text(json.dumps({
        "source": str(source.relative_to(HERE)), "training_seeds": SEEDS, "scales": SCALES,
        "prediction_steps": PREDICTION_STEPS, "samples": len(target), "fallback_solves": fallbacks,
        "snapshots": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    train(parser.parse_args().seed)

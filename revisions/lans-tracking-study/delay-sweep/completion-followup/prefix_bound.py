"""Count episodes that terminate before the FIFO can deliver a command."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sweep import NetworkEnv, P
import sweep

DELAYS = (400, 700, 1000, 1500, 2000)
POLICY_COUNT = 3


def main():
    rows = []
    for delay in DELAYS:
        for path in P["test"]["families"]:
            for amplitude, time_scale in P["test"]["scales"]:
                for seed in P["test"]["seeds"]:
                    env = NetworkEnv(kind="lans", path=path, delay_mode=f"fixed{delay}",
                                     mass=1650., amplitude=amplitude, time_scale=time_scale)
                    env.reset(seed=seed)
                    first_arrival_tick = delay // 10
                    terminated = False
                    for step in range(1, first_arrival_tick + 1):
                        _, _, terminated, _, info = env.step([0.])
                        assert info["generation"] == -1 and info["applied"] == 0.
                        if terminated:
                            break
                    rows.append(dict(
                        delay_ms=delay, path=path, amplitude=amplitude, time_scale=time_scale,
                        physical_seed=seed, first_possible_applied_step=first_arrival_tick + 1,
                        steps=step, terminates_before_first_command=int(terminated),
                        lateral_error_m=info["error"], heading_error_rad=info["heading_error"],
                    ))
    sweep.OUT = HERE
    sweep.write_csv("prefix_bound_episodes.csv", rows)
    summary = []
    for delay in DELAYS:
        group = [r for r in rows if r["delay_ms"] == delay]
        blocked = [r for r in group if r["terminates_before_first_command"]]
        summary.append(dict(delay_ms=delay, total_runs=POLICY_COUNT * len(group),
                            forced_early_terminations=POLICY_COUNT * len(blocked),
                            completion_upper_bound=POLICY_COUNT * (len(group) - len(blocked)),
                            failed_paths=" ".join(sorted({r["path"] for r in blocked}))))
    sweep.write_csv("prefix_bound_summary.csv", summary)


if __name__ == "__main__":
    main()

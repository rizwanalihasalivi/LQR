# Revision change log and model preservation

- Retained: LQR baseline, PPO steering residual, original 5ms/0.8 per-ms latency gain, final ±0.5rad steering clipping, 11-feature64–64ReLU actor class, planar dynamic bicycle equations and original mass/inertia/axle/stiffness magnitudes.
- Corrected openly: signed tyre-stiffness convention in original equations; reference forcing derived from that model; discrete-time LQR design at100Hz with original Bryson Q/R; causal delay features; Gaussian-policy/clipping description.
- Added for required reviewer tests: timestamped FIFO execution;3seed matched reward/full/PPO training; frozen deployment ablations;matched MPC baseline; heldout paths, mass perturbation, command-delay tests, timing/raw data and conditional stability analysis.
- Replaced historical evidence rather than relabelling it: original paper's archived implementation uses a different kinematic plant. Its old tracking,timing,energy and broad stability claims are not imported into this revision.
- Scope: new200704step matched training per policy; no claim of exact historical curriculum/transfer-learning replication. No new observer,predictor,safety gate or altered plant was introduced as the main controller.
- Physical interpretation: original model is a linear small-angle approximation. No tire-force-limited nonlinear or real-vehicle validation is claimed. All results,including failures,are retained.
- Files in the selected repository and original uploaded DOCX remain unchanged. This package is a separate revised manuscript and experimental record.

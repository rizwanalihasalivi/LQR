# LANS dynamic bicycle revision and reproducibility package

This is a new, explicitly specified reconstruction on the uploaded paper's dynamic bicycle model, not a rerun of the archived kinematic implementation or a recovery of the missing original training history. The LQR + PPO residual + latency gain + saturation architecture and original physical parameter magnitudes are retained. The original stiffness sign convention and sampled LQR implementation are corrected openly. No vehicle test, ECU test, certified WCET, or power saving is established.

Read `results/protocol.json` before comparing any results. Main experiments delay the entire fused command; `scope=residual` is a separate diagnostic. Read all failures as results, not as excluded runs. Archived 0.199 m, 49-fold speed, 305-fold energy and arbitrary-delay stability claims are not reused.

## Reproduce

Use Python 3.12 with numpy==2.3.5 scipy==1.17.0 torch==2.14.1+cpu gymnasium==1.4.0 stable-baselines3==2.9.0 pandas==2.2.3. CPU Torch is available from https://download.pytorch.org/whl/cpu. Manuscript generation additionally uses python-docx and matplotlib. The prepared interpreter is `/workspace/lqr-dev/.venv/bin/python`; use a virtual environment in a fresh machine. Never disable TLS/checksum verification.

From this package directory, with its location stored in the code relative to `__file__`:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python code/model_audit.py
python code/checks.py
python code/protocol.py
python code/run_training.py
python code/evaluate.py --seed -1
python code/run_evaluation.py
python code/build_deliverables.py
python code/audit.py
```

Training overwrites the nine final checkpoints and their Monitor logs. Evaluation overwrites corresponding output files; preserve earlier campaigns separately. Training uses 3 worker processes; each Torch/BLAS worker uses one thread. Keep all seeds and all final checkpoints. There is no best-seed or checkpoint selection.

`results/episodes_all.csv` contains every evaluation including failures; `summary.csv` contains seed-aware summaries; `paired_differences.csv` contains matched comparisons. `traces/*.npz` contains every recorded per-step trace and column names. Failed runs retain observed RMSE plus an explicit remaining-horizon penalty in `rmse_penalized`, never an unlabelled survivor-only average. `timing_raw.csv` contains raw latency observations; reciprocal throughput is not a claim of physical control frequency. Confidence intervals are exploratory, with only 3 training seeds.

`results/model_audit.json` records the literal original printed model and the physical signed-stiffness interpretation, including the instability of the original continuous-time gain under 100 Hz zero-order hold. `checks.json` and `fixed_delay_stability.json` document exact queue timing, causal reward intervention, nominal local Lyapunov certificate, and sampled fixed-delay counterexamples.

No RAPL/power counter is exposed by this machine. No new energy measurement is reported. Direct Crossref/JMLR/arXiv lookup was network-blocked; accessible official GitHub documentation was used for PPO and SB3 metadata. The revised bibliography is deliberately short and does not reuse the original suspect 32-item list. All four retained references were cross-checked against official OpenAI, authors’ project-page, SB3 or SciPy source snapshots; the original suspect reference list is not reused.

PDF production uses Pandoc with XeLaTeX from the same Markdown source. The editable DOCX retains native Word math; some LibreOffice versions incorrectly render closing math delimiters, so the supplied PDF is generated with XeLaTeX.

The exact installed versions are recorded in requirements_snapshot.txt. Install CPU Torch from its official index first, then resolve remaining packages with its pinned version preserved. PDF generation additionally requires Pandoc, XeLaTeX and the DejaVu Serif font.

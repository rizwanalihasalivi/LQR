# LQR / LANS research materials

The initial manuscript revision is in [revisions/dynamic-bicycle](revisions/dynamic-bicycle/README.md).
It retains the LQR–PPO residual framework and uses the dynamic bicycle model, with explicitly documented stiffness-sign and sampled-LQR corrections. Newer experimental studies are listed below; the original revised manuscript has not been silently overwritten with their results.

## New experimental improvements

The [tuned fallback study](revisions/lans-robust-fallback/README.md) keeps the same vehicle model, controller blocks and 11 policy inputs. It changes LQR Q/R, training and actor initialization. On matched fresh cases, LANS completes the 200 ms DLC, step and sine tests; the 200 ms slalom test still fails. High-delay improvement comes from the retuned LQR fallback, not learned runtime suppression.

Against the original stronger PPO, original fast LQR and declared MPC, new LANS ranks first in 3 of 14 conditions and third by average condition rank. Its aggregate failure-penalized RMSE is lower than the earlier LANS, while some nominal errors are worse. The target of first place in every scenario is **not established**.

A retained no-learning-prior ablation has slightly lower aggregate penalized RMSE than trained LANS (0.2519 m versus 0.2556 m). A consistent additional benefit from PPO training is therefore not demonstrated; parameter/initialization gains must not be relabelled as learned superiority.

- [Full measured report](revisions/lans-robust-fallback/Robust_Fallback_Report.md) · [PDF](revisions/lans-robust-fallback/Robust_Fallback_Report.pdf) · [Word](revisions/lans-robust-fallback/Robust_Fallback_Report.docx)
- [Ranks against original stronger baselines](revisions/lans-robust-fallback/results/final_rankings_against_original_baselines.csv)
- [Matched original/new results](revisions/lans-robust-fallback/results/legacy_reference/matched_summary.csv)
- [Fixed-delay local certificates and their limits](revisions/lans-robust-fallback/results/certificate.json)
- [Preceding parameter-only study, including negative results](revisions/lans-improvement/Improvement_Report.md)

The two new studies retain all 18 final trained policies, all failures and raw traces. No queue/history features, observer, predictor or additional controller branch was added to LANS.

## Revised paper and reviewer response

- [Revised manuscript — PDF](revisions/dynamic-bicycle/deliverables/LANS_Revised_Manuscript.pdf)
- [Revised manuscript — editable Word](revisions/dynamic-bicycle/deliverables/LANS_Revised_Manuscript.docx)
- [Response to all 12 reviewer comments — PDF](revisions/dynamic-bicycle/deliverables/Response_to_Reviewer_1.pdf)
- [Response — editable Word](revisions/dynamic-bicycle/deliverables/Response_to_Reviewer_1.docx)
- [Change log](revisions/dynamic-bicycle/deliverables/Change_Log.md)
- [Roman Urdu explanation](revisions/dynamic-bicycle/deliverables/Read_Me_First_Roman_Urdu.md)
- [Complete downloadable revision package](packages/LANS_Dynamic_Bicycle_Revision_Package.zip)

## Experiments and reproducibility

The revision includes 9 trained policies, 1,806,336 total training steps, 1,225 evaluation episodes with raw traces, and 21,000 matched timing observations.

- [Reproduction instructions](revisions/dynamic-bicycle/README.md)
- [Experiment protocol](revisions/dynamic-bicycle/results/protocol.json)
- [Source code](revisions/dynamic-bicycle/code)
- [Trained policies and training logs](revisions/dynamic-bicycle/models)
- [All episode results](revisions/dynamic-bicycle/results/episodes_all.csv)
- [Summary statistics](revisions/dynamic-bicycle/results/summary.csv)
- [Paired comparisons](revisions/dynamic-bicycle/results/paired_differences.csv)
- [Integrity audit](revisions/dynamic-bicycle/results/audit.json)
- [SHA-256 manifest](revisions/dynamic-bicycle/results/integrity_manifest.json)

The experiments include negative results. LANS did not establish nominal superiority over LQR, and all 15 full-LANS runs failed under constant 200 ms delay on the complete command channel. Arbitrary-delay/global stability, 49-fold speedup and GPU/CPU energy-saving claims are not supported and have been withdrawn or restricted. No physical vehicle validation, certified WCET, or reviewer acceptance is claimed.

The ZIP archives originally committed at the repository root are retained as historical material. Their results should not be confused with the fresh dynamic-model revision.

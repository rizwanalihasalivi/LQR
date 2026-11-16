# LQR / LANS research materials

The latest revision is in [revisions/dynamic-bicycle](revisions/dynamic-bicycle/README.md).
It retains the LQR–PPO residual framework and uses the dynamic bicycle model, with explicitly documented stiffness-sign and sampled-LQR corrections.

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

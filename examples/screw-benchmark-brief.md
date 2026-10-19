# Delegated operator task: real offline screw-driving benchmark

Run the three enabled benchmark phases and explain whether the frozen detector
meets the declared dataset criteria. This is a pilot of Harness tool orchestration,
not a comparison between Agent architectures or an autonomous modeling benchmark.

Read `examples/screw-benchmark.json`. All scope choices are supplied; ask only if
a genuinely necessary fact remains missing. Never replace unavailable observations
with invented measurements. Write a compact `artifacts/benchmark-summary.md`.

## Source facts

- The pinned Zenodo v1.1.3 s01 archive has 5,000 operations on 100 workpieces.
- `labels.csv` uses `result_value` OK/NOK; all `class_value` values are zero and
  are not quality labels. Station decisions are not independent physical inspection.
- Grouped train/validation/test partitions contain 60/20/20 workpieces.
- Signals are graph time (s), torque (Nm), and angle (degrees), supported by the
  archive README. Local timestamps have no verified timezone.
- Some operations omit later steps and graphs contain duplicate times. Keep
  these operations; detection is after completion, not early prediction.
- The last torque sample can include tool release. The fixed window comparator
  uses peak torque, not the final sample or a copied controller quality code.
- The pinned archive embeds an MIT notice; later online documentation uses CC BY.
  Keep source attribution and the original archive notice.

## Delegated benchmark choices, not measured facts

Seed 42, a 60/20/20 grouped split, six feature summaries, median/MAD model,
scale floors, validation F1 threshold selection on the fixed grid, and acceptance
limits are declared in the specification. No held-out tuning is permitted.

Physical validation remains `not_established`. This is not the MERL Panda robot,
Festo hardware, a live OPC endpoint, or an OpenMES instance. An anomaly flag is a
learned prediction requiring investigation, not a verified repair diagnosis.

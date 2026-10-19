# Public Panda measurement identification

The operator authorizes an offline exploration using the already downloaded MERL Panda data.
Codex acts as delegated_operator to provide context and engineering assumptions.

First ask one concise question to confirm the scope, provisional acquisition mappings,
and predeclared acceptance criteria. This requests modelling context, not renewed permission.
Read examples/panda-identification.json. Do not claim that provisional assumptions
are confirmed sensor calibration or exact manufacturer/experimental robot properties.

After the delegated answer, execute the four exact allowed commands once each,
in order: identify ingest, model, fit, validate, in this session's artifacts/identification.
The CLI imports the pinned public real-robot experiments, generates actual MJCF,
fits bounded corrections using only the publisher's training excitations, then
scores the frozen model on held-out excitations. Measurements are not synthetic.
Accelerations are publisher-derived; nominal structure/inertias and payload mapping
remain assumptions. fit.json is part of the predictor: MJCF alone omits Coulomb/offset.

Do not read holdout CSVs, write CLI evidence, tune bounds/criteria from test results,
or invoke KUKA commands. Read dataset/model/fit/validation metadata and write
artifacts/identification-intake.md and artifacts/identification-summary.md.
The summary must include training/test run and sample counts, pooled and per-joint
held-out RMSE, active parameter bounds, the declared checks (including failures),
input/model/fit hashes, dataset license and actual generated file paths.
Describe dataset acceptance separately from physical_validation=not_established.
Finish only after actual CLI execution and summary files exist; a proposal is insufficient.

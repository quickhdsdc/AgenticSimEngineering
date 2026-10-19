# Delegated operator: Tensile2D physical model and benchmark

Complete the sealed local workflow with the vendored Harness. The inputs are
publisher FEM reference simulations, not physical measurements. Use the enabled
prepare, model, fit, score, export commands once each in that order.

Before model, write `artifacts/tensile-model.json` in your session with exactly:

```json
{"model":"plane_strain_tri3_residual","initial_poisson":0.25,"initial_young_scale":0.9,"elastic_fit_cases":4,"neighbors":[1,2,4,8,16],"geometry_weights":[0.25,1,4]}
```

This is a frozen, controlled modeling request. Generate the supported physical
model from the mesh, explicit CGNS boundary tags and pressure P. Identify Poisson
ratio and Young=p5*scale using training-only elastic cases. Compare elastic FEM,
FEM with learned residuals, and the historical kNN baselines. Select only with
validation data; score the independent local holdout once. Do not invent a
nonlinear hardening law, units, measurements, or material semantics for p1-p4.

Export the official 200-row pickle prediction bundle, with OOD cases separate.
This export expands residual training to all 500 publisher training cases after
local scoring. It has no local score for those hidden official targets and is not
an uploaded leaderboard submission. Report its verified format and hash, the
actual local scores and weakest output, and the remaining limits. Write
`artifacts/tensile-summary.md` outside the CLI-owned evidence folder and finish.

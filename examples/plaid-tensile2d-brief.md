# Delegated operator brief: PLAID Tensile2D

Run the pinned Tensile2D surrogate pilot through the existing Harness. All task
context here is operator guidance, not new measurements. No public leaderboard
submission or plant action is requested.

1. Use the exact enabled `plaid prepare` command and read its `dataset.json`.
2. Write `artifacts/plaid-recipe.json` yourself. The complete supported recipe is
   `{ "method": "inverse_distance_mesh_knn", "neighbors": [1,2,4,8,16],
   "geometry_weights": [0.25,1,4] }`. You may choose nonempty subsets. Explain
   your recipe and compute budget. No additional methods, code or fields are supported.
3. Use fit once. The worker fits feature normalization on 300 training cases,
   compares your candidate recipes on 100 validation cases, and freezes the winner.
4. Use score once on 100 private holdout cases. Do not change your recipe after
   scoring. Then export predictions for the 200 official test inputs plus 2 OOD inputs.
5. Write `artifacts/plaid-summary.md` with actual eight-output metrics, 1-neighbor
   baseline comparison, costs, paths and limitations; finish with the report path.

Inputs are load P, five unnamed material parameters, and publisher mesh geometry.
Targets are U1/U2/sig11/sig22/sig12 node fields and max_von_mises/max_U2_top/
max_sig22_top scalars. Do not infer units or material law from parameter names.
The data card does not declare physical units. Spatial interpolation uses three
nearest nodes, and scalar predictions are independent of the predicted fields.

Data are reference simulations from Safran, under CC-BY-SA-4.0. Labels from
publisher train_500 are split by simulation case with fixed seed 20261005. Public
test/OOD labels are unavailable. Local scores are not official hidden-test scores.
`predictions.npz` is a project format, not a verified competition submission.

This tests bounded recipe authoring, validation selection and evidence handling.
It does not test arbitrary FEM model generation, blind constitutive discovery,
PDE solving, official AAS mapping, real measurements or physical acceptance.
Report `physical_validation=not_established` and `official_hidden_score=null`.

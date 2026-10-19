# Delegated operator: AAS conformance and explicit mapping benchmark

Execute the semantic fixture benchmark using prepare, check, then score. This is
a closed-set checker-assisted interpretation pilot, not autonomous authoring or
generalization to unknown equipment. All examples are synthetic fixtures.

The CLI authors the AAS using the vendored Harness factories. Official
aas-test-engines 1.0.3 checks the AAS 3.0 Environment JSON in a separate environment.
Project URNs identify a custom submodel; it is not an official IDTA template.
Separate deterministic mapping checks compare the declared source facts with asset
identity, semantic IDs, units/conversion, frame, range, references and source kind.

Use the check findings to make one assessment per case, using this JSON shape:

```json
[
  {"case_id":"case-001","decision":"accept","codes":[]}
]
```

Include all sixteen case IDs exactly once. Decisions are accept, reject or clarify;
codes should reflect the checks. A structural AAS failure takes priority; otherwise
reject confirmed mismatches and clarify missing facts. Do not fabricate timezone,
provenance, external semantic definitions or real measurements. For missing
provenance and a required-but-unknown timezone, explain the questions an operator
would need to answer; no answer is supplied in this pilot.

Write `artifacts/semantic-submission.json`, then invoke the enabled score command
once. Do not edit CLI-owned artifacts or inspect private gold labels. Write
`artifacts/semantic-summary.md`, separating official conformance from project
mapping checks, comparing the schema-only baseline, and reporting the true score.
Do not claim a live BaSyx round trip, API certification, deployed mapping validation
or superior model generation from this pilot. Physical validation remains
`not_established`.

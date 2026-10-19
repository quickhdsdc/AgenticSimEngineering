"""Agent-facing inventory of the repository's CLI surfaces and their limits.

This module reports source capabilities, not deployment health. It deliberately
has no imports from A4S, a simulator, a device client, or an LLM provider.
"""


def catalog() -> dict:
    return {
        "schema_version": "1",
        "orchestrator": "a4s",
        "invocation": "A4S shell tool runs project-local CLIs; no second agent loop",
        "capabilities": [
            {
                "id": "tensile2d_physics_residual_benchmark",
                "cli": "ase tensile --help; ase-dialogue --case tensile2d-benchmark",
                "state": "integrated_reference_benchmark",
                "scope": "TRI3 model generation, training-only elastic parameter identification, frozen holdout comparisons and official-format 200-case pickle export",
                "limit": "Plastic corrections are learned; nonlinear constitutive law, official hidden server score and physical validation remain unestablished",
            },
            {
                "id": "plaid_tensile2d",
                "cli": "ase plaid --help; ase-dialogue --case plaid-tensile2d",
                "state": "integrated_reference_pilot",
                "scope": "Pinned 702-case dataset, bounded eight-output mesh surrogate, validation selection, sealed local holdout and test predictions",
                "limit": "Internal 300/100/100 split; no official hidden score, verified competition upload, FEM law discovery or physical validation; publisher units unspecified",
            },
            {
                "id": "native_model_generation_benchmarks",
                "cli": "ase modelbench --help; ase-dialogue --case factoryflow-benchmark; ase-dialogue --case chronobench-benchmark",
                "state": "integrated_reference_pilot",
                "scope": "Two FactoryFlow DES and two ChronoBench MBS constrained JSON authoring tasks with native reference scoring",
                "limit": "Headless adapted subset; not original published rubric, arbitrary Python generation, GUI acceptance or physical validation",
            },
            {
                "id": "semantic_mapping_benchmark",
                "cli": "ase semantic --help; ase-dialogue --case semantic-benchmark --help",
                "state": "integrated_fixture_subset",
                "scope": "Vendored A4S authoring, isolated official AAS 3.0 JSON checker and sixteen explicit mapping challenges",
                "limit": "Synthetic closed-set custom submodel; no official IDTA template, live API or deployed mapping acceptance",
            },
            {
                "id": "production_quality_benchmark",
                "cli": "ase benchmark --help; ase-dialogue --case screw-benchmark --help",
                "state": "integrated_offline_subset",
                "scope": "Pinned real screw-driving curves, grouped splits, frozen baseline fit and private held-out scoring",
                "limit": "Post-operation station-label prediction; no early warning, causal diagnosis, live OPC/MES or Agent ablation",
            },
            {
                "id": "mbs_public_measurement_identification",
                "cli": "ase identify --help; ase-dialogue --case panda-identification --help",
                "state": "integrated_offline_subset",
                "scope": "Hash-verified MERL Panda import, nominal MJCF, bounded joint corrections and disjoint real-experiment scoring",
                "limit": "Provisional acquisition/model mappings; dataset acceptance is separate from physical calibration; no full inertial or forward-control validation",
            },
            {
                "id": "mbs_synthetic_identification",
                "cli": "ase calibrate --help; ase-dialogue --case kuka-calibration --help",
                "state": "integrated_synthetic_subset",
                "scope": "Pinned KUKA joint2 synthetic torque runs, generated MJCF, damping/armature fitting and disjoint holdout scoring",
                "limit": "Same MuJoCo model family; this KUKA case has no measured-data calibration or full inertial/physical accuracy claim",
            },
            {
                "id": "agent_workflow",
                "cli": "a4s run --help",
                "state": "component_available",
                "scope": "A4S planning, tool dispatch, approval gate, and ledger",
                "limit": "Online model and configured AAS endpoint required for a4s run",
            },
            {
                "id": "engineering_study",
                "cli": "ase study --help",
                "state": "integrated_offline_subset",
                "scope": "Registered DES, Newton MBS, and OpenRadioss FEM candidate evaluation",
                "limit": "Fixed shell smoke checks do not verify autonomous Agent planning",
            },
            {
                "id": "aas_semantics",
                "cli": "aas --help",
                "state": "component_available",
                "scope": "AAS shell, submodel, template, matching, and validation commands",
                "limit": "Unified Study-to-AAS mapping and BaSyx round trip remain unverified",
            },
            {
                "id": "opc_ua",
                "cli": "aas opc --help",
                "state": "component_available",
                "scope": "OPC UA browse, read, write, and method commands",
                "limit": "Live connectivity and writes depend on deployment and approval",
            },
            {
                "id": "openmes",
                "cli": "mes --help",
                "state": "component_available",
                "scope": "OpenMES business operations through the A4S CLI",
                "limit": "Current OpenMES API and deployment compatibility remain unverified here",
            },
            {
                "id": "festo_monitoring",
                "cli": "python -m a4s.machines.festo_cp_lab.recorder --help; a4s serve --help",
                "state": "integrated_local_subset",
                "scope": "Festo AAS bindings, OPC quality/timestamps, persistent fault episodes and live/simulation UI",
                "limit": "Loopback OPC verified; physical endpoints/calibration unverified; polling may miss short events",
            },
            {
                "id": "festo_fault_investigation",
                "cli": "python -m a4s.machines.festo_cp_lab.fault_context --help; python -m a4s.machines.festo_cp_lab.knowledge --help",
                "state": "integrated_local_subset",
                "scope": "Existing Harness receives scoped fault evidence and retrieves asset context and PDF passages",
                "limit": "Scripted provider dispatch verified; online diagnosis unverified; automatic handoff requires open browser",
            },
            {
                "id": "festo_maintenance_replanning",
                "cli": "mes maintenance --help; python -m a4s.machines.festo_cp_lab.replanning --help",
                "state": "integrated_fixture_subset",
                "scope": "Maintenance and scheduling preview, approval, execution/readback contracts; production UI",
                "limit": "Business tests use fixtures; configured OpenMES/Docker and live deployment require separate acceptance",
            },
            {
                "id": "model_authoring",
                "cli": "a4s extract --help; aas match --help; aas populate --help",
                "state": "component_subset",
                "scope": "Document extraction and AAS population workflow",
                "limit": "Autonomous cross-domain simulation model authoring is not integrated",
            },
            {
                "id": "visualization_3d",
                "cli": "ase visualize --help",
                "state": "integrated_offline_subset",
                "scope": "FIFO DES playback and original KUKA/G1 meshes driven by recorded Newton body poses",
                "limit": "Read-only reference simulation; FEM, OpenUSD and live state mapping remain unintegrated",
            },
        ],
    }

"""Sample7 crash-form model over vendored OpenRadioss; no heavy startup imports."""
import hashlib
from dataclasses import asdict
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from agenticsim.domain import BackendDescriptor, EngineeringTask, Evaluation, ModelDefinition, Observable, number, require

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "components/fem/openform_poc"
PREFIX = "urn:ase:stamping:"


def fingerprint(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": digest}


def load_formpipe():
    name = "ase_vendored_formpipe"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SOURCE / "formpipe/__init__.py",
                                                    submodule_search_locations=[str(SOURCE / "formpipe")])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    from importlib import import_module
    return tuple(import_module(name + "." + part) for part in ("spec", "quicklink", "openradioss"))


def validate_final_frame(text, expected_ms):
    match = re.search(r"^TIME\s+1\s+1\s+double\s*\n\s*([\d.eE+-]+)", text, re.MULTILINE)
    require(match is not None, "final VTK lacks simulation time")
    actual = float(match[1])
    require(abs(actual - expected_ms) <= max(1e-5, expected_ms * 1e-5), "incomplete final simulation time")
    return actual


class StampingFEM:
    model = ModelDefinition("fem-sample7", "1", (
        ("density", PREFIX + "density", "kg/m3"),
        ("form_time", PREFIX + "form-time", "s"),
        ("dwell", PREFIX + "dwell", "s")),
        tuple((PREFIX + name, unit) for name, unit in (
            ("min-thickness", "m"), ("global-max-thinning", "%"),
            ("interior-max-thinning", "%"), ("p99-thinning", "%"))),
        ("Sample7 AutoForm-developed blank and formed surface; rigid tools, deformable shell",
         "Crash-form approximation; no drawbeads, blankholder, springback or FLD prediction",
         "Material curve decoded heuristically from MTB; AutoForm is a reference simulation",
         "Interior excludes elements touching a boundary node; free-edge artifacts remain"))
    backend = BackendDescriptor("openradioss-win64", "1", ("evaluate",))

    def validate(self, task):
        task.validate(self.model, self.backend)
        for name, low, high in (("density", 100, 20000), ("form_time", .001, .1), ("dwell", 0, .01)):
            values = task.document["inputs"][name]["values"]
            require(len(values) == 1 and low <= values[0] <= high, f"invalid {name} range/shape")
        execution = task.document.get("execution")
        require(isinstance(execution, dict) and set(execution) == {"algorithm", "dt_min_s", "threads"},
                "explicit FEM execution settings required")
        require(execution["algorithm"] == "explicit-shell", "unsupported FEM algorithm")
        require(number(execution["dt_min_s"]) and 0 < execution["dt_min_s"] <= 1e-7,
                "mass-scaling target must be in (0, 1e-7] seconds")
        require(type(execution["threads"]) is int and 1 <= execution["threads"] <= 16, "invalid threads")
        for candidate in task.document["candidates"]:
            require(set(candidate) == {"id", "parameters"} and isinstance(candidate["parameters"], dict)
                    and set(candidate["parameters"]) == {"friction", "clearance_ratio"}, "invalid FEM candidate")
            for name, low, high in (("friction", 0, .5), ("clearance_ratio", 1, 1.5)):
                value = candidate["parameters"][name]
                require(isinstance(value, dict) and set(value) == {"value", "unit"}
                        and value["unit"] == "1" and number(value["value"])
                        and low <= value["value"] <= high, f"invalid {name}")

    def evaluate(self, task: EngineeringTask, candidate):
        self.validate(task)
        require(candidate in task.document["candidates"], "candidate not in task")
        require(os.name == "nt", "vendored OpenRadioss binaries require Windows")
        import numpy as np
        import scipy
        specs, quicklink, solver = load_formpipe()
        directory = Path(os.environ.get("ASE_RUN_DIRECTORY", ROOT / "work/runs/workers" / uuid4().hex)).resolve()
        require(directory.is_relative_to(ROOT / "work/runs/workers"), "worker directory must remain inside project runs")
        directory.mkdir(parents=True, exist_ok=True)
        work = directory / "fem"
        work.mkdir(exist_ok=False)  # Never parse a previous run as a new solve.
        values = {k: v["values"][0] for k, v in task.document["inputs"].items()}
        params = {k: v["value"] for k, v in candidate["parameters"].items()}
        numerical = task.document["execution"]
        process = specs.ProcessSpec(**params, form_time_ms=values["form_time"] * 1000,
                                    dwell_ms=values["dwell"] * 1000, dt_min_ms=numerical["dt_min_s"] * 1000)
        assets = SOURCE / "sample7_run/quicklink"
        spec, reference = quicklink.load(assets, process=process, density=values["density"] * 1e-6)
        # Legacy loader treats 0.1 as a default sentinel; preserve the explicit task value.
        spec.process.friction = params["friction"]
        warnings = spec.validate()
        solver.write_deck(spec, work)
        env = solver._env()
        env["OMP_NUM_THREADS"] = str(numerical["threads"])
        binaries = solver.OR_ROOT / "exec"
        for phase, args in (("starter", ["-i", "FORM_0000.rad", "-np", "1"]),
                            ("engine", ["-i", "FORM_0001.rad"])):
            with (work / (phase + ".log")).open("w", encoding="utf-8") as log:
                result = subprocess.run([str(binaries / (phase + "_win64.exe")), *args],
                                        cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=task.document["budget"]["timeout_s"])
            output = (work / (phase + ".log")).read_text(encoding="utf-8", errors="replace")
            markers = ("NORMAL TERMINATION", "TERMINATION WITH WARNING") if phase == "starter" else ("NORMAL TERMINATION",)
            require(result.returncode == 0 and "ERROR TERMINATION" not in output
                    and any(marker in output for marker in markers), f"{phase} failed; see {work}")
        anims = sorted(p for p in work.iterdir() if p.name.startswith("FORMA") and p.name[-3:].isdigit())
        require(bool(anims), "solver produced no animation frames")
        vtk = anims[-1].with_suffix(".vtk")
        with vtk.open("w") as handle:
            subprocess.run([str(binaries / "anim_to_vtk_win64.exe"), str(anims[-1])],
                           stdout=handle, stderr=subprocess.PIPE, check=True,
                           timeout=task.document["budget"]["timeout_s"])
        final_ms = validate_final_frame(vtk.read_text(), (values["form_time"] + values["dwell"]) * 1000)
        _, cells, cell_data = solver.parse_vtk(vtk)
        ids = cell_data["ELEMENT_ID"]
        shell_ids = ids[np.array([len(cell) >= 3 for cell in cells]) & (ids >= 1) & (ids <= spec.blank.n_elems)]
        require(np.array_equal(np.sort(shell_ids), np.arange(1, spec.blank.n_elems + 1)),
                "missing or duplicate blank element IDs")
        result = solver.parse(work, spec)
        fields = {k: result[k] for k in ("element_thickness", "element_thinning_pct", "element_plastic_strain", "edge_element_mask")}
        require(result["n_elements"] == spec.blank.n_elems
                and all(np.isfinite(v).all() for v in fields.values())
                and (fields["element_thickness"] > 0).all(), "incomplete/nonfinite FEM fields")
        np.savez_compressed(work / "fields.npz", points_mm=spec.blank.points3d, triangles=spec.blank.tris,
                            element_ids=np.arange(1, spec.blank.n_elems + 1), **fields)
        ref = np.asarray(reference["element_thinning_pct"])
        require(ref.shape == fields["element_thinning_pct"].shape and np.isfinite(ref).all(), "reference field mismatch")
        np.savez_compressed(work / "reference-fields.npz", element_thinning_pct=ref,
                            element_ids=np.arange(1, spec.blank.n_elems + 1))
        comparison = {"source": "reference_simulation", "reference": "AutoForm",
                      "thinning_mae_percentage_points": float(np.mean(np.abs(ref - fields["element_thinning_pct"]))),
                      "reference_global_max_thinning_pct": float(ref.max()),
                      "reference_p99_thinning_pct": float(np.percentile(ref, 99)),
                      "physical_validation": "not_established"}
        (work / "reference-comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        metadata = {"frame": "press: AutoForm (X,Y,Z) -> (X,-Z,Y); punch motion -Z",
                    "mesh": "fields.npz points_mm is initial blank; VTK contains final deformed mesh and tools",
                    "field_units": {"element_thickness": "mm", "element_thinning_pct": "%", "element_plastic_strain": "1"},
                    "element_mapping": "deck SH3N id = triangles row + 1; connectivity zero-based",
                    "warnings": warnings, "quasi_static_ratio_estimate": spec.quasi_static_ratio(),
                    "mass_scaling_factor_estimate": spec.mass_scaling_factor(),
                    "density_g_mm3": spec.material.density, "nominal_thickness_mm": spec.material.thickness,
                    "material": asdict(spec.material), "material_units": "g/mm/ms, MPa; strain dimensionless",
                    "material_mapping": "legacy heuristic MTB decode; fallback E=70000 MPa, nu=.33, r=1 if absent; flow extrapolated to strain 2",
                    "final_time_ms": final_ms, "convergence_validation": "not_established"}
        diagnostics = re.findall(r"NC=\s*(\d+)\s+T=\s*([\d.E+-]+)\s+DT=\s*([\d.E+-]+)\s+ERR=\s*([\d.E+-]+)%\s+DM/M=\s*([\d.E+-]+)", output)
        require(bool(diagnostics), "missing solver numerical diagnostics")
        cycle, time_ms, dt_ms, error, mass = diagnostics[-1]
        metadata["last_reported_diagnostics"] = {"cycle": int(cycle), "time_ms": float(time_ms),
            "dt_ms": float(dt_ms), "energy_error_pct": float(error), "added_mass_ratio": float(mass)}
        (work / "field-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        observations = tuple(Observable(PREFIX + key, unit, (float(result[source]) * scale,))
                             for key, unit, source, scale in (
            ("min-thickness", "m", "min_thickness_mm", .001),
            ("global-max-thinning", "%", "max_thinning_pct", 1),
            ("interior-max-thinning", "%", "interior_max_thinning_pct", 1),
            ("p99-thinning", "%", "p99_thinning_pct", 1)))
        evidence = Evaluation(observations, "reference_simulation", {
            "mapping_version": "1", "numpy": np.__version__, "scipy": scipy.__version__,
            "adapter": fingerprint(Path(__file__)), "numerical": numerical,
            "sources": [fingerprint(p) for p in sorted(assets.iterdir()) if p.is_file()],
            "implementation": [fingerprint(p) for p in sorted((SOURCE / "formpipe").glob("*.py"))],
            "binaries": [fingerprint(binaries / name) for name in
                         ("starter_win64.exe", "engine_win64.exe", "anim_to_vtk_win64.exe")],
            "artifacts": [fingerprint(p) for p in sorted(work.iterdir()) if p.is_file()],
            "reference_comparison": comparison, "model_metadata": metadata})
        evidence.validate(self.model)
        return evidence

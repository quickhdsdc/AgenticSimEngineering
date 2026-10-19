"""Versioned, dependency-free engineering evaluation contracts."""
from dataclasses import asdict, dataclass
import json
from math import isfinite

SCHEMA = "1.0"
SOURCES = {"synthetic_fixture", "measurement", "reference_simulation", "learned_prediction"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value) -> bool:
    return type(value) in (int, float) and isfinite(value)


@dataclass(frozen=True)
class Observable:
    semantic_id: str
    unit: str
    values: tuple[float, ...]
    axis: tuple[float, ...] | None = None
    axis_unit: str | None = None

    def validate(self):
        require(bool(self.semantic_id) and bool(self.unit), "explicit semantics and units required")
        require(bool(self.values) and all(number(v) for v in self.values), "nonempty finite observations required")
        if self.axis is None:
            require(len(self.values) == 1 and self.axis_unit is None, "scalar observation requires one value")
        else:
            require(bool(self.axis_unit) and len(self.axis) == len(self.values), "axis shape/unit mismatch")
            require(all(number(v) for v in self.axis), "finite axis required")
            require(all(b > a for a, b in zip(self.axis, self.axis[1:])), "axis must increase")


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    version: str
    input_semantics: tuple[tuple[str, str, str], ...]
    output_semantics: tuple[tuple[str, str], ...]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class BackendDescriptor:
    backend_id: str
    version: str
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class Evaluation:
    observations: tuple[Observable, ...]
    source: str
    provenance: dict

    def validate(self, model: ModelDefinition):
        require(self.source in SOURCES, "unknown result source")
        for observation in self.observations:
            observation.validate()
        actual = [(o.semantic_id, o.unit) for o in self.observations]
        require(len(actual) == len(set(actual)), "duplicate observations")
        require(set(actual) == set(model.output_semantics), "output semantic/unit mapping mismatch")
        require(bool(self.provenance), "result provenance required")

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, data):
        return cls(tuple(Observable(o["semantic_id"], o["unit"], tuple(o["values"]),
                                    None if o.get("axis") is None else tuple(o["axis"]),
                                    o.get("axis_unit")) for o in data["observations"]),
                   data["source"], data["provenance"])


@dataclass(frozen=True)
class EngineeringTask:
    """Finite candidate comparison v1; no arbitrary expression evaluation."""
    document: dict

    def validate(self, model: ModelDefinition, backend: BackendDescriptor):
        d = self.document
        require(isinstance(d, dict), "task must be an object")
        fields = {"schema_version", "task_id", "kind", "model", "backend", "required_capabilities", "context",
                  "inputs", "candidates", "budget", "objective", "constraints", "acceptance"}
        require(fields <= set(d) <= fields | {"execution"}, "task fields must match schema 1.0")
        for field in ("model", "backend", "context", "inputs", "budget", "objective", "acceptance"):
            require(isinstance(d[field], dict), f"{field} must be an object")
        require(isinstance(d["required_capabilities"], list)
                and all(isinstance(v, str) for v in d["required_capabilities"]), "capabilities must be strings")
        require(isinstance(d["constraints"], list) and all(isinstance(c, dict) for c in d["constraints"]),
                "constraints must be objects")
        require(d.get("schema_version") == SCHEMA, "unsupported task schema")
        require(d.get("kind") == "candidate_comparison", "unsupported task kind")
        require(isinstance(d.get("task_id"), str) and bool(d["task_id"]), "task_id required")
        require(d.get("model") == {"id": model.model_id, "version": model.version}, "unknown model/version")
        require(d.get("backend") == {"id": backend.backend_id, "version": backend.version}, "unknown backend/version")
        require("evaluate" in backend.capabilities, "backend cannot evaluate")
        require(set(d.get("required_capabilities", ["evaluate"])) <= set(backend.capabilities), "unsupported capability")
        require(d.get("context", {}).get("source") in SOURCES, "context source required")
        require(bool(d["context"].get("description")), "context description required")
        inputs = d.get("inputs", {})
        require(set(inputs) == {name for name, _, _ in model.input_semantics}, "input names mismatch")
        for name, semantic, unit in model.input_semantics:
            value = inputs[name]
            require(isinstance(value, dict) and set(value) == {"semantic_id", "unit", "values"}, "invalid input fields")
            require(value.get("semantic_id") == semantic and value.get("unit") == unit, "input semantic/unit mismatch")
            require(isinstance(value.get("values"), list) and bool(value["values"])
                    and all(number(v) for v in value["values"]), "finite input arrays required")
        candidates = d.get("candidates", [])
        require(isinstance(candidates, list) and bool(candidates), "candidates required")
        require(all(isinstance(c, dict) for c in candidates), "candidates must be objects")
        ids = [c.get("id") for c in candidates]
        require(all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids)), "unique candidate IDs required")
        budget = d.get("budget", {})
        require(type(budget.get("max_evaluations")) is int and 0 < len(candidates) <= budget["max_evaluations"] <= 1000,
                "evaluation budget exceeded (maximum 1000)")
        require(number(budget.get("timeout_s")) and 0 < budget["timeout_s"] <= 3600, "invalid per-run timeout")
        objective = d.get("objective", {})
        require(objective.get("direction") in {"minimize", "maximize"}, "objective direction required")
        for criterion in [objective, *d.get("constraints", [])]:
            require((criterion.get("semantic_id"), criterion.get("unit")) in model.output_semantics,
                    "criterion semantic/unit mismatch")
            require(criterion.get("reduction") == "scalar", "only explicit scalar criteria supported")
        for constraint in d.get("constraints", []):
            require(constraint.get("operator") in {"<=", ">="} and number(constraint.get("limit")), "invalid constraint")
        require(d.get("acceptance", {}).get("scope") == "model_constraints_only",
                "v1 only supports model-constraint acceptance, not physical validation")


def assess(task: EngineeringTask, result: Evaluation) -> dict:
    """Engineering acceptance is separate from execution success."""
    observations = {o.semantic_id: o for o in result.observations}
    residuals = []
    for c in task.document.get("constraints", []):
        o = observations[c["semantic_id"]]
        require(o.axis is None and len(o.values) == 1 and o.unit == c["unit"], "criterion must be a matching scalar")
        residual = o.values[0] - c["limit"] if c["operator"] == "<=" else c["limit"] - o.values[0]
        residuals.append({"semantic_id": o.semantic_id, "unit": o.unit, "violation": max(0., residual)})
    objective = task.document["objective"]
    o = observations[objective["semantic_id"]]
    require(o.axis is None and len(o.values) == 1 and o.unit == objective["unit"], "objective must be a matching scalar")
    return {"status": "accepted" if all(r["violation"] == 0 for r in residuals) else "rejected",
            "scope": "model_constraints_only", "objective": o.values[0], "residuals": residuals,
            "physical_validation": "not_established"}

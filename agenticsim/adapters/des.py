"""Vendored SimPy reference adapter; heavy imports occur only at execution."""
import hashlib
import importlib.util
import platform
from pathlib import Path

from agenticsim.domain import BackendDescriptor, EngineeringTask, Evaluation, ModelDefinition, Observable, number, require

PREFIX = "urn:ase:queue:"
SOURCE = Path(__file__).resolve().parents[2] / "components/des/src/gradientshop/reference/simpy_engine.py"


class SingleServer:
    model = ModelDefinition("fifo-single-server", "1", (
        ("interarrival", PREFIX + "interarrival", "s"),
        ("service", PREFIX + "service", "s")),
        ((PREFIX + "makespan", "s"), (PREFIX + "departure", "s")),
        ("FIFO, one server, empty initial queue", "Fixed sample path; service_scale changes service times, not capacity"))
    backend = BackendDescriptor("vendored-simpy", "1", ("evaluate",))

    def validate(self, task: EngineeringTask):
        task.validate(self.model, self.backend)
        require("execution" not in task.document, "DES reference does not accept numerical execution settings")
        arrays = [task.document["inputs"][name]["values"] for name in ("interarrival", "service")]
        require(len(arrays[0]) == len(arrays[1]) <= 10000, "equal input lengths required, maximum 10000 jobs")
        require(all(v >= 0 for v in arrays[0]) and all(v > 0 for v in arrays[1]), "invalid arrival/service durations")
        for criterion in [task.document["objective"], *task.document["constraints"]]:
            require(criterion["semantic_id"] == PREFIX + "makespan", "departure is not a scalar criterion")
        for c in task.document["candidates"]:
            require(set(c) == {"id", "parameters"} and isinstance(c["parameters"], dict)
                    and set(c["parameters"]) == {"service_scale"}, "unknown decision parameter")
            scale = c["parameters"]["service_scale"]
            require(isinstance(scale, dict) and set(scale) == {"value", "unit"} and scale["unit"] == "1" and number(scale["value"])
                    and 0.1 <= scale["value"] <= 10, "service_scale requires dimensionless value in [0.1, 10]")

    def evaluate(self, task: EngineeringTask, candidate: dict) -> Evaluation:
        self.validate(task)
        require(candidate in task.document["candidates"], "candidate not in validated task")
        import numpy
        import simpy
        # Exact in-repository file; no sibling paths or ambient package lookup.
        spec = importlib.util.spec_from_file_location("ase_vendored_simpy", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        inputs = task.document["inputs"]
        service = [v * candidate["parameters"]["service_scale"]["value"] for v in inputs["service"]["values"]]
        departure = tuple(float(v) for v in module.simpy_single_server(inputs["interarrival"]["values"], service))
        result = Evaluation((Observable(PREFIX + "makespan", "s", (max(departure),)),
                             Observable(PREFIX + "departure", "s", departure, tuple(range(len(departure))), "job_index")),
                            "reference_simulation", {"source_file": SOURCE.relative_to(SOURCE.parents[5]).as_posix(),
                            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                            "numpy": numpy.__version__, "simpy": simpy.__version__, "mapping_version": "1",
                            "python": platform.python_version(), "platform": platform.platform(),
                            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "randomness": "none; explicit sample path", "input_source": task.document["context"]["source"]})
        result.validate(self.model)
        return result

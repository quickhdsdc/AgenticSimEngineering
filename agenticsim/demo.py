"""Analytical cooling fixture; synthetic observations, no plant validation claim."""
from dataclasses import asdict
from math import exp, isfinite
from .contracts import ModelCard, Parameter, Quantity, Request, Trajectory, SimulationBackend, require_capabilities


class CoolingBackend:
    card = ModelCard(
        model_id="urn:ase:demo:cooling", version="1", domain="thermal-ode",
        backend="stdlib-analytical-v1", fidelity="analytical-lumped-demo",
        capabilities=("simulate",),
        parameters=(Parameter("rate", "urn:ase:demo:cooling-rate", "1/s", 0.001, 1.0),),
        assumptions=("Uniform temperature; constant ambient 293.15 K; initial 353.15 K",
                     "Synthetic contract fixture, not a validated remanufacturing model"),
    )

    def simulate(self, request: Request) -> Trajectory:
        self.card.validate(request.parameters)
        request.validate_times()
        rate = request.parameters["rate"].value
        return Trajectory(request.times_s,
                          tuple(293.15 + 60 * exp(-rate * t) for t in request.times_s),
                          "urn:ase:demo:temperature", "K")


def calibrate_grid(backend: SimulationBackend, observations: Trajectory,
                   parameter: str, candidates: tuple[Quantity, ...]) -> dict:
    """Finite candidate search, single parameter; optimiser separate from simulation."""
    require_capabilities(backend.card, {"simulate"})
    Request({}, observations.times_s).validate_times()
    if len(observations.values) != len(observations.times_s) or not all(map(isfinite, observations.values)):
        raise ValueError("observations must be finite and match time coordinates")
    if not candidates:
        raise ValueError("candidate set is empty")
    # Validate all candidates before performing any evaluations.
    for candidate in candidates:
        backend.card.validate({parameter: candidate})
    history = []
    for candidate in candidates:
        predicted = backend.simulate(Request({parameter: candidate}, observations.times_s))
        if (predicted.unit != observations.unit or predicted.observable_id != observations.observable_id
                or predicted.times_s != observations.times_s or len(predicted.values) != len(observations.values)
                or not all(map(isfinite, predicted.values))):
            raise ValueError("prediction and observation contracts do not match")
        mse = sum((a - b) ** 2 for a, b in zip(predicted.values, observations.values)) / len(observations.values)
        history.append({"parameter": asdict(candidate), "mse": mse})
    return {"method": "finite-grid-search", "loss_unit": f"({observations.unit})^2",
            "best": min(history, key=lambda row: row["mse"]), "evaluations": history,
            "claim": "Best among supplied candidates; no global optimum or identifiability claim"}


def experiment() -> dict:
    backend = CoolingBackend()
    truth = Request({"rate": Quantity(0.12, "1/s")}, tuple(float(t) for t in range(21)))
    observations = backend.simulate(truth)
    calibration = calibrate_grid(backend, observations, "rate",
                                tuple(Quantity(i / 100, "1/s") for i in range(1, 31)))
    return {"schema_version": "0.1", "kind": "synthetic-calibration-demo",
            "model": backend.card.to_dict(), "observation_source": "synthetic, same analytical model",
            "generation_request": asdict(truth), "observations": asdict(observations),
            "calibration": calibration}

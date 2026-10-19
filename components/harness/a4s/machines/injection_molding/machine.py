"""A simulated injection moulding machine, driven by a real production run.

There is no injection moulding machine on this desk, and the interesting parts
of the harness -- Operation delegation, the approval gate, provenance for a
value that came off a machine rather than out of a document -- need one that
behaves like the real thing rather than one that returns constants.

So the machine replays a published production run. 1,167 consecutive shots of
a real machine, each with its process parameters and the weight of the part it
produced, cycling on a clock. Reading `MaxInjectionPressure` twice a second
apart gives two different numbers because they are two different shots, and the
numbers covary the way a real process does, because they came from one.

Two things this makes honest that a constant-returning stub would not:

**Setpoints move the process, they do not replace it.** Asking for a barrel
temperature of 250 does not make `zylinderheizzone_1` read 250; it shifts the
replayed value towards the setpoint over the next few shots, the way a heater
does. An agent that sets a value and immediately reads it back sees the
machine still moving, which is the situation it will meet on a shopfloor.

**Quality is predicted, not looked up.** The weight for a *hypothetical* set of
parameters comes from a ridge regression fitted on the run at startup, not from
the row it was taken from -- so an agent can ask "what would happen if" and get
an answer that is wrong in the way a model is wrong rather than wrong in the
way a lie is wrong. `quality_model_report()` states its fit, so nobody has to
guess how much to trust it.

Data: Bogedale, L.; Doerfel, S.; Schrodt, A.; Heim, H.-P. "Online Prediction of
Molded Part Quality in the Injection Molding Process Using High-Resolution Time
Series." Polymers 2023, 15, 978. CC-BY 4.0. See `data/SOURCE.md`.
"""

from __future__ import annotations

import csv
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DATASET = DATA_DIR / "ds1_scalar_and_quality.csv"

# The published columns, and the English names the AAS uses for them. The
# dataset is German; an idShort has to be [A-Za-z][A-Za-z0-9_]* by AASd-002 and
# ought to be readable, so the mapping lives here rather than being invented at
# each call site.
COLUMNS: Dict[str, str] = {
    "cycle_counter": "CycleCounter",
    "cycle_time": "CycleTime",
    "umschaltvolumen": "SwitchoverVolume",
    "maximaler_spritzdruck": "MaxInjectionPressure",
    "umschaltspritzdruck": "SwitchoverPressure",
    "einspritzzeit": "InjectionTime",
    "massepolster": "MeltCushion",
    "dosierzeit": "DosingTime",
    "zylinderheizzone_1": "BarrelZone1",
    "zylinderheizzone_2": "BarrelZone2",
    "zylinderheizzone_3": "BarrelZone3",
    "zylinderheizzone_4": "BarrelZone4",
    "zylinderheizzone_5": "BarrelZone5",
    "zylinderheizzone_6": "BarrelZone6",
    "zylinderheizzone_7": "BarrelZone7",
    "zylinderheizzone_8": "BarrelZone8",
    "werkzeugheizkreis_1": "MouldHeatingCircuit1",
    "oeil_temperatur_ist": "OilTemperature",
    "staudruck_ist": "BackPressure",
    "weight": "PartWeight",
    "distanceA": "PartDimensionA",
}

# The unit each reading carries, so a value written into an AAS can be checked
# against the unit the element declares. Empty means dimensionless.
UNITS: Dict[str, str] = {
    "CycleCounter": "", "CycleTime": "s", "SwitchoverVolume": "cm3",
    "MaxInjectionPressure": "bar", "SwitchoverPressure": "bar",
    "InjectionTime": "s", "MeltCushion": "cm3", "DosingTime": "s",
    "BarrelZone1": "degC", "BarrelZone2": "degC", "BarrelZone3": "degC",
    "BarrelZone4": "degC", "BarrelZone5": "degC", "BarrelZone6": "degC",
    "BarrelZone7": "degC", "BarrelZone8": "degC",
    "MouldHeatingCircuit1": "degC", "OilTemperature": "degC",
    "BackPressure": "bar", "PartWeight": "g", "PartDimensionA": "",
}

# Readings an operator may command. Everything else is measured, and asking to
# set it is refused rather than silently accepted -- a machine that pretends to
# obey is worse than one that says no.
SETTABLE = (
    "SwitchoverVolume", "MaxInjectionPressure", "SwitchoverPressure",
    "BackPressure", "BarrelZone1", "BarrelZone2", "BarrelZone3", "BarrelZone4",
    "BarrelZone5", "BarrelZone6", "BarrelZone7", "BarrelZone8",
    "MouldHeatingCircuit1",
)

# What the quality model is given. Deliberately the process parameters only:
# feeding it the cycle counter would let it memorise the run.
FEATURES = (
    "SwitchoverVolume", "MaxInjectionPressure", "SwitchoverPressure",
    "InjectionTime", "MeltCushion", "DosingTime",
    "BarrelZone1", "BarrelZone8", "MouldHeatingCircuit1",
    "OilTemperature", "BackPressure",
)
TARGET = "PartWeight"

# How fast a setpoint is approached, per shot. A barrel heater does not jump,
# and an agent that sets a temperature and reads it back immediately should see
# that it has not arrived yet.
APPROACH_PER_SHOT = 0.35


class MachineError(RuntimeError):
    """The machine refuses: a bad setpoint, an unknown reading, a stopped machine."""


def _number(raw: str) -> Optional[float]:
    """Parse the dataset's German decimal comma."""
    text = (raw or "").strip().strip('"')
    if not text:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def load_shots(path: Path = DEFAULT_DATASET) -> List[Dict[str, float]]:
    """Read the run into a list of shots keyed by the English names."""
    if not path.exists():
        raise MachineError(f"no dataset at {path}")
    shots: List[Dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            shot = {
                COLUMNS[key]: value
                for key, raw in row.items()
                if (key in COLUMNS) and (value := _number(raw)) is not None
            }
            if shot:
                shots.append(shot)
    if not shots:
        raise MachineError(f"{path.name} yielded no usable shots")
    return shots


# ---------------------------------------------------------------------------
# the quality model
# ---------------------------------------------------------------------------

@dataclass
class QualityModel:
    """Ridge regression from process parameters to part weight.

    Ridge and not a gradient-boosted ensemble: the point here is a defensible
    answer to "what would this parameter set produce", not a competitive score
    on the source paper's benchmark. Ridge is a dozen lines of numpy, needs no
    extra dependency, and its coefficients can be read -- which matters when the
    thing it is standing in for is a claim about a physical process.
    """

    features: List[str]
    target: str
    coefficients: Any = None
    intercept: float = 0.0
    mean: Any = None
    scale: Any = None
    r_squared: float = 0.0
    rmse: float = 0.0
    trained_on: int = 0

    @classmethod
    def fit(cls, shots: List[Dict[str, float]], alpha: float = 1.0) -> "QualityModel":
        import numpy as np

        usable = [s for s in shots if all(f in s for f in FEATURES) and TARGET in s]
        if len(usable) < len(FEATURES) * 2:
            raise MachineError(
                f"only {len(usable)} complete shots; not enough to fit {len(FEATURES)} features"
            )
        matrix = np.array([[s[f] for f in FEATURES] for s in usable], dtype="float64")
        target = np.array([s[TARGET] for s in usable], dtype="float64")

        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale == 0] = 1.0
        scaled = (matrix - mean) / scale

        # Ridge in closed form. The intercept is not penalised, so it is fitted
        # as the target mean and left out of the regularised system.
        centred = target - target.mean()
        gram = scaled.T @ scaled + alpha * np.eye(scaled.shape[1])
        coefficients = np.linalg.solve(gram, scaled.T @ centred)

        predicted = scaled @ coefficients + target.mean()
        residual = target - predicted
        total = target - target.mean()
        r_squared = 1.0 - float(residual @ residual) / float(total @ total or 1.0)
        rmse = math.sqrt(float(residual @ residual) / len(residual))

        return cls(
            features=list(FEATURES), target=TARGET, coefficients=coefficients,
            intercept=float(target.mean()), mean=mean, scale=scale,
            r_squared=round(r_squared, 4), rmse=round(rmse, 4), trained_on=len(usable),
        )

    def predict(self, parameters: Dict[str, float]) -> float:
        import numpy as np

        missing = [f for f in self.features if f not in parameters]
        if missing:
            raise MachineError(f"missing process parameters: {', '.join(missing)}")
        row = np.array([float(parameters[f]) for f in self.features], dtype="float64")
        scaled = (row - self.mean) / self.scale
        return round(float(scaled @ self.coefficients + self.intercept), 4)

    def report(self) -> Dict[str, Any]:
        """State the fit, so nobody has to guess how far to trust a prediction."""
        import numpy as np

        weights = dict(
            sorted(
                zip(self.features, (round(float(c), 4) for c in np.atleast_1d(self.coefficients))),
                key=lambda kv: -abs(kv[1]),
            )
        )
        return {
            "kind": "ridge regression on standardised features",
            "target": self.target,
            "unit": UNITS.get(self.target, ""),
            "trained_on_shots": self.trained_on,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "standardised_coefficients": weights,
        }


# ---------------------------------------------------------------------------
# the machine
# ---------------------------------------------------------------------------

@dataclass
class Machine:
    """One simulated machine: a clock, a cursor into the run, and setpoints."""

    shots: List[Dict[str, float]]
    model: QualityModel
    seconds_per_shot: float = 2.0
    running: bool = False
    started_at: float = 0.0
    shots_produced: int = 0
    setpoints: Dict[str, float] = field(default_factory=dict)
    _cursor: int = 0
    _lock: Any = field(default_factory=threading.Lock)

    @classmethod
    def build(cls, dataset: Path = DEFAULT_DATASET, seconds_per_shot: float = 2.0) -> "Machine":
        shots = load_shots(dataset)
        return cls(shots=shots, model=QualityModel.fit(shots), seconds_per_shot=seconds_per_shot)

    # -- state -----------------------------------------------------------

    def start(self) -> Dict[str, Any]:
        with self._lock:
            if self.running:
                return {"running": True, "note": "already running"}
            self.running = True
            self.started_at = time.time()
            return {"running": True, "started_at": self.started_at}

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self.running:
                return {"running": False, "note": "already stopped"}
            # Freeze the cursor where it is, so restarting continues the run
            # rather than jumping back to the beginning.
            produced = self._shot_index()
            self.shots_produced += (produced - self._cursor) % len(self.shots)
            self._cursor = produced
            self.running = False
            return {"running": False, "shots_produced": self.shots_produced}

    def _shot_index(self) -> int:
        if not self.running:
            return self._cursor
        elapsed = time.time() - self.started_at
        return (self._cursor + int(elapsed / self.seconds_per_shot)) % len(self.shots)

    # Shot index at which each setpoint was commanded, so the approach can be
    # measured from when the operator asked rather than from the run's start.
    setpoints_at: Dict[str, int] = field(default_factory=dict)

    # -- reading ---------------------------------------------------------

    def reading(self, name: str) -> float:
        """One reading, with any setpoint applied as an approach, not a jump."""
        index = self._shot_index()
        shot = self.shots[index]
        if name not in shot:
            raise MachineError(
                f"the machine has no reading named {name!r}. "
                f"It publishes: {', '.join(sorted(shot))}"
            )
        measured = shot[name]
        target = self.setpoints.get(name)
        if target is None:
            return round(measured, 4)

        # Geometric approach from the value at the moment of the command.
        elapsed_shots = max(0, index - self.setpoints_at.get(name, index))
        remaining = (1.0 - APPROACH_PER_SHOT) ** (elapsed_shots + 1)
        return round(target + (measured - target) * remaining, 4)

    def telemetry(self) -> Dict[str, Any]:
        index = self._shot_index()
        shot = self.shots[index]
        return {
            "running": self.running,
            "shot": index,
            "of": len(self.shots),
            "readings": {
                name: {"value": self.reading(name), "unit": UNITS.get(name, "")}
                for name in sorted(shot)
            },
            "setpoints": dict(self.setpoints),
        }

    # -- commanding ------------------------------------------------------

    def set_point(self, name: str, value: float) -> Dict[str, Any]:
        """Command a setpoint. Refuses anything the machine only measures."""
        if name not in SETTABLE:
            raise MachineError(
                f"{name!r} is measured, not commanded. Settable: {', '.join(SETTABLE)}"
            )
        low, high = SETPOINT_LIMITS[name]
        if not low <= value <= high:
            raise MachineError(
                f"{name} must be between {low} and {high} {UNITS.get(name, '')}; got {value}"
            )
        with self._lock:
            self.setpoints[name] = float(value)
            self.setpoints_at[name] = self._shot_index()
        return {
            "setpoint": name,
            "commanded": value,
            "unit": UNITS.get(name, ""),
            "current": self.reading(name),
            "note": "the process approaches a setpoint over several shots",
        }

    def clear_setpoint(self, name: str) -> Dict[str, Any]:
        with self._lock:
            self.setpoints.pop(name, None)
            self.setpoints_at.pop(name, None)
        return {"cleared": name}

    # -- prediction ------------------------------------------------------

    def predict_quality(self, parameters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Predict part weight, for given parameters or for the current state."""
        if parameters:
            source = "the parameters supplied"
            values = {k: float(v) for k, v in parameters.items()}
        else:
            source = "the machine's current readings"
            values = {f: self.reading(f) for f in self.model.features}
        predicted = self.model.predict(values)
        return {
            "predicted": predicted,
            "of": self.model.target,
            "unit": UNITS.get(self.model.target, ""),
            "from": source,
            "r_squared": self.model.r_squared,
            "rmse": self.model.rmse,
            "inputs": values,
        }


# Plausible operating envelopes, taken from the range each column actually
# occupies in the run plus headroom. A setpoint outside them is refused: a
# simulated machine that accepts a barrel temperature of 4000 degrees teaches
# an agent that the machine will accept anything.
SETPOINT_LIMITS: Dict[str, tuple] = {
    "SwitchoverVolume": (0.0, 60.0),
    "MaxInjectionPressure": (100.0, 2200.0),
    "SwitchoverPressure": (100.0, 2200.0),
    "BackPressure": (0.0, 30.0),
    "MouldHeatingCircuit1": (20.0, 350.0),
    **{f"BarrelZone{n}": (20.0, 350.0) for n in range(1, 9)},
}

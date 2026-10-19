"""Narrow reader for hash-verified MERL Panda experiment DataFrames.

Imported only inside the isolated identification environment. Never loads
published learned models or the simulated/dynamics-component subsets.
"""
import hashlib
import importlib
import io
import pickle
import re
import zipfile

from agenticsim.domain import require
from agenticsim.domain.measurements import TRAIN, HOLDOUT
from agenticsim.runtime.paths import ROOT

ARCHIVE = ROOT / "work/cache/datasets/merl-lip4rid-v1/PUB-5510-LIP4RID.zip"
SHA256 = "527ec997b8f957f102e07f132f2663156ffcfe9366ed1c9f1d6e14ad21409288"


def verify_archive():
    require(ARCHIVE.is_file(), "fetch MERL public data with work/tools/fetch_merl_dataset.py")
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    require(digest.hexdigest() == SHA256, "MERL archive fingerprint mismatch")


class ExperimentUnpickler(pickle.Unpickler):
    # The official pinned archive uses these constructors. This allowlist is
    # intentionally narrower than pandas.read_pickle; no arbitrary import.
    allowed = {
        ("pandas.core.frame", "DataFrame"), ("pandas.core.internals.managers", "BlockManager"),
        ("pandas.core.internals.blocks", "new_block"), ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.base", "Index"), ("pandas.core.indexes.range", "RangeIndex"),
        ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
        ("numpy", "ndarray"), ("numpy", "dtype"), ("builtins", "slice"), ("functools", "partial"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.allowed:
            raise pickle.UnpicklingError("constructor is outside the pinned experiment allowlist")
        if (module, name) == ("pandas.core.internals.blocks", "new_block"):
            return legacy_new_block
        return getattr(importlib.import_module(module), name)


def legacy_new_block(values, placement, *, ndim):
    """Published pandas pickle used a slice; pandas 2.2 requires BlockPlacement."""
    from pandas._libs.internals import BlockPlacement
    from pandas.core.internals.blocks import new_block
    return new_block(values, placement=BlockPlacement(placement), ndim=ndim)


def member(seed):
    require(type(seed) is int and seed in TRAIN + HOLDOUT, "unregistered Panda excitation")
    sin = 50 if seed in TRAIN else 100
    return f"Data/Robots/PANDA/Experiments/panda7dof_num_sin_{sin}_seed_{seed}_as_filtered_fcut4.0.pkl"


def read_experiment(bundle, seed):
    import numpy as np
    import pandas as pd
    require(pd.__version__ == "2.2.3", "pinned pandas 2.2.3 required for published DataFrame decoding")
    name = member(seed)
    require(bundle.getinfo(name).file_size < 2_000_000, "unexpected Panda experiment size")
    payload = bundle.read(name)
    frame = ExperimentUnpickler(io.BytesIO(payload)).load()
    require(type(frame) is pd.DataFrame and frame.columns.is_unique, "experiment must have unique DataFrame columns")
    expected = {"t"} | {f"{prefix}_{j}" for prefix in ("q", "dq", "ddq", "tau", "tau_interp") for j in range(1, 8)}
    require(set(frame.columns) == expected, "published Panda column mapping changed")
    require(100 < len(frame) < 5000, "unexpected Panda experiment length")
    for col in frame.columns:
        require(re.fullmatch(r"t|(?:q|dq|ddq|tau|tau_interp)_[1-7]", col) is not None
                and np.issubdtype(frame[col].dtype, np.number), "non-numeric experiment column")
    require(np.isfinite(frame.to_numpy()).all(), "nonfinite public measurements")
    return frame, hashlib.sha256(payload).hexdigest()


def inspect():
    import numpy as np
    verify_archive()
    with zipfile.ZipFile(ARCHIVE) as bundle:
        frame, checksum = read_experiment(bundle, TRAIN[0])
    print({"samples": len(frame), "sha256": checksum,
           "time_range": [float(frame.t.iloc[0]), float(frame.t.iloc[-1])],
           "median_dt": float(np.median(np.diff(frame.t))),
           "q_range": [[float(frame[f'q_{j}'].min()), float(frame[f'q_{j}'].max())] for j in range(1, 8)],
           "tau_rmse": [float(np.sqrt(np.mean(frame[f'tau_interp_{j}']**2))) for j in range(1, 8)]})


if __name__ == "__main__":
    inspect()

"""Official eight-output PKL encoding and restricted local format validation."""
import io
import pickle
from agenticsim.domain import require
from agenticsim.domain.plaid import FIELDS, SCALARS


def validate_predictions(predictions, node_counts):
    import numpy as np
    require(type(predictions) is list and len(predictions) == len(node_counts) == 200, 'official test requires 200 rows')
    for row, count in zip(predictions, node_counts):
        require(isinstance(row, dict) and set(row) == set((*FIELDS, *SCALARS)), 'official output keys disagree')
        for name in FIELDS:
            require(isinstance(row[name], np.ndarray) and row[name].shape == (count,) and
                    row[name].dtype.kind == 'f' and np.isfinite(row[name]).all(), 'invalid official field shape/type/value')
        for name in SCALARS:
            require(type(row[name]) is float and np.isfinite(row[name]), 'invalid official scalar')
    return predictions


def read_predictions(raw):
    import numpy as np
    class Reader(pickle.Unpickler):
        def find_class(self, module, name):
            allowed = {('numpy', 'dtype'): np.dtype, ('numpy', 'ndarray'): np.ndarray}
            for prefix in ('numpy.core', 'numpy._core'):
                allowed[(prefix+'.numeric', '_frombuffer')] = np._core.numeric._frombuffer
                allowed[(prefix+'.multiarray', '_reconstruct')] = np._core.multiarray._reconstruct
            require((module, name) in allowed, 'non-array class in prediction file')
            return allowed[module, name]
        def persistent_load(self, pid):
            raise ValueError('persistent pickle references unavailable')
    require(type(raw) is bytes and len(raw) < 200_000_000, 'prediction file exceeds bound')
    return Reader(io.BytesIO(raw)).load()


def write_predictions(path, predictions, node_counts):
    import numpy as np
    validate_predictions(predictions, node_counts)
    raw = pickle.dumps(predictions, protocol=4)
    decoded = validate_predictions(read_predictions(raw), node_counts)
    require(all(np.array_equal(row[name], other[name]) for row, other in zip(predictions, decoded)
                for name in (*FIELDS, *SCALARS)), 'official prediction roundtrip failed')
    path.write_bytes(raw)

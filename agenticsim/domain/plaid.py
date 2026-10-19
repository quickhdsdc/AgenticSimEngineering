"""Dependency-free contracts for the bounded Tensile2D surrogate pilot."""
import math
from agenticsim.domain import require

FIELDS = ('U1', 'U2', 'sig11', 'sig22', 'sig12')
SCALARS = ('max_von_mises', 'max_U2_top', 'max_sig22_top')
INPUTS = ('P', 'p1', 'p2', 'p3', 'p4', 'p5')
CONTRACT = 'agenticsim-plaid-tensile2d-surrogate-v1'


def validate_recipe(recipe):
    require(isinstance(recipe, dict) and set(recipe) == {'method', 'neighbors', 'geometry_weights'},
            'recipe requires method, neighbors and geometry_weights only')
    require(recipe['method'] == 'inverse_distance_mesh_knn', 'unsupported surrogate method')
    require(isinstance(recipe['neighbors'], list) and 1 <= len(recipe['neighbors']) <= 5,
            'supply one to five neighbor candidates')
    require(all(type(k) is int and k in (1, 2, 4, 8, 16) for k in recipe['neighbors']), 'unsupported neighbor count')
    weights = recipe['geometry_weights']
    require(isinstance(weights, list) and 1 <= len(weights) <= 3, 'supply one to three geometry weights')
    require(all(type(w) in (int, float) and math.isfinite(w) and w in (0.25, 1, 4) for w in weights),
            'unsupported geometry weight')
    require(len(set(recipe['neighbors'])) == len(recipe['neighbors']) and len(set(weights)) == len(weights),
            'duplicate recipe candidates')
    return recipe


def split_ids(ids):
    """Fixed split by simulation case, independent of labels and file ordering."""
    import random
    require(len(ids) == 500 and len(set(ids)) == 500 and all(type(i) is int for i in ids),
            'expected 500 unique labeled publisher IDs')
    shuffled = sorted(ids)
    random.Random(20261005).shuffle(shuffled)
    return {'train': sorted(shuffled[:300]), 'validation': sorted(shuffled[300:400]),
            'holdout': sorted(shuffled[400:])}

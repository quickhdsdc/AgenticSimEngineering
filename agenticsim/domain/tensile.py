"""Solver-neutral task validation for the complete local Tensile2D benchmark."""
from agenticsim.domain import require, number

CONTRACT = 'agenticsim-tensile2d-physics-residual-v1'


def validate_model_spec(spec):
    require(isinstance(spec, dict) and set(spec) == {'model', 'initial_poisson', 'initial_young_scale',
            'elastic_fit_cases', 'neighbors', 'geometry_weights'}, 'invalid tensile model specification')
    require(spec['model'] == 'plane_strain_tri3_residual', 'unsupported physical model family')
    require(number(spec['initial_poisson']) and 0.1 <= spec['initial_poisson'] <= 0.45, 'Poisson guess outside bounds')
    require(number(spec['initial_young_scale']) and 0.5 <= spec['initial_young_scale'] <= 1.5, 'Young scale outside bounds')
    require(type(spec['elastic_fit_cases']) is int and 2 <= spec['elastic_fit_cases'] <= 8, 'elastic fit case budget invalid')
    from agenticsim.domain.plaid import validate_recipe
    validate_recipe({'method': 'inverse_distance_mesh_knn', 'neighbors': spec['neighbors'],
                     'geometry_weights': spec['geometry_weights']})
    return spec

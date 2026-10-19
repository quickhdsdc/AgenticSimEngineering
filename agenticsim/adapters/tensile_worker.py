"""Training-only elasticity identification plus learned residual benchmark."""
import json
from pathlib import Path
import sys
import time

from agenticsim.domain import require
from agenticsim.domain.plaid import FIELDS, SCALARS
from agenticsim.domain.tensile import CONTRACT
from agenticsim.runtime.calibration import load_json, save_json, sha256


def prepare(directory, source):
    import numpy as np
    import pyarrow.parquet as pq
    from scripts.fetch_plaid_dataset import CACHE
    from agenticsim.adapters.plaid_worker import prepare as prepare_surrogate, restricted_sample, child
    report = prepare_surrogate(directory, source)
    from agenticsim.adapters.plaid_worker import arrays_to_samples
    data = arrays_to_samples(directory/'.evaluation/data.npz')
    index = 0
    for path in sorted((CACHE/'data').glob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=['sample']):
            for raw in batch.column(0).to_pylist():
                sample = restricted_sample(raw)
                zone = child(child(sample['meshes'][0.0], 'Base_2_2'), 'Zone')
                bcs = child(zone, 'ZoneBC')
                for name, tag in (('bottom', 'Bottom'), ('top', 'Top'), ('bottom_left', 'BottomLeft')):
                    data[index]['bc_'+name] = child(child(bcs, tag), 'PointList')[1].ravel().astype(int)-1
                if 'max_q' in sample['scalars']:
                    data[index]['max_q'] = np.array(float(sample['scalars']['max_q']))
                index += 1
    np.savez_compressed(directory/'.evaluation/data.npz', **{f'{i}_{key}':value for i,s in data.items() for key,value in s.items()})
    report.update(contract=CONTRACT, task='physical-model identification + residual surrogate comparison',
                  boundary_source='explicit CGNS Bottom/BottomLeft/Top tags',
                  adaptation='fixed 300/100/100 local split; official export refits residual references on train_500',
                  model_scope='elastic FEM with learned plastic corrections; no recovered nonlinear constitutive law')
    save_json(directory/'dataset.json', report)
    return report


def mesh_for(sample):
    from agenticsim.adapters.tensile_fem import PlaneStrainMesh
    return PlaneStrainMesh(sample['xy'], sample['tri'],
                          {name:sample['bc_'+name] for name in ('bottom', 'top', 'bottom_left')})


def identify(samples, train, spec):
    import numpy as np
    from scipy.optimize import least_squares
    eligible = [i for i in train if samples[i]['max_q'] <= 1e-10]
    require(len(eligible) >= spec['elastic_fit_cases'], 'insufficient training elastic cases')
    ids = eligible[:spec['elastic_fit_cases']]
    meshes = {i:mesh_for(samples[i]) for i in ids}
    def residual(parameters):
        out = []
        for i in ids:
            sample = samples[i]
            (predicted, _), _ = meshes[i].elastic(sample['features'][5]*parameters[1], parameters[0], sample['features'][0])
            scale = np.max(np.abs(sample['fields'][:,:2]), axis=0)
            out.append(((predicted[:,:2]-sample['fields'][:,:2])/scale/(len(predicted)*len(ids))**0.5).ravel())
        return np.concatenate(out)
    initial = [spec['initial_poisson'], spec['initial_young_scale']]
    fitted = least_squares(residual, initial, bounds=([0.1,0.5],[0.45,1.5]),
                           xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=40)
    require(fitted.success and np.isfinite(fitted.x).all(), 'elastic identification failed')
    return {'poisson':float(fitted.x[0]), 'young_scale':float(fitted.x[1]), 'initial':initial,
            'training_case_ids':ids, 'eligible_training_cases':len(eligible), 'function_evaluations':fitted.nfev,
            'initial_normalized_displacement_rmse':float(np.linalg.norm(residual(initial))),
            'fitted_normalized_displacement_rmse':float(np.linalg.norm(fitted.fun)),
            'bounds':[[0.1,0.45],[0.5,1.5]], 'jacobian_condition':float(np.linalg.cond(fitted.jac)),
            'material_mapping':'Young=p5*identified_scale; p1-p4 used by residual learner, meanings remain unknown',
            'identification_scope':'reference dataset elasticity, not measured material calibration'}


class PhysicsResidual:
    def __init__(self, samples, train, parameters, cache=None):
        self.samples, self.train, self.parameters = samples, train, parameters
        self.cache = {} if cache is None else cache
        self.residual = None
        self.method = 'elastic_fem'

    def base(self, case):
        if case not in self.cache:
            sample = self.samples[case]
            mesh = mesh_for(sample)
            outputs, diagnostics = mesh.elastic(sample['features'][5]*self.parameters['young_scale'],
                                                 self.parameters['poisson'], sample['features'][0])
            require(diagnostics['converged'], 'base FEM failed equilibrium check')
            self.cache[case] = (*outputs, diagnostics)
        return self.cache[case]

    def train_residual(self):
        from agenticsim.adapters.plaid_worker import Surrogate
        residual_samples = {i:dict(sample) for i,sample in self.samples.items()}
        for i in self.train:
            fields, scalars, _ = self.base(i)
            residual_samples[i]['fields'] = self.samples[i]['fields']-fields
            residual_samples[i]['scalars'] = self.samples[i]['scalars']-scalars
        self.residual = Surrogate(residual_samples, self.train)

    def predict(self, case, k=1, geometry_weight=1):
        import numpy as np
        fields, scalars, _ = self.base(case)
        if self.method == 'elastic_fem':
            return fields, scalars
        require(self.residual is not None, 'fit residual model first')
        correction, _ = self.residual.predict(case, k, geometry_weight)
        fields = fields + correction
        sample = self.samples[case]
        # Corrections preserve the supplied displacement boundary conditions.
        fields[sample['bc_bottom'],1] = 0
        fields[sample['bc_bottom_left'],0] = 0
        stress = fields[:,2:]
        vm = np.sqrt(stress[:,0]**2+stress[:,1]**2-stress[:,0]*stress[:,1]+3*stress[:,2]**2)
        scalars = np.array([vm.max(), fields[sample['bc_top'],1].max(), fields[sample['bc_top'],3].max()])
        return fields, scalars


def cache_save(directory, name, cache):
    import numpy as np
    arrays = {}
    diagnostics = {}
    for i,(fields,scalars,diag) in cache.items():
        arrays[f'{i}_fields'], arrays[f'{i}_scalars'] = fields, scalars
        diagnostics[str(i)] = diag
    np.savez_compressed(directory/'.evaluation'/name, **arrays)
    save_json(directory/'.evaluation'/(name+'.json'), diagnostics)


def cache_load(directory, name):
    import numpy as np
    diagnostics = load_json(directory/'.evaluation'/(name+'.json'))
    with np.load(directory/'.evaluation'/name, allow_pickle=False) as arrays:
        return {int(i):(arrays[i+'_fields'], arrays[i+'_scalars'], diag) for i,diag in diagnostics.items()}


def run_phase(directory, phase):
    import numpy as np
    from agenticsim.adapters.plaid_worker import arrays_to_samples, metrics, Surrogate, plot_case
    samples = arrays_to_samples(directory/'.evaluation/data.npz')
    splits = load_json(directory/'.evaluation/splits.json')
    spec = load_json(directory/'generated-model.json')['specification']
    baseline = {'neighbors':1, 'geometry_weight':1}
    started = time.monotonic()
    if phase == 'fit':
        parameters = identify(samples, splits['train'], spec)
        model = PhysicsResidual(samples, splits['train'], parameters)
        elastic = {'method':'elastic_fem', **baseline}
        rows = [{'candidate':elastic, 'validation':metrics(model, splits['validation'],elastic)}]
        model.train_residual()
        model.method = 'physics_residual'
        for k in spec['neighbors']:
            for weight in spec['geometry_weights']:
                candidate = {'method':'physics_residual', 'neighbors':k, 'geometry_weight':weight}
                rows.append({'candidate':candidate, 'validation':metrics(model, splits['validation'],candidate)})
        best = min(rows, key=lambda r:r['validation']['total'])
        cache_save(directory,'fem-training-validation.npz', model.cache)
        report = {'parameters':parameters, 'selected':best['candidate'], 'validation':best['validation'], 'trials':rows,
                  'selection_data':'validation_only', 'training_cases':300,
                  'maximum_base_equilibrium_residual':max(v[2]['relative_equilibrium_residual'] for v in model.cache.values()),
                  'prediction_source':'learned_prediction' if best['candidate']['method']=='physics_residual' else 'reference_simulation',
                  'field_qoi_consistency':'three scalars derived from the five predicted fields',
                  'limitation':'elastic base converges; learned corrections do not establish nonlinear equilibrium or hardening law'}
    elif phase == 'score':
        fit = load_json(directory/'fit.json')
        model = PhysicsResidual(samples,splits['train'],fit['parameters'],cache_load(directory,'fem-training-validation.npz'))
        elastic_score = metrics(model, splits['holdout'], baseline)
        model.train_residual()
        model.method = fit['selected']['method']
        selected = metrics(model,splits['holdout'],fit['selected'])
        raw = Surrogate(samples,splits['train'])
        nearest = metrics(raw,splits['holdout'],baseline)
        # Freeze the previous pilot recipe as a comparator; no tuning on this holdout.
        knn = metrics(raw,splits['holdout'],{'neighbors':8,'geometry_weight':4})
        groups = {'elastic':[i for i in splits['holdout'] if samples[i]['max_q']<=1e-10],
                  'plastic':[i for i in splits['holdout'] if samples[i]['max_q']>1e-10]}
        boundary_error = 0.0
        for i in splits['holdout']:
            fields, _ = model.predict(i, fit['selected']['neighbors'], fit['selected']['geometry_weight'])
            boundary_error = max(boundary_error, float(np.abs(fields[samples[i]['bc_bottom'],1]).max()),
                                 float(np.abs(fields[samples[i]['bc_bottom_left'],0]).max()))
        report = {'scope':'internal_holdout_within_publisher_train_500', 'selected':selected, 'elastic_fem':elastic_score,
                  'nearest_neighbor':nearest, 'prior_knn_recipe':knn,
                  'strata':{name:metrics(model,ids,fit['selected']) if ids else None for name,ids in groups.items()},
                  'maximum_base_equilibrium_residual':max(model.cache[i][2]['relative_equilibrium_residual'] for i in splits['holdout']),
                  'bc_max_abs':boundary_error, 'official_hidden_score':None, 'physical_validation':'not_established',
                  'acceptance':'reported_metrics_no_physical_acceptance',
                  'nonlinear_constitutive_validation':'not_established', 'agent_accuracy_advantage':'not_established'}
        cache_save(directory,'fem-holdout.npz',{i:model.cache[i] for i in splits['holdout']})
        predictions = {f'{i}_{key}':v for i in splits['holdout'] for key,v in zip(('fields','scalars'),model.predict(i,fit['selected']['neighbors'],fit['selected']['geometry_weight']))}
        np.savez_compressed(directory/'.evaluation/holdout-predictions.npz',**predictions)
        plot_case(directory,model,splits['holdout'][0],fit['selected'])
    else:
        from agenticsim.adapters.tensile_submission import write_predictions
        fit = load_json(directory/'fit.json')
        cache = cache_load(directory,'fem-training-validation.npz')
        cache.update(cache_load(directory,'fem-holdout.npz'))
        official_train = splits['train']+splits['validation']+splits['holdout']
        model = PhysicsResidual(samples,sorted(official_train),fit['parameters'],cache)
        model.train_residual()
        model.method = fit['selected']['method']
        predictions = []
        node_counts = []
        for case in splits['official_test']:
            fields,scalars = model.predict(case,fit['selected']['neighbors'],fit['selected']['geometry_weight'])
            predictions.append({**{name:fields[:,j] for j,name in enumerate(FIELDS)},
                                **{name:float(scalars[j]) for j,name in enumerate(SCALARS)}})
            node_counts.append(len(samples[case]['xy']))
        write_predictions(directory/'submission.pkl',predictions,node_counts)
        ood = {}
        for case in splits['ood']:
            fields,scalars = model.predict(case,fit['selected']['neighbors'],fit['selected']['geometry_weight'])
            ood[f'{case}_fields'],ood[f'{case}_scalars'] = fields,scalars
        np.savez_compressed(directory/'ood-predictions.npz',**ood)
        report = {'format':'official 200-row list of eight-output dictionaries, pickle protocol 4',
                  'local_format_validation':'passed_shape_keys_finiteness_order_and_roundtrip',
                  'official_test_ids':splits['official_test'],'node_counts':node_counts,'ood_ids':splits['ood'],
                  'official_training_cases':500,'local_score_training_cases':300,
                  'refit':'frozen parameters/configuration; residual references expanded after local score',
                  'submission_sha256':sha256(directory/'submission.pkl'),'submission_bytes':(directory/'submission.pkl').stat().st_size,
                  'official_hidden_score':None,'server_status':'not_submitted_requires_competition_login',
                  'physical_validation':'not_established','source':'learned_prediction'}
    report['compute_seconds'] = time.monotonic()-started
    save_json(directory/(phase+'.json'),report)
    return report


def main():
    request = json.load(sys.stdin)
    directory = Path(request['directory'])
    result = prepare(directory,request['source']) if request['phase']=='prepare' else run_phase(directory,request['phase'])
    print(json.dumps(result,allow_nan=False))


if __name__ == '__main__':
    main()

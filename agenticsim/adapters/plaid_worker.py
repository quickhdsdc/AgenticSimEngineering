"""CPU-only, credential-stripped worker; optional libraries import on execution."""
import csv
import io
import json
from pathlib import Path
import pickle
import sys
import time

from agenticsim.domain import require
from agenticsim.domain.plaid import CONTRACT, FIELDS, INPUTS, SCALARS, split_ids, validate_recipe
from agenticsim.runtime.calibration import load_json, save_json, sha256


def restricted_sample(raw):
    import numpy as np
    class Reader(pickle.Unpickler):
        def find_class(self, module, name):
            allowed = {('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype}
            for prefix in ('numpy.core', 'numpy._core'):
                allowed[(prefix + '.multiarray', '_reconstruct')] = np._core.multiarray._reconstruct
                allowed[(prefix + '.multiarray', 'scalar')] = np._core.multiarray.scalar
                allowed[(prefix + '.numeric', '_frombuffer')] = np._core.numeric._frombuffer
            require((module, name) in allowed, 'unsupported serialized class')
            return allowed[module, name]
        def persistent_load(self, pid):
            raise ValueError('persistent pickle references unavailable')
    require(type(raw) is bytes and len(raw) < 8_000_000, 'unexpected sample encoding/size')
    result = Reader(io.BytesIO(raw)).load()
    require(isinstance(result, dict), 'sample must be a dictionary')
    return result


def child(node, name):
    found = [c for c in node[2] if c[0] == name]
    require(len(found) == 1, 'missing or ambiguous CGNS node: ' + name)
    return found[0]


def decode(sample, labeled):
    import numpy as np
    require(set(sample['meshes']) == {0.0}, 'only quasistatic t=0 sample supported')
    zone = child(child(sample['meshes'][0.0], 'Base_2_2'), 'Zone')
    grid = child(zone, 'GridCoordinates')
    xy = np.column_stack([child(grid, 'Coordinate' + axis)[1] for axis in ('X', 'Y')]).astype(float)
    tri = child(child(zone, 'Elements_TRI_3'), 'ElementConnectivity')[1].reshape(-1, 3).astype(int) - 1
    require(3 <= len(xy) <= 100_000 and np.isfinite(xy).all(), 'invalid mesh coordinates')
    require(tri.min() >= 0 and tri.max() < len(xy), 'invalid one-based CGNS connectivity')
    sides = xy[tri[:, 1:]] - xy[tri[:, 0, None]]
    area = np.abs(sides[:, 0, 0]*sides[:, 1, 1] - sides[:, 0, 1]*sides[:, 1, 0]).sum()/2
    span = np.ptp(xy, axis=0)
    require((span > 0).all() and area > 0, 'degenerate mesh')
    geometry = np.r_[area, span, xy.std(axis=0)]
    features = np.r_[[sample['scalars'][name] for name in INPUTS], geometry]
    require(np.isfinite(features).all(), 'invalid model inputs')
    data = {'xy': xy, 'tri': tri, 'features': features}
    if labeled:
        points = child(zone, 'PointData')
        fields = np.column_stack([child(points, name)[1] for name in FIELDS]).astype(float)
        scalars = np.array([sample['scalars'][name] for name in SCALARS], dtype=float)
        require(fields.shape == (len(xy), len(FIELDS)) and np.isfinite(fields).all() and np.isfinite(scalars).all(),
                'invalid labeled sample')
        require((np.max(np.abs(fields), axis=0) > 0).all() and (scalars != 0).all(), 'zero metric denominator')
        data.update(fields=fields, scalars=scalars)
    else:
        require(not any(name in sample['scalars'] for name in SCALARS), 'hidden labels unexpectedly public')
    return data


def prepare(directory, source):
    import numpy as np
    import pyarrow.parquet as pq
    import yaml
    from scripts.fetch_plaid_dataset import CACHE, check_file
    for entry in source['files']:
        check_file(CACHE / entry['path'], entry)
    header = (CACHE / 'README.md').read_text(encoding='utf-8').split('---', 2)[1]
    info = yaml.safe_load(header)['dataset_info']['description']
    publisher = info['split']
    require(len(publisher['train_500']) == 500 and len(publisher['test']) == 200, 'publisher split mismatch')
    splits = split_ids(publisher['train_500'])
    splits.update(official_test=publisher['test'], ood=publisher['OOD'])
    labeled = set(publisher['train_500'])
    arrays = {}
    samples = []
    index = 0
    for entry in source['files']:
        if not entry['path'].endswith('.parquet'):
            continue
        for batch in pq.ParquetFile(CACHE / entry['path']).iter_batches(batch_size=16, columns=['sample']):
            for raw in batch.column(0).to_pylist():
                sample = decode(restricted_sample(raw), index in labeled)
                for key, value in sample.items():
                    arrays[f'{index}_{key}'] = value
                samples.append({'id': index, 'nodes': len(sample['xy']), 'triangles': len(sample['tri'])})
                index += 1
    require(index == 702 and sorted(labeled | set(splits['official_test']) | set(splits['ood'])) == list(range(702)),
            'sample count or coverage mismatch')
    np.savez_compressed(directory / '.evaluation/data.npz', **arrays)
    save_json(directory / '.evaluation/splits.json', splits)
    with (directory / 'train.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case_id', *INPUTS, 'mesh_area', 'span_x', 'span_y', 'node_std_x', 'node_std_y', *SCALARS])
        for i in splits['train']:
            writer.writerow([i, *arrays[f'{i}_features'], *arrays[f'{i}_scalars']])
    dataset = {'contract': CONTRACT, 'source': 'reference_simulation', 'dataset': source['dataset'],
               'revision': source['revision'], 'license': source['license'], 'samples': len(samples),
               'splits': {name: len(ids) for name, ids in splits.items()}, 'seed': 20261005,
               'field_targets': FIELDS, 'scalar_targets': SCALARS,
               'node_count_range': [min(s['nodes'] for s in samples), max(s['nodes'] for s in samples)],
               'physical_validation': 'not_established', 'official_hidden_score': None,
               'units': 'publisher does not specify units; no SI conversion or physical mapping claimed',
               'frame': 'publisher CGNS CoordinateX/CoordinateY; acquisition mapping not applicable',
               'adaptation': '300/100/100 local case split within train_500; bounded mesh-kNN recipes',
               'geometry_features': ['triangle_area', 'span_x', 'span_y', 'node_std_x', 'node_std_y']}
    save_json(directory / 'dataset.json', dataset)
    return dataset


def arrays_to_samples(path):
    import numpy as np
    samples = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            case, name = key.split('_', 1)
            samples.setdefault(int(case), {})[name] = data[key]
    return samples


class Surrogate:
    def __init__(self, samples, train):
        import numpy as np
        from scipy.spatial import cKDTree
        self.samples, self.train = samples, train
        self.x = np.array([samples[i]['features'] for i in train])
        self.scale = self.x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1
        self.trees = {i: cKDTree(samples[i]['xy']) for i in train}
        self.spatial = {}

    def predict(self, case, k, geometry_weight):
        import numpy as np
        target = self.samples[case]
        difference = (self.x - target['features'])/self.scale
        difference[:, len(INPUTS):] *= geometry_weight
        distance = np.linalg.norm(difference, axis=1)
        indices = np.argsort(distance, kind='stable')[:k]
        weights = 1 / np.maximum(distance[indices], 1e-12)**2
        weights /= weights.sum()
        fields = np.zeros((len(target['xy']), len(FIELDS)))
        scalars = np.zeros(len(SCALARS))
        for index, weight in zip(indices, weights):
            other = self.train[index]
            cache_key = (case, other)
            if cache_key not in self.spatial:
                d, neighbors = self.trees[other].query(target['xy'], k=3)
                spatial_weights = 1/np.maximum(d, 1e-12)**2
                spatial_weights /= spatial_weights.sum(axis=1, keepdims=True)
                self.spatial[cache_key] = np.sum(self.samples[other]['fields'][neighbors] * spatial_weights[:, :, None], axis=1)
            fields += weight * self.spatial[cache_key]
            scalars += weight * self.samples[other]['scalars']
        return fields, scalars


def metrics(model, ids, candidate):
    import numpy as np
    errors = []
    for case in ids:
        predicted_fields, predicted_scalars = model.predict(case, candidate['neighbors'], candidate['geometry_weight'])
        gold = model.samples[case]
        field_error = np.mean((predicted_fields-gold['fields'])**2, axis=0)/np.max(np.abs(gold['fields']), axis=0)**2
        scalar_error = ((predicted_scalars-gold['scalars'])/gold['scalars'])**2
        errors.append(np.r_[field_error, scalar_error])
    values = np.sqrt(np.mean(errors, axis=0))
    require(np.isfinite(values).all(), 'nonfinite benchmark metrics')
    return {'rrmse': dict(zip((*FIELDS, *SCALARS), values.tolist())), 'total': float(values.mean()), 'cases': len(ids)}


def compute(directory, phase, recipe=None):
    import numpy as np
    samples = arrays_to_samples(directory / '.evaluation/data.npz')
    splits = load_json(directory / '.evaluation/splits.json')
    model = Surrogate(samples, splits['train'])
    started = time.monotonic()
    baseline = {'neighbors': 1, 'geometry_weight': 1}
    if phase == 'fit':
        recipe = validate_recipe(recipe)
        rows = []
        for k in recipe['neighbors']:
            for weight in recipe['geometry_weights']:
                candidate = {'neighbors': k, 'geometry_weight': weight}
                rows.append({'candidate': candidate, 'validation': metrics(model, splits['validation'], candidate)})
        best = min(rows, key=lambda row: row['validation']['total'])
        report = {'method': recipe['method'], 'selected': best['candidate'], 'validation': best['validation'],
                  'trials': rows, 'baseline_validation': metrics(model, splits['validation'], baseline),
                  'selection_data': 'validation_only', 'training_cases': len(splits['train']),
                  'prediction_source': 'learned_prediction', 'algorithm_version': CONTRACT,
                  'geometry_interpolation': '3-nearest-node inverse-distance; not mesh morphing or a PDE solver',
                  'scalar_consistency': 'scalar targets regressed independently; not derived from predicted fields'}
        save_json(directory / 'fit.json', report)
        np.savez_compressed(directory / 'model.npz', training_ids=np.array(splits['train']),
                            feature_scale=model.scale, training_features=model.x)
    elif phase == 'score':
        fit = load_json(directory / 'fit.json')
        report = {'scope': 'internal_holdout_within_publisher_train_500',
                  'selected': metrics(model, splits['holdout'], fit['selected']),
                  'baseline': metrics(model, splits['holdout'], baseline),
                  'official_hidden_score': None, 'physical_validation': 'not_established',
                  'acceptance': 'metrics_only_no_physical_acceptance', 'source': 'reference_simulation',
                  'prediction_source': 'learned_prediction', 'published_score_comparable': False}
        save_json(directory / 'score.json', report)
        plot_case(directory, model, splits['holdout'][0], fit['selected'])
    else:
        fit = load_json(directory / 'fit.json')
        outputs = {}
        ids = splits['official_test'] + splits['ood']
        for case in ids:
            fields, scalars = model.predict(case, **{'k': fit['selected']['neighbors'],
                                                   'geometry_weight': fit['selected']['geometry_weight']})
            outputs[f'{case}_fields'], outputs[f'{case}_scalars'] = fields, scalars
        np.savez_compressed(directory / 'predictions.npz', **outputs)
        report = {'cases': len(ids), 'official_test_ids': splits['official_test'], 'ood_ids': splits['ood'],
                  'field_order': FIELDS, 'scalar_order': SCALARS, 'source': 'learned_prediction',
                  'format': 'project NPZ; not a verified official competition submission',
                  'official_hidden_score': None, 'physical_validation': 'not_established',
                  'prediction_sha256': sha256(directory / 'predictions.npz')}
        save_json(directory / 'export.json', report)
    report['compute_seconds'] = time.monotonic() - started
    save_json(directory / (phase + '.json'), report)
    return report


def plot_case(directory, model, case, candidate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    gold = model.samples[case]
    predicted, _ = model.predict(case, candidate['neighbors'], candidate['geometry_weight'])
    index = FIELDS.index('sig22')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, values, title in zip(axes, (gold['fields'][:, index], predicted[:, index],
                                        np.abs(predicted[:, index]-gold['fields'][:, index])),
                                 ('Reference simulation', 'Learned prediction', 'Absolute error')):
        artist = ax.tripcolor(gold['xy'][:, 0], gold['xy'][:, 1], gold['tri'], values, shading='gouraud')
        if title != 'Absolute error':
            artist.set_clim(gold['fields'][:, index].min(), gold['fields'][:, index].max())
        ax.set_aspect('equal')
        ax.set_title(title)
        fig.colorbar(artist, ax=ax)
    fig.suptitle(f'Tensile2D case {case}: sig22 (publisher units unspecified)')
    fig.tight_layout()
    fig.savefig(directory / 'holdout-field.png', dpi=150)
    plt.close(fig)


def main():
    request = json.load(sys.stdin)
    directory = Path(request['directory'])
    if request['phase'] == 'prepare':
        result = prepare(directory, request['source'])
    else:
        result = compute(directory, request['phase'], request.get('recipe'))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

"""Bounded native workers. Only audited upstream reference code is executable."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / 'components/model_benchmarks'


def factory(model, horizon, normalize=True):
    import simpy
    sys.path.insert(0, str(SOURCES / 'factoryflow'))
    from factorysimpy.nodes.node import Node
    from factorysimpy.nodes.source import Source
    from factorysimpy.nodes.machine import Machine
    from factorysimpy.nodes.sink import Sink
    from factorysimpy.edges.buffer import Buffer
    env = simpy.Environment()
    top = Node(env, 'TOP')
    nodes = {}
    for row in model['nodes']:
        cls = {'source': Source, 'machine': Machine, 'sink': Sink}[row['kind']]
        parameters = {key: row[key] for key in ('inter_arrival_time', 'processing_delay', 'work_capacity') if key in row}
        node = cls(env, row['id'], **parameters) if normalize else cls(env, row['id'])
        for key in ('inter_arrival_time', 'processing_delay', 'work_capacity'):
            if key in row:
                setattr(node, key, row[key])
        nodes[row['id']] = node
        top.add_child_node(node)
    for row in model['buffers']:
        edge = Buffer(env, row['id'], capacity=row['capacity']) if normalize else Buffer(env, row['id'])
        edge.capacity = row['capacity']
        edge.connect(nodes[row['from']], nodes[row['to']])
        top.add_child_edge(edge)
    top.fill_hierarchical_id()
    top.run_simulation(horizon)
    return factory_metrics(top, normalize)


def factory_metrics(top, audit=False):
    nodes, edges = top.get_all_nodes_edges()
    sinks = [v for v in nodes.values() if 'num_item_received' in getattr(v, 'stats', {})]
    metrics = {'received': sum(v.stats['num_item_received'] for v in sinks),
               'cycle_time_sum': sum(v.stats['total_cycle_time'] for v in sinks)}
    if audit:
        metrics['capacity_audit'] = sorted([
            {'id': v.id, 'declared': v.work_capacity, 'realized': v.worker_thread.capacity}
            for v in nodes.values() if hasattr(v, 'worker_thread')] + [
            {'id': v.id, 'declared': v.capacity, 'realized': v.inbuiltstore.capacity}
            for v in edges.values() if hasattr(v, 'inbuiltstore')], key=lambda row: row['id'])
    return metrics


def factory_reference(task, horizon, normalize=True):
    # Audited S1/S2 only; remove drawing/path mutation, preserve native model construction.
    import simpy  # noqa: F401
    sys.path.insert(0, str(SOURCES / 'factoryflow'))
    tree = ast.parse((SOURCES / 'factoryflow' / f'{task}_GT.py').read_text())
    keep = []
    for node in tree.body:
        text = ast.unparse(node)
        if any(token in text for token in ('sys.path.', 'draw_blockdiagram(', '.render(',
                                           'TOP.validate()', 'TOP.show_hierarchy()', 'TOP.run_simulation(')):
            continue
        keep.append(node)
    scope = {'__file__': str(SOURCES / 'factoryflow' / f'{task}_GT.py')}
    exec(compile(ast.Module(body=keep, type_ignores=[]), '<audited-reference>', 'exec'), scope)
    top = scope['TOP']
    if normalize:
        from factorysimpy.base.buffer_store import BufferStore
        nodes, edges = top.get_all_nodes_edges()
        # Original scripts mutate capacity fields after resource construction.
        # Normalize only the adapter's fresh resources, before any event runs.
        for node in nodes.values():
            if hasattr(node, 'worker_thread'):
                node.worker_thread = simpy.Resource(top.env, capacity=node.work_capacity)
                node.time_per_work_occupancy = [0.0] * (node.work_capacity + 1)
        for edge in edges.values():
            if hasattr(edge, 'inbuiltstore'):
                edge.inbuiltstore = BufferStore(top.env, capacity=edge.capacity, mode=edge.mode)
    top.run_simulation(horizon)
    return factory_metrics(top, normalize)


def chrono_import():
    # Windows conda DLL resolution without activation or global PATH edits.
    handles = []
    dll = Path(sys.prefix) / 'Library/bin'
    if os.name == 'nt' and dll.is_dir():
        os.environ['PATH'] = str(dll) + os.pathsep + os.environ.get('PATH', '')
        handles.append(os.add_dll_directory(str(dll)))
    metadata = list((Path(sys.prefix) / 'conda-meta').glob('pychrono-*.json'))
    if len(metadata) != 1 or json.loads(metadata[0].read_text())['version'] != '9.0.1':
        raise RuntimeError('this pilot requires the pinned conda PyChrono 9.0.1 environment')
    import pychrono.core as chrono
    return chrono, handles


def chrono_model(model, horizon, reference=False, task=None):
    chrono, handles = chrono_import()
    if reference:
        path = SOURCES / 'chronobench' / task / 'cleaned_truth1.py'
        tree = ast.parse(path.read_text())
        # Keep upstream dynamics and visual shapes. Suppress Irrlicht window and infinite loop.
        keep = [n for n in tree.body if not isinstance(n, ast.While)
                and 'chronoirr' not in ast.unparse(n) and not ast.unparse(n).startswith('vis.')]
        scope = {}
        exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), 'exec'), scope)
        system = scope['sys']
        body = scope['body_1'] if task == 'mass_spring_damper' else scope['pend_1']
    else:
        system = chrono.ChSystemNSC()
        system.SetGravitationalAcceleration(chrono.ChVector3d(*model['gravity']))
        ground = chrono.ChBody()
        ground.SetFixed(True)
        ground.EnableCollision(False)
        system.Add(ground)
        body = chrono.ChBody()
        body.SetMass(model['mass'])
        body.SetInertiaXX(chrono.ChVector3d(*model['inertia']))
        body.SetPos(chrono.ChVector3d(*model['position']))
        body.EnableCollision(False)
        system.AddBody(body)
        if model['kind'] == 'mass_spring_damper':
            spring = chrono.ChLinkTSDA()
            spring.Initialize(body, ground, True, chrono.ChVector3d(*model['body_anchor']),
                              chrono.ChVector3d(*model['ground_anchor']))
            spring.SetRestLength(model['rest_length'])
            spring.SetSpringCoefficient(model['spring_coefficient'])
            spring.SetDampingCoefficient(model['damping_coefficient'])
            system.AddLink(spring)
            body.AddVisualShape(chrono.ChVisualShapeBox(1, 1, 1))
            spring.AddVisualShape(chrono.ChVisualShapeSpring(0.05, 80, 15))
        else:
            joint = chrono.ChLinkLockRevolute()
            joint.Initialize(ground, body, chrono.ChFramed(chrono.ChVector3d(*model['pivot']),
                                                       chrono.ChQuaterniond(1, 0, 0, 0)))
            system.AddLink(joint)
            body.AddVisualShape(chrono.ChVisualShapeCylinder(0.2, 2),
                chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QuatFromAngleY(chrono.CH_PI_2)))
    trajectory = []
    for step in range(round(horizon / 0.001)):
        system.DoStepDynamics(0.001)
        if (step + 1) % 100 == 0:
            p, v = body.GetPos(), body.GetPosDt()
            trajectory.append([system.GetChTime(), p.x, p.y, p.z, v.x, v.y, v.z])
    return {'trajectory': trajectory, 'backend': 'PyChrono 9.0.1', 'gui_rendered': False}


def main():
    request = json.loads(sys.stdin.read())
    with contextlib.redirect_stdout(io.StringIO()):
        if request['suite'] == 'factoryflow':
            normalize = request.get('normalize', True)
            value = factory_reference(request['task'], request['horizon'], normalize) if request['reference'] else factory(request['model'], request['horizon'], normalize)
        else:
            value = chrono_model(request.get('model'), request['horizon'], request['reference'], request['task'])
    print(json.dumps(value, allow_nan=False))


if __name__ == '__main__':
    main()

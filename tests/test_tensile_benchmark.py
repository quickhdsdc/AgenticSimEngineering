"""Analytical mechanics, label isolation and official submission boundaries."""
from pathlib import Path
import shutil
import subprocess
import unittest
from uuid import uuid4

from agenticsim.domain.tensile import validate_model_spec
from agenticsim.runtime.benchmark import seal
from agenticsim.runtime.calibration import load_json, save_json
from agenticsim.runtime.paths import ROOT
from agenticsim.runtime.tensile_benchmark import execute


class TensileTests(unittest.TestCase):
    def native(self, code):
        worker = ROOT/'.venvs/plaid/Scripts/python.exe'
        if not worker.exists():
            self.skipTest('install isolated PLAID worker')
        result = subprocess.run([str(worker),'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_model_contract_rejects_code_and_unbounded_constants(self):
        spec = load_json(ROOT/'examples/tensile2d-model.json')
        validate_model_spec(spec)
        for update in ({'python':'eval(1)'},{'initial_poisson':float('nan')},
                       {'initial_young_scale':0},{'elastic_fit_cases':True},{'neighbors':[500]}):
            with self.assertRaises(ValueError):
                validate_model_spec(dict(spec,**update))

    def test_phase_rejects_tampered_evidence_before_worker(self):
        directory = ROOT/'work/runs/tests'/uuid4().hex
        try:
            (directory/'.evaluation').mkdir(parents=True)
            save_json(directory/'dataset.json',{'scope':'local'})
            seal(directory,'prepare',['dataset.json'])
            save_json(directory/'dataset.json',{'scope':'official'})
            with self.assertRaisesRegex(ValueError,'evidence changed'):
                execute('model',directory,worker_python=Path('missing'))
        finally:
            shutil.rmtree(directory,ignore_errors=True)

    def test_fem_matches_analytical_plane_strain_and_load_linearity(self):
        self.native('''
import numpy as np
from agenticsim.adapters.tensile_fem import PlaneStrainMesh
xy=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
tri=np.array([[0,1,2],[1,3,2]])
mesh=PlaneStrainMesh(xy,tri)
(fields,scalars),diag=mesh.elastic(1000.,0.3,-1.)
expected=np.column_stack((-0.00039*xy[:,0],0.00091*xy[:,1]))
assert np.allclose(fields[:,:2],expected,atol=1e-14),fields
assert np.allclose(fields[:,2:],[0.,1.,0.],atol=1e-12),fields
assert np.allclose(scalars,[1.,0.00091,1.],atol=1e-12)
assert diag['converged'] and diag['relative_equilibrium_residual']<1e-12
assert diag['bc_max_abs']==0.
(twice,_),_=mesh.elastic(1000.,0.3,-2.)
assert np.allclose(twice,2*fields,atol=1e-12)
for bad in (np.array([[0,0,2]]),np.array([[0,1,7]])):
    try: PlaneStrainMesh(xy,bad)
    except ValueError: pass
    else: raise AssertionError('invalid mesh accepted')
''')

    def test_official_pickle_shape_finiteness_and_class_restrictions(self):
        self.native('''
import pickle
import numpy as np
from agenticsim.adapters.tensile_submission import read_predictions,validate_predictions
from agenticsim.domain.plaid import FIELDS,SCALARS
rows=[{**{name:np.ones(i%3+3) for name in FIELDS},**{name:1. for name in SCALARS}} for i in range(200)]
counts=[i%3+3 for i in range(200)]
validate_predictions(read_predictions(pickle.dumps(rows,protocol=4)),counts)
for bad in (rows[:-1],rows+[rows[0]]):
    try: validate_predictions(bad,counts)
    except ValueError: pass
    else: raise AssertionError('nonofficial cohort accepted')
for name,value in (('U1',np.ones(99)),('U1',np.array([np.nan]*3)),('max_U2_top',float('inf'))):
    altered=[dict(row) for row in rows]; altered[0][name]=value
    try: validate_predictions(altered,counts)
    except ValueError: pass
    else: raise AssertionError('invalid output accepted')
try: read_predictions(pickle.dumps(eval))
except ValueError: pass
else: raise AssertionError('pickle code accepted')
''')

    def test_residual_prediction_ignores_target_labels(self):
        self.native('''
import numpy as np
from agenticsim.adapters.tensile_worker import PhysicsResidual
xy=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
samples={i:dict(xy=xy,tri=np.array([[0,1,2],[1,3,2]]),
    features=np.array([-1.,1.,2.,3.,4.,1000.,1.,1.,1.,.5,.5])+i*.01,
    fields=np.ones((4,5))*(i+1),scalars=np.ones(3),
    bc_bottom=np.array([0,1]),bc_top=np.array([2,3]),bc_bottom_left=np.array([0])) for i in range(3)}
model=PhysicsResidual(samples,[0,1],dict(poisson=.3,young_scale=1.))
model.train_residual(); model.method='physics_residual'
before=model.predict(2,2,1)
samples[2]['fields'][:]=1e12; samples[2]['scalars'][:]=-1e12
after=model.predict(2,2,1)
assert all(np.array_equal(a,b) for a,b in zip(before,after))
assert np.all(after[0][[0,1],1]==0) and after[0][0,0]==0
s=after[0][:,2:]; vm=np.sqrt(s[:,0]**2+s[:,1]**2-s[:,0]*s[:,1]+3*s[:,2]**2)
assert after[1][0]==vm.max() and after[1][1]==after[0][[2,3],1].max()
''')

    def test_identification_uses_training_only_and_recovers_known_constants(self):
        self.native('''
import numpy as np
from agenticsim.adapters.tensile_worker import mesh_for,identify
xy=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
samples={i:dict(xy=xy,tri=np.array([[0,1,2],[1,3,2]]),
    features=np.array([-1.-i,1.,2.,3.,4.,1000.,1.,1.,1.,.5,.5]),
    bc_bottom=np.array([0,1]),bc_top=np.array([2,3]),bc_bottom_left=np.array([0]),max_q=0.) for i in range(3)}
for i,sample in samples.items():
    (sample['fields'],sample['scalars']),_=mesh_for(sample).elastic(1100.,.32,sample['features'][0])
samples[2]['fields'][:]=1e12
spec=dict(initial_poisson=.25,initial_young_scale=.9,elastic_fit_cases=2)
result=identify(samples,[0,1],spec)
assert result['training_case_ids']==[0,1]
assert abs(result['poisson']-.32)<1e-7 and abs(result['young_scale']-1.1)<1e-7,result
''')

    def test_metric_weights_cases_equally_despite_mesh_size(self):
        self.native('''
import numpy as np
from agenticsim.adapters.plaid_worker import metrics
class Model:
    samples={1:dict(fields=np.full((3,5),2.),scalars=np.full(3,4.)),
             2:dict(fields=np.full((30,5),2.),scalars=np.full(3,4.))}
    def predict(self,i,*args):
        sample=self.samples[i]; delta=1. if i==1 else 0.
        return sample['fields']+delta,sample['scalars']+delta
score=metrics(Model(),[1,2],dict(neighbors=1,geometry_weight=1))
assert abs(score['rrmse']['U1']-(.125)**.5)<1e-12
assert abs(score['rrmse']['max_U2_top']-(.03125)**.5)<1e-12
''')


if __name__ == '__main__':
    unittest.main()

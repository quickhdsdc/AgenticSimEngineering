"""mbs_isaac — build accurate multibody-system models in Isaac Sim / PhysX from a
declarative spec, then simulate them headless into world-model training data.

Layers:
  spec.py      declarative MBSModel (pure Python, validated, no Isaac needed)
  builder.py   MBSModel -> UsdPhysics articulation on a stage
  simulate.py  headless PhysX rollout -> MoSim-format .npz
  models.py    example/parametric specs (n_link_arm, double_pendulum, ...)

Typical use:
    from mbs_isaac.models import double_pendulum
    from mbs_isaac.simulate import simulate
    simulate(double_pendulum(), episodes=3, steps=2000, out="data/dp.npz")
"""
from .spec import JointSpec, LinkSpec, MBSModel, Shape  # noqa: F401

__all__ = ["MBSModel", "LinkSpec", "JointSpec", "Shape"]

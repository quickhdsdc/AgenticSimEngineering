"""
Example MBS model specs — worked references you can build & simulate, and
templates to copy for your own systems.

Each factory returns a validated MBSModel. `n_link_arm` is parametric: it scales
from a double pendulum (n=2) up to an arbitrary planar chain, which is the ideal
first testbed (energy conservation gives an exact correctness check).
"""
from __future__ import annotations

from .spec import JointSpec, LinkSpec, MBSModel, Shape


def n_link_arm(n=2, link_length=0.5, link_mass=1.0, link_radius=0.03,
               damping=0.0, planar_axis="Y", name=None) -> MBSModel:
    """Planar serial chain of `n` uniform rods hanging from a fixed base.

    Links lie along local +X; revolute joints rotate about `planar_axis` so the
    chain swings in the plane orthogonal to that axis under gravity (-Z).
    Joint i connects the distal end (link_length, 0, 0) of link i-1 to the origin
    of link i. n=2 is a double pendulum.
    """
    name = name or f"arm{n}"
    # Ground/base body (the MBS "inertial frame" body). Required: in a reduced-
    # coordinate articulation the root link's attachment to world is NOT a DoF —
    # anchoring link0 to world with a revolute joint would silently lose that DoF.
    # So we add an explicit base, fix it to world, and hang the chain off it.
    links = [LinkSpec(
        name="base",
        mass=1.0,  # fixed to world -> dynamically irrelevant
        shape=Shape(kind="box", size=(0.05, 0.05, 0.05)),
    )]
    joints = [JointSpec(name="joint_ground", joint_type="fixed", parent="", child="base")]
    for i in range(n):
        links.append(LinkSpec(
            name=f"link{i}",
            mass=link_mass,
            com=(link_length / 2, 0.0, 0.0),
            shape=Shape(kind="cylinder", radius=link_radius, length=link_length, axis="X"),
            shape_offset=(link_length / 2, 0.0, 0.0),
        ))
        joints.append(JointSpec(
            name=f"joint{i}",
            joint_type="revolute",
            parent="base" if i == 0 else f"link{i-1}",
            child=f"link{i}",
            axis=planar_axis,
            parent_anchor=(0.0, 0.0, 0.0) if i == 0 else (link_length, 0.0, 0.0),
            child_anchor=(0.0, 0.0, 0.0),
            damping=damping,
            max_effort=200.0,
        ))
    return MBSModel(name=name, links=links, joints=joints, fixed_base=True).validate()


def double_pendulum(**kw) -> MBSModel:
    return n_link_arm(n=2, name="double_pendulum", **kw)


CATALOG = {
    "double_pendulum": double_pendulum,
    "arm3": lambda: n_link_arm(3, name="arm3"),
    "arm5": lambda: n_link_arm(5, name="arm5"),
}

"""
声明层：多体系统的形式化描述（纯 Python，**不依赖 Isaac**）。

这一层只回答"这个多体系统是什么"：有哪些连杆、各自的惯性参数、用什么关节连接。
builder.py 负责把它落地成 Isaac Sim / PhysX 的 articulation。

【为什么要和 Isaac 完全解耦】
启动一次 Isaac 要 30-60 秒、吃 4-8 GB 内存。把描述和校验做成纯 Python，
就能在几毫秒内验证上百个模型变体的合法性，只在真要仿真时才启动 PhysX。
这个分离在实践中省了大量时间。

【坐标约定】（按教科书 MBS / SIMPACK 的习惯选的）
  - 右手系，重力默认沿 -Z
  - 每个连杆有自己的**局部坐标系**，原点放在该连杆的**近端铰点**
    （即把它连到父件的那个关节处）。几何和质心都在这个局部系里表达
  - 关节连接父件和子件：`parent_anchor` 是铰点在父件局部系的位置，
    `child_anchor` 是同一个物理点在子件局部系的位置（通常就是子件原点 (0,0,0)）
  - 惯量可以显式给（绕质心的主惯性矩），也可以由形状+质量自动算

详细教程见 `docs/guides/MBS建模教程_Isaac实现.md`
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Vec3 = tuple[float, float, float]
Axis = Literal["X", "Y", "Z"]
JointType = Literal["revolute", "prismatic", "fixed"]


@dataclass
class Shape:
    """Primitive collision/visual geometry, used to auto-compute inertia if the
    link does not specify one explicitly. Dimensions are full extents [m]."""
    kind: Literal["box", "cylinder"] = "box"
    size: Vec3 = (0.1, 0.1, 0.1)          # box: (dx,dy,dz)
    radius: float = 0.03                   # cylinder
    length: float = 0.2                    # cylinder length along its local axis
    axis: Axis = "X"                       # cylinder long axis

    def inertia_diag(self, mass: float) -> Vec3:
        """Principal moments of inertia about the shape's COM (uniform density)."""
        if self.kind == "box":
            dx, dy, dz = self.size
            return (
                mass / 12.0 * (dy * dy + dz * dz),
                mass / 12.0 * (dx * dx + dz * dz),
                mass / 12.0 * (dx * dx + dy * dy),
            )
        # solid cylinder: Iaxial = 1/2 m r^2 ; Itransverse = 1/12 m (3r^2 + L^2)
        r, L = self.radius, self.length
        Ia = 0.5 * mass * r * r
        It = mass / 12.0 * (3 * r * r + L * L)
        return {"X": (Ia, It, It), "Y": (It, Ia, It), "Z": (It, It, Ia)}[self.axis]


@dataclass
class LinkSpec:
    name: str
    mass: float
    com: Vec3 = (0.0, 0.0, 0.0)                    # COM in link local frame
    inertia_diag: Optional[Vec3] = None           # about COM; if None -> from shape
    # Orientation of the principal-inertia axes w.r.t. the link frame (w,x,y,z).
    # A real link's inertia tensor is generally NOT diagonal in its own frame;
    # USD stores it as (diagonalInertia, principalAxes). Copying only the diagonal
    # and dropping this rotation silently changes the rotational dynamics.
    principal_axes: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    shape: Shape = field(default_factory=Shape)
    # where geometry is centered in the link frame (visual/collision placement)
    shape_offset: Vec3 = (0.0, 0.0, 0.0)
    # Give the geometry a collider. Turn off when studying pure articulated
    # dynamics: placeholder shapes can self-collide and corrupt the comparison.
    collision: bool = True
    # Explicit initial world pose. Needed for closed-loop mechanisms, whose links
    # cannot be laid out by walking a tree (the loop must be assembled to satisfy
    # its closure equations). If None, the builder derives the position from the
    # joint chain and uses identity orientation.
    init_pos: Optional[Vec3] = None
    init_rot: Optional[tuple[float, float, float, float]] = None  # (w,x,y,z)

    def resolved_inertia(self) -> Vec3:
        return self.inertia_diag if self.inertia_diag is not None else self.shape.inertia_diag(self.mass)


@dataclass
class JointSpec:
    name: str
    joint_type: JointType
    parent: str                                    # parent link name ("" = world)
    child: str                                     # child link name
    axis: Axis = "Z"
    parent_anchor: Vec3 = (0.0, 0.0, 0.0)          # joint pos in parent frame
    child_anchor: Vec3 = (0.0, 0.0, 0.0)           # joint pos in child frame
    # joint frame ORIENTATION in each body, as quaternion (w, x, y, z).
    # Essential for real robots: Franka's joints all use axis "X" but are aimed
    # by these rotations. Identity = axis interpreted in the body's own frame.
    parent_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    child_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # NOTE: per USD convention revolute limits are in DEGREES (prismatic: meters).
    lower: Optional[float] = None
    upper: Optional[float] = None
    damping: float = 0.0                           # joint viscous damping
    # optional built-in PD drive; leave stiffness=0 for pure torque control,
    # which is what world-model data collection needs (action == applied torque)
    drive_stiffness: float = 0.0
    drive_damping: float = 0.0
    max_effort: float = 1.0e6                       # effort/force limit
    # PhysX joint armature: extra inertia added on the joint's own DoF. Robot
    # assets routinely set it (it models rotor/gearbox inertia and stabilises the
    # solver). It does not affect kinematics at all, but changes the dynamics
    # substantially — a classic silent mismatch when copying a model.
    armature: float = 0.0
    joint_friction: float = 0.0
    # PhysX clamps joint speed to this (deg/s for revolute). Robot assets set it,
    # and like armature it is invisible to kinematics but changes the dynamics.
    max_joint_velocity: Optional[float] = None
    # Loop-closure joint: closes a kinematic loop (four-bar, Bricard, ...).
    # PhysX articulations are reduced-coordinate TREES; a loop cannot be part of
    # the tree, so this joint is added as an extra constraint on top of it. It is
    # excluded from the tree/DoF bookkeeping and is subject to constraint drift,
    # which is precisely what the closed-loop benchmarks measure.
    loop_closure: bool = False


@dataclass
class MBSModel:
    name: str
    links: list[LinkSpec]
    joints: list[JointSpec]
    fixed_base: bool = True                        # anchor first body to world
    gravity: Vec3 = (0.0, 0.0, -9.81)
    # Solver effort. PhysX solves constraints iteratively, so accuracy — above all
    # energy behaviour on conservative systems — is bought with iterations. The
    # defaults are game-physics defaults; constrained/closed-loop mechanisms
    # generally need far more.
    solver_position_iterations: int = 32
    solver_velocity_iterations: int = 4

    # ---- validation: catch modelling mistakes before touching PhysX ---------
    def validate(self) -> "MBSModel":
        names = [l.name for l in self.links]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate link names in {names}")
        known = set(names) | {""}
        for j in self.joints:
            if j.parent not in known:
                raise ValueError(f"joint {j.name}: unknown parent '{j.parent}'")
            if j.child not in names:
                raise ValueError(f"joint {j.name}: unknown child '{j.child}'")
            if j.lower is not None and j.upper is not None and j.lower > j.upper:
                raise ValueError(f"joint {j.name}: lower>upper ({j.lower}>{j.upper})")
        for l in self.links:
            if l.mass <= 0:
                raise ValueError(f"link {l.name}: mass must be > 0 (got {l.mass})")
            if any(i <= 0 for i in l.resolved_inertia()):
                raise ValueError(f"link {l.name}: non-positive inertia {l.resolved_inertia()}")
        # articulation 骨架必须是**树**：每个连杆最多只能是一个"非闭环关节"的子件。
        # 标了 loop_closure 的关节是叠加在树之上的额外约束，因此豁免这条检查。
        # （为什么只能是树：见教程第 4 节，减缩坐标的结构性限制）
        child_count: dict[str, int] = {}
        for j in self.joints:
            if j.loop_closure:
                continue
            child_count[j.child] = child_count.get(j.child, 0) + 1
        multi = [k for k, v in child_count.items() if v > 1]
        if multi:
            raise ValueError(
                f"links driven by >1 tree joint (not a tree): {multi}. "
                "If this joint closes a kinematic loop, set loop_closure=True."
            )
        return self

    @property
    def dof(self) -> int:
        """PhysX 口径的**树自由度**。闭环关节是通过约束"减少"自由度而不是增加，
        所以不计入这里；机构的真实自由度（mobility）更低。
        例：四杆机构树自由度 = 3，但真实自由度 = 1。"""
        return sum(1 for j in self.joints
                   if j.joint_type != "fixed" and not j.loop_closure)

    @property
    def total_mass(self) -> float:
        return sum(l.mass for l in self.links)

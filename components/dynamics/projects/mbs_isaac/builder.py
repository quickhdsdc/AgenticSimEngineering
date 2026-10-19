"""
构建层：把声明式的 MBSModel（spec.py）落地成 USD stage 上的 PhysX articulation。

这是"用代码在 NVIDIA 技术栈里搭 MBS 模型"的核心。凡是你在 GUI 里要点的物理量
（质量、惯量、质心、关节轴、限位、驱动），这里全部用代码显式写死，所以模型精确且可复现。

前置条件：必须在 SimulationApp 已创建之后调用（pxr 的 USD physics 库随 Kit 加载）。
`build(model, stage, root_path)` 返回 articulation 根 prim 的路径。

【USD 概念速记】
  prim          : USD 场景树上的一个节点（Stage 面板里看到的每一行）
  XxxAPI.Apply(): 给 prim 贴上一类属性的"能力标签"(schema)，类似加 mixin。
                  贴了 RigidBodyAPI 物理引擎才当它是刚体
  relationship  : USD 里指向别的 prim 的"指针"（关节用它指定连接哪两个刚体）
  articulation  : PhysX 的减缩坐标求解单元，**只能是树形拓扑**（详见教程第 4 节）

详细教程见 `docs/guides/MBS建模教程_Isaac实现.md`
"""
from __future__ import annotations

from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

from .spec import LinkSpec, MBSModel


def _add_geom(stage, link_path: str, link: LinkSpec):
    """Create the visual+collision geometry prim for a link and return it."""
    geom_path = f"{link_path}/geom"
    s = link.shape
    if s.kind == "box":
        geom = UsdGeom.Cube.Define(stage, geom_path)
        geom.CreateSizeAttr(1.0)
        # Cube is unit-sized; scale to full extents
        UsdGeom.Xformable(geom).AddScaleOp().Set(Gf.Vec3f(*[d / 1.0 for d in s.size]))
    else:  # cylinder
        geom = UsdGeom.Cylinder.Define(stage, geom_path)
        geom.CreateRadiusAttr(s.radius)
        geom.CreateHeightAttr(s.length)
        geom.CreateAxisAttr(s.axis)
    UsdGeom.Xformable(geom).AddTranslateOp().Set(Gf.Vec3f(*link.shape_offset))
    if link.collision:
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    return geom


def _make_rigid_link(stage, root_path: str, link: LinkSpec, world_pos: Gf.Vec3d):
    """造一个带质量/惯量的刚体连杆 prim，放在给定的世界位姿上。

    注意：对 articulation 里的连杆，这里设的位姿只是"初始摆放"，PhysX 之后会
    沿关节树重新推算并覆盖它（见教程第 5 节）。真正决定装配的是关节帧。
    """
    link_path = f"{root_path}/{link.name}"
    xform = UsdGeom.Xform.Define(stage, link_path)
    pos = Gf.Vec3d(*link.init_pos) if link.init_pos is not None else world_pos
    UsdGeom.Xformable(xform).AddTranslateOp().Set(pos)
    if link.init_rot is not None:
        w, x, y, z = link.init_rot
        UsdGeom.Xformable(xform).AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    prim = xform.GetPrim()

    UsdPhysics.RigidBodyAPI.Apply(prim)  # 贴"我是刚体"标签
    # 【陷阱 8 相关】禁止休眠。PhysX 会冻结慢速刚体，这会静默毁掉保守系统的仿真：
    # 无阻尼机构先掉能量、然后彻底停住。判别特征是"之后每一步能量完全相同"。
    # （注：四杆那次 58% 能量损失其实是碰撞不是休眠，两者症状不同——
    #   碰撞是一次性突变，休眠是冻结不动。这里设成 0 是防御性的。）
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateSleepThresholdAttr(0.0)
    rb.CreateStabilizationThresholdAttr(0.0)

    # 惯性参数：这四项就是 MBS 里每个 body 要填的全部内容
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(link.mass))                        # 质量
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*link.com))             # 质心（局部系）
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*link.resolved_inertia()))  # 主惯性矩
    # 【陷阱 3】主惯性轴朝向。真实连杆的惯量张量在自身坐标系下一般非对角，
    # USD 用 (对角惯量 + 主轴四元数) 表示。只复制对角线而丢掉这个旋转，
    # 会得到"运动学完美、转动动力学错"的隐蔽 bug。
    pa = link.principal_axes
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(pa[0], Gf.Vec3f(*pa[1:])))

    _add_geom(stage, link_path, link)
    return link_path


def _make_joint(stage, root_path: str, j, link_paths: dict):
    """造一个铰接约束。这是整个 builder 里信息量最大的函数。"""
    jp = f"{root_path}/joints/{j.name}"
    body1 = link_paths[j.child]
    body0 = link_paths.get(j.parent, "") if j.parent else ""  # parent 为空表示连到 world

    if j.joint_type == "fixed":
        joint = UsdPhysics.FixedJoint.Define(stage, jp)
    elif j.joint_type == "prismatic":
        joint = UsdPhysics.PrismaticJoint.Define(stage, jp)
        joint.CreateAxisAttr(j.axis)
    else:
        joint = UsdPhysics.RevoluteJoint.Define(stage, jp)
        joint.CreateAxisAttr(j.axis)

    # body0/body1 是 USD 的 relationship（指针），指明这个约束连接哪两个刚体
    if body0:
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])

    # 【陷阱 7】减缩坐标 articulation 只能表示"树"，闭环关节必须放到它外面：
    # 否则 PhysX 会静默忽略这个约束，机构直接散架（实测四杆残差涨到 2.9 m，杆长才 1 m）。
    # 设了 excludeFromArticulation 之后，PhysX 会把它当作叠加在树之上的
    # "最大坐标约束"来求解 —— 残差立刻降到 7.5 微米。
    # 代价是这类约束靠迭代满足，会有漂移（这正是闭环 benchmark 要测的东西）。
    if getattr(j, "loop_closure", False):
        joint.CreateExcludeFromArticulationAttr(True)

    # 铰点位置：分别在 body0、body1 各自的**局部坐标系**下表达
    joint.CreateLocalPos0Attr(Gf.Vec3f(*j.parent_anchor))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*j.child_anchor))
    # 【最关键 · 教程第 5 节】铰在两个刚体各自局部系里的**朝向**（四元数 w,x,y,z）。
    # 装配信息存在关节里，不存在连杆里 —— PhysX 沿关节树推算连杆位姿并覆盖
    # 连杆自己的 transform。这两行留成单位四元数，整个机构会被摊成一条直线，
    # 而且不报任何错。Franka 全部 7 个关节 axis 都是 "X"，靠的就是这两个四元数定向。
    joint.CreateLocalRot0Attr(Gf.Quatf(j.parent_rot[0], Gf.Vec3f(*j.parent_rot[1:])))
    joint.CreateLocalRot1Attr(Gf.Quatf(j.child_rot[0], Gf.Vec3f(*j.child_rot[1:])))

    if j.joint_type != "fixed":
        # 【陷阱 5】旋转关节的限位在 USD 里单位是**度**，不是弧度
        if j.lower is not None:
            joint.CreateLowerLimitAttr(float(j.lower))
        if j.upper is not None:
            joint.CreateUpperLimitAttr(float(j.upper))
        # 内置 PD 驱动。做 world model 数据采集时要把 stiffness/damping 清零，
        # 否则"动作"实际是位置目标，学到的是闭环系统而非裸动力学。
        dof = "angular" if j.joint_type == "revolute" else "linear"
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), dof)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(j.drive_stiffness))
        drive.CreateDampingAttr(float(j.drive_damping))
        drive.CreateMaxForceAttr(float(j.max_effort))
        # 【陷阱 4】PhysX 专有的关节动力学属性：armature（关节附加惯量，模拟转子/减速器惯量）
        # 和 maxJointVelocity（速度钳制）。这两个都**完全不影响运动学**，只改动力学，
        # 所以复制模型时最容易漏掉，漏了就表现为"几何一模一样但动起来不一样"。
        # 官方 Franka 的 maxJointVelocity=124.6 deg/s，正是我们早先采集数据时
        # 那个神秘的 2.62 rad/s 速度天花板的来源。
        pj = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
        pj.CreateJointFrictionAttr(float(j.joint_friction))
        if j.armature:
            pj.CreateArmatureAttr(float(j.armature))
        if j.max_joint_velocity is not None:
            pj.CreateMaxJointVelocityAttr(float(j.max_joint_velocity))
    return jp


def build(model: MBSModel, stage, root_path: str = "/World") -> str:
    """把 `model` 描述的多体系统建到 stage 上，返回 articulation 根路径。

    连杆摆放：沿关节树从根部累加 parent_anchor 偏移，让每个连杆起始于它的铰点。
    但这只是**初始摆放**，且默认朝向为单位四元数 —— 对姿态非平凡的系统
    （真实机器人、闭环机构）必须用 LinkSpec.init_pos/init_rot 显式给定，
    真正决定装配的是关节帧（见 _make_joint 里 LocalRot 的注释）。
    """
    model.validate()  # 先做纯 Python 校验，把建模错误拦在碰 PhysX 之前

    # ---- 1) 建 articulation 根节点 ----
    art_root = f"{root_path}/{model.name}"
    UsdGeom.Xform.Define(stage, art_root)
    root_prim = stage.GetPrimAtPath(art_root)
    # 贴上这个标签，PhysX 才会把下面的刚体+关节组装成**减缩坐标树**求解
    # （精度和稳定性远好于最大坐标，但拓扑只能是树，见教程第 4 节）
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)

    art_api = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    art_api.CreateSleepThresholdAttr(0.0)  # 同连杆处的休眠说明
    art_api.CreateStabilizationThresholdAttr(0.0)
    # 求解器迭代次数 = **约束精度**旋钮（实测：4→255 迭代把闭环残差从 35μm 降到 6μm，
    # 但很快饱和）。注意它几乎不改善**能量**误差 —— 那个由时间步 dt 控制，
    # 两个旋钮正交，搞混会白烧算力（实测数据见教程第 9 节）。
    art_api.CreateSolverPositionIterationCountAttr(int(model.solver_position_iterations))
    art_api.CreateSolverVelocityIterationCountAttr(int(model.solver_velocity_iterations))
    UsdGeom.Scope.Define(stage, f"{art_root}/joints")

    # ---- 2) 沿关节树推算各连杆的初始世界位置 ----
    pos: dict[str, Gf.Vec3d] = {}
    parent_of = {j.child: j for j in model.joints}

    def world_pos(link_name: str) -> Gf.Vec3d:
        if link_name in pos:
            return pos[link_name]
        j = parent_of.get(link_name)
        if j is None or not j.parent:  # 根连杆放在原点
            pos[link_name] = Gf.Vec3d(0, 0, 0)
        else:
            base = world_pos(j.parent)
            pos[link_name] = base + Gf.Vec3d(*j.parent_anchor)
        return pos[link_name]

    # ---- 3) 造所有刚体，再造所有关节 ----
    link_paths = {}
    for link in model.links:
        link_paths[link.name] = _make_rigid_link(stage, art_root, link, world_pos(link.name))

    for j in model.joints:
        _make_joint(stage, art_root, j, link_paths)

    # ---- 4) 物理场景（重力）----
    scene = UsdPhysics.Scene.Define(stage, f"{root_path}/physicsScene")
    g = Gf.Vec3f(*model.gravity)
    mag = g.GetLength()
    if mag > 0:
        scene.CreateGravityDirectionAttr(g / mag)
        scene.CreateGravityMagnitudeAttr(mag)
    return art_root

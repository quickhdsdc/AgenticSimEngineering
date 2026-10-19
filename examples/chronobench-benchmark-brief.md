# Delegated operator: ChronoBench modeling pilot

Run prepare, author both JSON models from the original prompts plus the
following explicit operator context, then score once. These numeric parameters
are reference assumptions supplied by the operator, not equipment measurements.
Use SI units and the supplied world frame (Y vertical, right-handed).

Each model has `kind` equal to its task ID, `gravity` (3-vector), `mass`,
`inertia` (principal 3-vector), `position` (center of mass, 3-vector).
Initial velocities are zero, ground fixed, collision disabled.

- mass_spring_damper: gravity=(0,0,0), mass=1 kg, inertia=(1,1,1) kg m2,
  position=(-1,-3,0) m. Add `body_anchor`=(0,0,0) and
  `ground_anchor`=(-1,0,0), expressed in their respective body-local frames.
  Ground body pose is identity. Add `rest_length`=1.5 m,
  `spring_coefficient`=50 N/m, `damping_coefficient`=1 N s/m.
  Use a ChLinkTSDA connection, visual box and spring.
- pendulum: gravity=(0,-9.81,0), mass=1 kg, inertia=(0.2,1,1) kg m2,
  position=(1,0,1) m. Add `pivot`=(0,0,1) m in the world frame.
  The revolute axis is world Z (identity joint-frame rotation).
  Use ChLinkLockRevolute and a cylinder visual shape.

The controlled builder creates visual shapes but runs headless for finite
1 s and 2 s horizons at 1 ms steps. GUI, camera/light rendering and arbitrary
Python authoring from the original tasks are outside this pilot. Numeric
trajectory residuals are against retained upstream scripts with GUI loops
removed. Do not claim original J-LLM rubric results or physical validation.

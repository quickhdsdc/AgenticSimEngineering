"""CLI entry: build + simulate one catalog model, headless.

Run with the Isaac venv python (from ):
  projectsisaac_sim.venvScriptspython.exe -m mbs_isaac.run_example --model double_pendulum

Passive (gravity-only) by default -> good for the energy sanity check.
Add --torque 0.4 to inject random excitation and build a world-model dataset.
"""
import argparse

from .models import CATALOG
from .simulate import simulate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="double_pendulum", choices=list(CATALOG))
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--torque", type=float, default=0.0, help="excitation scale; 0=passive")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model = CATALOG[args.model]()
    from .paths import data_path
    out = args.out or data_path(f"{model.name}.npz")
    print(f"model={model.name} DoF={model.dof} links={len(model.links)} mass={model.total_mass:.2f}kg")
    simulate(model, episodes=args.episodes, steps=args.steps, dt=args.dt,
             torque_scale=args.torque, out=out)


if __name__ == "__main__":
    main()

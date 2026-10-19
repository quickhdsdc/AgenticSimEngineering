# Open-loop rollout comparison: MoSim world model vs PhysX ground truth (Franka).
# Feeds the recorded torque sequence to the trained model starting from the true
# initial state and integrates forward with NO re-grounding -> error accumulation curve.
# Run from projects/mosim/:  .venv\Scripts\python.exe eval_rollout_franka.py
import argparse
import glob
import os

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MoSim.src.tools import load_data, load_ode_from_ckpt

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-root", default="./ckpt/franka_isaac")
parser.add_argument("--test", default="../../data/franka/franka_test.npz")
parser.add_argument("--t", type=float, default=0.002)
parser.add_argument("--out", default="../../results/franka")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

# Pick checkpoint with the lowest recorded test loss
ckpts = glob.glob(os.path.join(args.ckpt_root, "**", "*.pth"), recursive=True)
assert ckpts, f"no checkpoints under {args.ckpt_root}"
scored = []
for p in ckpts:
    meta = torch.load(p, map_location="cpu", weights_only=False)
    scored.append((meta.get("test_set_loss", np.inf), meta.get("epoch", -1), p))
scored.sort()
best_loss, best_epoch, best_path = scored[0]
print(f"best ckpt: {best_path} (epoch {best_epoch}, test loss {best_loss:.3e})")

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
model = load_ode_from_ckpt(best_path).to(device).eval()
# Keep the adaptive dopri5 integrator: the learned vector field is stiff, and a
# single fixed rk4 step over 2 ms explodes (single-step MSE 4e7 vs 0.12 with
# dopri5). Loose tolerances match training accuracy (MSE 0.1212 vs 0.1213) and
# are cheaper; the underflow AssertionError on divergence is caught in the loop.
model.if_fast = True

state, action, target, meta = load_data(args.test, device="cpu")
nq, nv = meta["nq"], meta["nv"]
E, T, _ = state.shape
print(f"test set: {E} episodes x {T} steps, dt={args.t}")

with torch.no_grad():
    s = state[:, 0].to(device)
    preds = torch.zeros(E, T, nq + nv)
    T_valid = T
    for n in range(T):
        try:
            s = model(s, action[:, n].to(device), 0.0, args.t)
        except AssertionError as e:  # torchdiffeq 'underflow in dt' on divergence
            print(f"WARNING: integrator underflow at step {n} (t={(n+1)*args.t:.3f}s), truncating rollout ({e})")
            T_valid = n
            break
        if not torch.isfinite(s).all():
            print(f"WARNING: non-finite state at step {n} (t={(n+1)*args.t:.3f}s), truncating rollout")
            T_valid = n
            break
        preds[:, n] = s.cpu()

if T_valid < T:
    preds = preds[:, :T_valid]
    target = target[:, :T_valid]
    T = T_valid
    assert T > 0, "model diverged on the very first step"

truth = target  # (E, T, nq+nv), truth[:, n] = state after step n
err_q = torch.sqrt(((preds[..., :nq] - truth[..., :nq]) ** 2).mean(dim=(0, 2))).numpy()
err_v = torch.sqrt(((preds[..., nq:] - truth[..., nq:]) ** 2).mean(dim=(0, 2))).numpy()
tgrid = (np.arange(T) + 1) * args.t

os.makedirs(args.out, exist_ok=True)
np.savez(
    os.path.join(args.out, "rollout_errors.npz"),
    t=tgrid, err_q=err_q, err_v=err_v, ckpt=best_path,
)

for horizon in [0.1, 0.5, 1.0, 2.0, 4.0]:
    i = min(int(horizon / args.t) - 1, T - 1)
    print(f"horizon {horizon:>4.1f}s: q RMSE {err_q[i]:.4f} rad | v RMSE {err_v[i]:.4f} rad/s")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].semilogy(tgrid, err_q, label="joint position RMSE [rad]")
axes[0].semilogy(tgrid, err_v, label="joint velocity RMSE [rad/s]", alpha=0.8)
axes[0].set_xlabel("prediction horizon [s]")
axes[0].set_ylabel("RMSE over episodes & joints")
axes[0].set_title(f"MoSim open-loop rollout error vs PhysX (E={E})")
axes[0].legend()
axes[0].grid(True, which="both", alpha=0.3)

ep = 0
for j in range(3):
    axes[1].plot(tgrid, truth[ep, :, j], lw=1.5, label=f"q{j+1} PhysX")
    axes[1].plot(tgrid, preds[ep, :, j], "--", lw=1.2, label=f"q{j+1} MoSim")
axes[1].set_xlabel("time [s]")
axes[1].set_ylabel("joint position [rad]")
axes[1].set_title("episode 0, joints 1-3: truth vs open-loop prediction")
axes[1].legend(fontsize=8, ncol=2)
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
out_png = os.path.join(args.out, "rollout_comparison.png")
fig.savefig(out_png, dpi=140)
print(f"saved plot -> {out_png}")

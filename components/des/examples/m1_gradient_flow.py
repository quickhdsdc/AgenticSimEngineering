"""Proof of gradient flow through a discrete-event simulation (M1).

Run: ``uv run python examples/m1_gradient_flow.py``

Demonstrates the core GradientShop claim in miniature: a single scalar loss
(mean queueing delay) computed by a *discrete-event* single-server queue is
differentiated end-to-end w.r.t. the service rate, and that gradient is used to
optimise the rate. No random re-sampling per step -- one backward pass replaces
the trial-and-error that a black-box DES would force (proposal RQ1).
"""

from __future__ import annotations

import torch

from gradientshop.ddes import samplers
from gradientshop.ddes.queue import single_server

N, LAM, WARMUP = 8000, 0.7, 1000


def mean_wait(mu: torch.Tensor, u_arr: torch.Tensor, u_srv: torch.Tensor) -> torch.Tensor:
    interarrival = samplers.exponential(torch.tensor(LAM, dtype=torch.float64), u_arr)
    service = samplers.exponential(mu, u_srv)
    return single_server(interarrival, service).mean_wait(warmup=WARMUP)


def main() -> None:
    g = torch.Generator().manual_seed(0)
    u_arr = samplers.uniform_noise(N, generator=g)
    u_srv = samplers.uniform_noise(N, generator=g)

    print(f"Single-server queue, lambda={LAM}, {N} jobs (warmup {WARMUP})\n")

    # 1) One forward + backward pass: the gradient the LLM adapter will consume.
    mu = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    wq = mean_wait(mu, u_arr, u_srv)
    (grad,) = torch.autograd.grad(wq, mu)
    print(f"  Wq(mu=1.20)        = {wq.item():.4f}")
    print(f"  dWq/dmu (autodiff) = {grad.item():+.4f}   (negative: faster server -> less queueing)\n")

    # 2) Gradient descent on the service rate, fixed common random numbers.
    mu = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([mu], lr=0.05)
    print("  Optimising mu to drive Wq toward 0 (pure gradient steps, no re-sampling):")
    for step in range(0, 21):
        opt.zero_grad()
        loss = mean_wait(mu, u_arr, u_srv)
        loss.backward()
        if step % 5 == 0:
            print(f"    step {step:2d}: mu={mu.item():.3f}  Wq={loss.item():.4f}")
        opt.step()
        with torch.no_grad():
            mu.clamp_(min=LAM + 1e-3)  # keep the queue stable (mu > lambda)


if __name__ == "__main__":
    main()

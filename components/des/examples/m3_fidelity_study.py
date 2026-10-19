"""RQ1 probe: how faithfully does the relaxed dispatch track the discrete one?

Run: ``uv run python examples/m3_fidelity_study.py``

Sweeps the SoftSort temperature tau on a single-machine weighted-completion-time
instance and reports, for each tau:
  * soft_cost  -- the relaxed objective,
  * gap        -- |soft_cost - exact discrete objective|,
  * grad_cos   -- cosine between the smooth gradient and the straight-through
                  (exact-forward) gradient; ~1 means the smooth gradient still
                  points the right way.

The tension RQ1 studies: large tau is smooth but biased (big gap); small tau is
faithful (small gap) but the gradient gets spiky/degenerate (cosine drops).
"""

from __future__ import annotations

import torch

from gradientshop.ddes import dispatch


def main() -> None:
    g = torch.Generator().manual_seed(7)
    n = 12
    processing = torch.rand(n, generator=g, dtype=torch.float64) + 0.2
    weight = torch.rand(n, generator=g, dtype=torch.float64) + 0.2
    scores = torch.randn(n, generator=g, dtype=torch.float64)

    taus = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    curve = dispatch.fidelity_curve(processing, scores, weight, taus)

    print(f"Single machine, 1||sum w_j C_j, n={n}")
    print(f"exact discrete objective = {curve['exact_cost']:.4f}\n")
    print(f"{'tau':>7} {'soft_cost':>11} {'gap':>10} {'gap%':>8} {'grad_cos':>10}")
    for i, tau in enumerate(curve["tau"]):
        gap = curve["gap"][i]
        print(f"{tau:>7.3f} {curve['soft_cost'][i]:>11.4f} {gap:>10.4f} "
              f"{100 * gap / curve['exact_cost']:>7.2f}% {curve['grad_cosine'][i]:>10.4f}")

    print("\nReading: fidelity (gap -> 0) improves as tau shrinks, but the smooth\n"
          "gradient's agreement with the exact-forward gradient degrades at very\n"
          "small tau -- the sweet spot in between is the operating point a coupled\n"
          "LLM adapter should target, and the open question for follow-on work.")


if __name__ == "__main__":
    main()

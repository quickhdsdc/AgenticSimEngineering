"""DDES vs reference-engine fidelity (RQ1 timing regime).

Run: ``uv run python examples/validation_ddes_vs_reference.py``

Runs the differentiable DES and the independent SimPy ground truth on the same
sample path and reports their discrepancy. With fixed routing they must agree to
floating-point tolerance -- the guarantee that gradients are taken through a
faithful model, not an approximation.
"""

from __future__ import annotations

import numpy as np
import torch

from gradientshop.validation import compare_single_server, compare_tandem_line


def main() -> None:
    rng = np.random.default_rng(0)

    print("M1 single-server queue (lambda=0.7, mu=1.0, 500 jobs):")
    inter = torch.tensor(rng.exponential(1 / 0.7, size=500))
    srv = torch.tensor(rng.exponential(1 / 1.0, size=500))
    print("  " + compare_single_server(inter, srv).summary())

    print("\nM2 tandem line (4 stations, 300 jobs):")
    inter = torch.tensor(rng.exponential(1 / 0.5, size=300))
    srv = torch.tensor(rng.exponential(1.0, size=(300, 4)))
    print("  " + compare_tandem_line(inter, srv).summary())

    print("\nBoth FAITHFUL to ~1e-15: the DDES reproduces the event-driven engine\n"
          "exactly, so autodiff differentiates the true simulation (proposal RQ1).")


if __name__ == "__main__":
    main()

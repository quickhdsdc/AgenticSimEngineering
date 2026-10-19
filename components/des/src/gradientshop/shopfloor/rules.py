"""Dispatching rules -- the shopfloor's decision vocabulary.

Classic priority rules (the discrete action space of scheduling agents, e.g. in
Yazdanparast-style remanufacturing flowshops). Each rule maps job attributes to
a **priority score**, oriented so that a *larger* score means "dispatch earlier"
-- matching the descending convention of :func:`gradientshop.ddes.dispatch.soft_sort`.

Two uses:
* A single rule gives a hard baseline dispatch order.
* Stacked as columns, the rules form a differentiable **feature basis**: a policy
  ``theta`` over rules, ``score = priority_matrix @ theta``, lets a learner (or an
  LLM) blend named heuristics -- the policy space of the D2 world model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["Jobs", "RULES", "rule_names", "priority", "priority_matrix"]


@dataclass
class Jobs:
    """A batch of jobs awaiting dispatch. All tensors are ``[N]``."""

    processing: Tensor            # processing / service requirement
    weight: Tensor                # tardiness / value weight
    due: Tensor                   # due date
    arrival: Tensor | None = None  # arrival time (defaults to index order)

    def arrival_or_index(self) -> Tensor:
        if self.arrival is not None:
            return self.arrival
        return torch.arange(self.processing.shape[0], dtype=self.processing.dtype)


# name -> (jobs -> priority score, higher dispatched first)
RULES: dict[str, Callable[[Jobs], Tensor]] = {
    "FIFO": lambda j: -j.arrival_or_index(),          # earliest arrival first
    "SPT": lambda j: -j.processing,                   # shortest processing first
    "LPT": lambda j: j.processing,                    # longest processing first
    "WSPT": lambda j: j.weight / j.processing,        # weight per unit time (opt for sum w_j C_j)
    "EDD": lambda j: -j.due,                           # earliest due date first
    "MINSLACK": lambda j: -(j.due - j.processing),     # least slack first
    "WEIGHT": lambda j: j.weight,                      # heaviest / most important first
    "CR": lambda j: -j.due / j.processing,             # critical ratio: most urgent first
}


def rule_names() -> list[str]:
    return list(RULES)


def priority(rule: str, jobs: Jobs) -> Tensor:
    """Priority score of a single named rule."""
    if rule not in RULES:
        raise KeyError(f"unknown rule {rule!r}; available: {rule_names()}")
    return RULES[rule](jobs)


def priority_matrix(jobs: Jobs, rules: list[str] | None = None,
                    *, standardize: bool = True) -> tuple[list[str], Tensor]:
    """Stack chosen rules into an ``[N, F]`` feature matrix (one column per rule).

    Columns are standardised by default so a policy ``theta`` weights rules on a
    comparable scale. Returns ``(names, matrix)``.
    """
    names = rules if rules is not None else rule_names()
    cols = [priority(name, jobs) for name in names]
    mat = torch.stack(cols, dim=1)
    if standardize:
        mean = mat.mean(dim=0, keepdim=True)
        std = mat.std(dim=0, keepdim=True).clamp_min(1e-8)
        mat = (mat - mean) / std
    return names, mat

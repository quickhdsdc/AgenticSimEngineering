"""D2 validation: the agent<->DDES loop learns a dispatch policy from the
gradient feedback, and the LLM-style agent path runs offline via a fake model."""

from __future__ import annotations

import json

import torch

from gradientshop.agent import (
    DispatchWorldModel,
    GradientDescentAgent,
    PromptedLLMAgent,
    run_coupling_loop,
)


def _instance(n=15, seed=0):
    g = torch.Generator().manual_seed(seed)
    processing = torch.rand(n, generator=g, dtype=torch.float64) + 0.3
    weight = torch.rand(n, generator=g, dtype=torch.float64) + 0.2
    total = processing.sum().item()
    # Due dates spread around the total workload so tardiness is non-trivial.
    due = torch.rand(n, generator=g, dtype=torch.float64) * total
    return processing, weight, due


def test_gradient_agent_loop_reduces_cost():
    """The closed agent<->DDES loop drives the true schedule cost down toward
    the WSPT optimum, learning a dispatch policy purely from replay gradients."""
    processing, weight, due = _instance(seed=1)
    world = DispatchWorldModel(processing, weight, due, tau=0.1, objective="wct")
    agent = GradientDescentAgent(world.n_features, lr=0.3)
    history = run_coupling_loop(world, agent, steps=80)

    from gradientshop.ddes.dispatch import exact_dispatch_cost
    wspt_opt = exact_dispatch_cost(processing, weight / processing, weight).item()

    assert history[-1].objective < history[0].objective
    assert history[-1].objective <= wspt_opt * 1.10   # within 10% of optimum
    # Feedback is well-formed and grounded at every turn.
    for obs in history:
        assert len(obs.gradient) == world.n_features
        assert "weighted completion time" in obs.as_text()


def test_observation_text_is_grounded_and_parses():
    world = DispatchWorldModel(*_instance(seed=2))
    obs = world.evaluate([0.0] * world.n_features)
    text = obs.as_text()
    # Every feature name is mentioned -> the critique is grounded in the model.
    for name in world.feature_names:
        assert name in text
    assert len(obs.suggested_direction()) == world.n_features


def test_prompted_llm_agent_runs_with_fake_backend():
    """A fake 'LLM' that returns a gradient step in JSON drives the same loop,
    exercising the real PromptedLLMAgent prompt/parse path end-to-end."""
    world = DispatchWorldModel(*_instance(seed=3), tau=0.1, objective="wct")

    def fake_llm(messages: list[dict]) -> str:
        # The model would reason over messages[-1]['content']; here we emulate a
        # competent reply by taking a step along the reported descent direction.
        content = messages[-1]["content"]
        assert "Sensitivity of the objective" in content  # got a grounded prompt
        obs = fake_llm.last_obs  # type: ignore[attr-defined]
        step = [t + 0.3 * d for t, d in zip(obs.theta, obs.suggested_direction())]
        return "Here are the revised weights: " + json.dumps(step)

    # Wrap the world so the fake model can see the current observation.
    # Non-zero init avoids the all-equal-scores saddle (see GradientDescentAgent).
    agent = PromptedLLMAgent(fake_llm, world.n_features, initial=[0.05] * world.n_features)

    theta = agent.initial_theta()
    first = last = None
    for t in range(40):
        obs = world.evaluate(theta, step=t)
        fake_llm.last_obs = obs  # type: ignore[attr-defined]
        if t == 0:
            first = obs.objective
        last = obs.objective
        theta = agent.propose(obs)
    assert last < first

"""RQ2 with a real LLM in the driver's seat: a ReAct agent that CHOOSES to call
the DDES tools to optimise a dispatch policy.

Run: ``uv run python examples/d2_react_agent.py``

Needs a configured LLM endpoint: copy ``src/gradientshop/agent/config.toml.example``
to ``src/gradientshop/agent/config.toml`` and fill in a real model (or set
``GS_CONFIG`` / ``GS_LLM_PROFILE`` and the relevant ``*_API_KEY`` env var). Unlike
``d2_llm_ddes_loop.py`` (which hard-codes the gradient step), here the model reads
the verbalised gradient critique and decides the next policy itself.
"""

from __future__ import annotations

import asyncio
import sys

import torch

from opsworld.agent import Agent, build_llm, load_persona
from opsworld.agent.config import default_profile, load_llm
from gradientshop.agent.tools import build_dispatch_session, build_dispatch_tools


def _instance(n=12, seed=4):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(n, generator=g, dtype=torch.float64) + 0.3
    w = torch.rand(n, generator=g, dtype=torch.float64) + 0.2
    d = torch.rand(n, generator=g, dtype=torch.float64) * p.sum().item()
    return p, w, d


async def main() -> None:
    try:
        cfg = load_llm()
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[no LLM configured] {e}\n\nThis demo needs a real endpoint. For an "
              f"offline run of the same coupling, use examples/d2_llm_ddes_loop.py.")
        sys.exit(0)

    print(f"[LLM] profile={default_profile() or 'default'} model={cfg.model}")
    session = build_dispatch_session(*_instance(), tau=0.1, objective="wct")
    agent = Agent(
        llm=build_llm(),
        tools=build_dispatch_tools(),
        system_prompt=load_persona("dispatch_scheduler"),
        max_steps=20,
        on_event=print,
    )
    goal = ("Optimise the dispatch policy to minimise weighted completion time. "
            "First inspect the scenario and validate the simulator, then iterate on "
            "the policy using the gradient feedback. Report the best policy found.")
    session = await agent.run(goal, session)
    print("\nBest policy:", session.get("best"))
    print("Audit log:", session.get("audit_log"))
    print("Final:", session.get("final"))


if __name__ == "__main__":
    asyncio.run(main())

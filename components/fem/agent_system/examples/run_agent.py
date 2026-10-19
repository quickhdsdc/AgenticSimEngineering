"""Agent 端到端:给 sample7 的真实 QuickLink zip → agent 自主调工具 → 出报告(+审计)。

需要真实 LLM 端点(见 config/config.toml);无离线/仿真档。
  · 默认用 config.toml 的 [llm] 默认档;指定档:环境变量 MOLD_LLM_PROFILE=deepseek
  · 自托管 vLLM Qwen:MOLD_CONFIG=config/config.vllm.toml
需安装 openai:  uv pip install --python .venv openai

运行:
  PYTHONUTF8=1 agent_system/.venv/Scripts/python.exe agent_system/examples/run_agent.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mold_agent.harness import Agent, build_llm, load_persona  # noqa: E402
from mold_agent.harness.config import default_profile, load_llm  # noqa: E402
from mold_agent.skills import build_forming_tools              # noqa: E402

# 唯一在 GUI 真实跑过的样件
ZIP = (Path(__file__).resolve().parents[2] / "Samples" / "7-Q2603019"
       / "0220M000ND_stamping-Al 1050 Stamping.zip")
ZIP = str(ZIP)


async def main() -> None:
    cfg = load_llm()
    print(f"[LLM] profile={default_profile() or 'default'}  api_type={cfg.api_type}  model={cfg.model}")
    agent = Agent(
        llm=build_llm(),
        tools=build_forming_tools(),
        system_prompt=load_persona("forming_feasibility_engineer"),
        on_event=print,
    )
    goal = f"请评估这个零件的冲压成型可行性并生成报告。AutoForm QuickLink 导出包:{ZIP}"
    print("=" * 70)
    print("USER:", goal)
    print("=" * 70)
    session = await agent.run(goal)
    print("=" * 70)
    print("报告:", session.get("report_path"))
    print("审计:", session.get("audit_log"))
    print("结论:", session.get("final"))


if __name__ == "__main__":
    asyncio.run(main())

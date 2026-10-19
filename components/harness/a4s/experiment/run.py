"""Running the comparison, and what is held constant while it runs.

One harness, one model, one step budget per task, one repository. The arms
differ in exactly one thing:

    baseline    24 tool schemas resident in every request, no shell
    cli          5 tool schemas, `shell`, and a generated verb synopsis

**Skill injection is off for both by default.** The skills are procedural
knowledge written in `aas` commands: the CLI arm could copy them, the baseline
arm would have to translate them, and that asymmetry would be measured as a
tool-surface effect. Its own contribution is already measured separately (the
DPP chain went from a budget-exhausted 20 steps to a finished 16 when the
matching skill was injected). `--skills` turns it on for both if that combined
condition is what is wanted.

Both arms are told the same thing about the domain, get the same instruction
verbatim, and are judged by the same verifier against the same repository. A
run that reports success without changing the repository fails, in either arm.
"""

from __future__ import annotations

import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..environments.local import LocalEnvironment
from ..kernel.loop import AgentLoop, LoopConfig
from ..kernel.prompt import build_system_prompt
from ..kernel.providers import ModelConfig, build
from ..ledger import Identity, Ledger
from ..tools import ToolRegistry
from .baseline import BaselineRegistry
from .tasks import Context, Task, Verdict

ARMS = ("baseline", "cli")


@dataclass
class Outcome:
    arm: str
    task: str
    repeat: int
    status: str
    passed: bool
    detail: str
    steps: int
    input_tokens: int
    output_tokens: int
    seconds: float
    tool_calls: int
    session: str
    schema_bytes: int


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def build_arm(
    arm: str,
    *,
    endpoint: str,
    workdir: Path,
    ledger: Ledger,
    instruction: str,
    model: ModelConfig,
    skills: bool,
) -> tuple[AgentLoop, ToolRegistry]:
    environment = LocalEnvironment(cwd=str(workdir), env={"A4S_AAS_ENDPOINT": endpoint})
    if arm == "baseline":
        tools: ToolRegistry = BaselineRegistry(environment, ledger, workdir, endpoint)
        prompt = build_system_prompt(
            _skills_root(),
            endpoint=endpoint,
            include_synopsis=False,
            instruction=instruction if skills else None,
            extra=BASELINE_IDENTITY,
        )
    else:
        tools = ToolRegistry(environment, ledger, workdir)
        prompt = build_system_prompt(
            _skills_root(),
            endpoint=endpoint,
            include_synopsis=True,
            instruction=instruction if skills else None,
            shell=environment.shell_name,
        )
    loop = AgentLoop(
        provider=build(model),
        model=model,
        tools=tools,
        ledger=ledger,
        system_prompt=prompt,
        config=LoopConfig(max_steps=1),   # replaced per task by the caller
    )
    return loop, tools


BASELINE_IDENTITY = (
    "You reach the AAS through the tools listed above -- one tool per action. "
    "There is no shell and no command line; if an action has no tool, say so "
    "rather than improvising. Read before you write, and call `finish` with "
    "your report when the task is done."
)


def run_one(
    task: Task,
    arm: str,
    repeat: int,
    *,
    endpoint: str,
    workdir: Path,
    home: Path,
    model_name: Optional[str],
    provider_name: Optional[str],
    skills: bool,
) -> Outcome:
    context = Context(endpoint=endpoint, workdir=workdir)
    try:
        if task.teardown:
            task.teardown(context)
        if task.setup:
            task.setup(context)
    except Exception as exc:  # noqa: BLE001 - a broken fixture is a result
        # A suite that dies on run 11 of 16 loses the ten that worked. The
        # attempt is recorded as an error against the task and the run goes on.
        return Outcome(
            arm=arm, task=task.name, repeat=repeat, status="setup-failed",
            passed=False, detail=f"{type(exc).__name__}: {exc}"[:200],
            steps=0, input_tokens=0, output_tokens=0, seconds=0.0,
            tool_calls=0, session="-", schema_bytes=0,
        )

    session = f"exp-{arm}-{task.name}-{repeat}-{uuid.uuid4().hex[:6]}"
    ledger = Ledger(home / "experiment.db", session, actor=arm)
    try:
        ledger.use_identity(Identity.load_or_create(home / "keys" / f"{arm}.key"))
    except ImportError:
        pass

    model = ModelConfig.from_env(provider_name)
    if model_name:
        model.model = model_name

    loop, tools = build_arm(
        arm,
        endpoint=endpoint,
        workdir=workdir,
        ledger=ledger,
        instruction=task.instruction,
        model=model,
        skills=skills,
    )
    loop.config = LoopConfig(max_steps=task.max_steps)
    schema_bytes = len(json.dumps(tools.schemas()))

    started = time.monotonic()
    try:
        result = loop.run(task.instruction)
        status, summary = result.status, result.summary
        steps, tokens_in, tokens_out = result.steps, result.input_tokens, result.output_tokens
    except Exception as exc:  # noqa: BLE001 - a crashed arm is a result, not a stop
        status, summary = "error", f"{type(exc).__name__}: {exc}"
        steps = tokens_in = tokens_out = 0
    seconds = round(time.monotonic() - started, 1)

    calls = len([e for e in ledger.read(session_id=session, kind="tool.call", limit=200)])
    ledger.close()

    context.summary, context.status = summary or "", status
    try:
        verdict = task.check(context)
    except Exception as exc:  # noqa: BLE001 - a broken verifier is a result too
        verdict = Verdict(False, f"verifier raised {type(exc).__name__}: {exc}"[:200])
    try:
        if task.teardown:
            task.teardown(context)
    except Exception:  # noqa: BLE001 - cleanup failure must not lose the result
        pass

    return Outcome(
        arm=arm, task=task.name, repeat=repeat, status=status,
        passed=verdict.passed, detail=verdict.detail, steps=steps,
        input_tokens=tokens_in, output_tokens=tokens_out, seconds=seconds,
        tool_calls=calls, session=session, schema_bytes=schema_bytes,
    )


def run_suite(
    *,
    endpoint: str,
    workdir: Path,
    home: Path,
    arms: List[str],
    tasks: List[Task],
    repeats: int,
    model_name: Optional[str] = None,
    provider_name: Optional[str] = None,
    skills: bool = False,
    on_result: Any = None,
) -> List[Outcome]:
    outcomes: List[Outcome] = []
    for repeat in range(1, repeats + 1):
        for task in tasks:
            for arm in arms:
                outcome = run_one(
                    task, arm, repeat,
                    endpoint=endpoint, workdir=workdir, home=home,
                    model_name=model_name, provider_name=provider_name, skills=skills,
                )
                outcomes.append(outcome)
                if on_result:
                    on_result(outcome)
    return outcomes


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return round(statistics.mean(values), 1) if values else 0.0


def summarise(outcomes: List[Outcome]) -> Dict[str, Any]:
    """Per-arm aggregates, plus per-task detail.

    Means are reported over *all* attempts, passed or not. Averaging only the
    successes would flatter whichever arm fails on the expensive tasks.
    """
    report: Dict[str, Any] = {"arms": {}, "tasks": {}}
    for arm in sorted({o.arm for o in outcomes}):
        rows = [o for o in outcomes if o.arm == arm]
        report["arms"][arm] = {
            "attempts": len(rows),
            "passed": sum(1 for o in rows if o.passed),
            "steps_mean": _mean([o.steps for o in rows]),
            "input_tokens_mean": _mean([o.input_tokens for o in rows]),
            "output_tokens_mean": _mean([o.output_tokens for o in rows]),
            "seconds_mean": _mean([o.seconds for o in rows]),
            "tool_calls_mean": _mean([o.tool_calls for o in rows]),
            "schema_bytes": rows[0].schema_bytes if rows else 0,
        }
    for task in sorted({o.task for o in outcomes}):
        report["tasks"][task] = {
            arm: {
                "passed": sum(1 for o in outcomes if o.task == task and o.arm == arm and o.passed),
                "of": sum(1 for o in outcomes if o.task == task and o.arm == arm),
                "steps": _mean([o.steps for o in outcomes if o.task == task and o.arm == arm]),
                "in": _mean([o.input_tokens for o in outcomes if o.task == task and o.arm == arm]),
                "out": _mean([o.output_tokens for o in outcomes if o.task == task and o.arm == arm]),
            }
            for arm in sorted({o.arm for o in outcomes})
        }
    return report


def render(outcomes: List[Outcome]) -> str:
    report = summarise(outcomes)
    lines: List[str] = []
    arms = sorted(report["arms"])

    lines.append("Per arm")
    lines.append(f"  {'arm':<10} {'pass':>7} {'steps':>7} {'in':>9} {'out':>8} "
                 f"{'calls':>6} {'sec':>6} {'schema':>8}")
    for arm in arms:
        a = report["arms"][arm]
        lines.append(
            f"  {arm:<10} {a['passed']:>3}/{a['attempts']:<3} {a['steps_mean']:>7} "
            f"{a['input_tokens_mean']:>9} {a['output_tokens_mean']:>8} "
            f"{a['tool_calls_mean']:>6} {a['seconds_mean']:>6} {a['schema_bytes']:>8}"
        )

    lines.append("")
    lines.append("Per task (pass | steps | input tokens)")
    header = f"  {'task':<10}" + "".join(f"{arm:>26}" for arm in arms)
    lines.append(header)
    for task in report["tasks"]:
        row = f"  {task:<10}"
        for arm in arms:
            t = report["tasks"][task][arm]
            row += f"{t['passed']}/{t['of']}  {t['steps']:>5}  {t['in']:>8}".rjust(26)
        lines.append(row)

    failures = [o for o in outcomes if not o.passed]
    if failures:
        lines.append("")
        lines.append("Failures")
        for o in failures:
            lines.append(f"  {o.arm:<10} {o.task:<10} [{o.status}] {o.detail[:100]}")
    return "\n".join(lines)


def to_json(outcomes: List[Outcome]) -> Dict[str, Any]:
    return {"outcomes": [asdict(o) for o in outcomes], "summary": summarise(outcomes)}

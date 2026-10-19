"""Operator context and resumable conversations using the vendored A4S loop."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from uuid import uuid4

from a4s.environments.local import LocalEnvironment
from a4s.kernel.config import load_env
from a4s.kernel.loop import AgentLoop, LoopConfig
from a4s.kernel.prompt import build_system_prompt
from a4s.kernel.providers import ModelConfig, build
from a4s.ledger import Kind, Ledger
from a4s.tools import ToolRegistry

ROOT = Path(__file__).resolve().parents[2]
BRIEF = ROOT / "examples/twin-operator-brief.md"
BRIEFS = {"exploration": BRIEF, "kuka-calibration": ROOT / "examples/kuka-calibration-brief.md",
          "panda-identification": ROOT / "examples/panda-identification-brief.md",
          "screw-benchmark": ROOT / "examples/screw-benchmark-brief.md",
          "semantic-benchmark": ROOT / "examples/semantic-benchmark-brief.md",
          "factoryflow-benchmark": ROOT / "examples/factoryflow-benchmark-brief.md",
          "chronobench-benchmark": ROOT / "examples/chronobench-benchmark-brief.md",
          "plaid-tensile2d": ROOT / "examples/plaid-tensile2d-brief.md",
          "tensile2d-benchmark": ROOT / "examples/tensile2d-benchmark-brief.md"}


class DialogueTools(ToolRegistry):
    """Local file work plus exact, preselected simulation commands.

    Keep dispatch/ledger/ask_user/finish in the existing registry. There is
    no general shell surface in this exploration session.
    """

    def __init__(self, ledger, workspace, commands):
        environment = LocalEnvironment(cwd=str(ROOT), env={
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")})
        self.workspace = workspace.resolve()
        self.commands = commands
        super().__init__(environment, ledger, ROOT, roots=[ROOT])

    def _shell(self, command, timeout=600, **kwargs):
        if command not in self.commands:
            return "error: use one of these exact local commands: " + json.dumps(list(self.commands))
        if kwargs:
            return "error: additional execution arguments are unavailable"
        # These sealed numerical phases exceed the generic shell tool's 120s
        # default on this mesh set. The worker still has its own 600s ceiling.
        if command.startswith("python -m agenticsim tensile "):
            timeout = 650
        return super()._shell(command, timeout=min(max(timeout, 1), 700))

    def _read_file(self, path, **kwargs):
        target = (ROOT / path).resolve()
        documents = {ROOT / "README.md"}
        readable = [ROOT / "examples", self.workspace]
        if target not in documents and not any(target.is_relative_to(p.resolve()) for p in readable):
            return "error: read README.md, examples/, or this session workspace"
        if any(p.startswith(".") for p in target.relative_to(ROOT).parts):
            return "error: private files are unavailable"
        if target.name in {"ledger.db", "report.json", "conversation.json"} or target.suffix == ".db":
            return "error: session control files are unavailable"
        if target.is_relative_to(self.workspace / "artifacts/calibration") and target.name == "excitation-c.csv":
            return "error: held-out samples are reserved for deterministic validation"
        if target.is_relative_to(self.workspace / "artifacts/identification") and re.fullmatch(
                r"panda-seed-(21|22|23|25|26|27|29|30|31|32|33|34|35|36|37|38)\.csv", target.name):
            return "error: held-out measurements are reserved for deterministic validation"
        return super()._read_file(path, **kwargs)

    def _write_file(self, path, content):
        target = (ROOT / path).resolve()
        if not target.is_relative_to(self.workspace / "artifacts"):
            return "error: write only under this session's artifacts/ directory"
        if any(target.is_relative_to(self.workspace / "artifacts" / folder) for folder in ("calibration", "identification", "benchmark", "semantic", "modelbench", "plaid", "tensile")) or any(
                part.startswith(".") for part in target.relative_to(self.workspace).parts):
            return "error: calibration/identification evidence and private files are owned by the CLI"
        return super()._write_file(path, content)


def config_gaps(config):
    gaps = config.missing()
    # The existing provider permits local OpenAI-compatible services without
    # a real key. Its default public endpoint still needs a credential.
    if config.provider == "openai" and not config.api_key and not config.base_url:
        gaps.append("A4S_API_KEY or OPENAI_API_KEY (or a configured local A4S_BASE_URL)")
    if config.provider not in {"openai", "azure", "anthropic"}:
        gaps.append("A4S_PROVIDER must be openai, azure, or anthropic")
    return gaps


def execute_turn(provider, config, workspace, instruction, *, restored=False, max_steps=20, case="exploration"):
    if case not in BRIEFS:
        raise ValueError("unknown dialogue case")
    commands = {
        "python -m agenticsim capabilities",
        "python -m agenticsim study examples/queue-study.json --output "
        + (workspace / "artifacts/studies").relative_to(ROOT).as_posix(),
        "python -m agenticsim study examples/kuka-iiwa14-visual-study.json "
        "--worker-python .venvs/newton/Scripts/python.exe --output "
        + (workspace / "artifacts/studies").relative_to(ROOT).as_posix(),
    }
    if case == "kuka-calibration":
        target = (workspace / "artifacts/calibration").relative_to(ROOT).as_posix()
        commands = {"python -m agenticsim capabilities"} | {
            f"python -m agenticsim calibrate {phase} --directory {target}"
            for phase in ("generate", "model", "fit", "validate")}
    if case == "panda-identification":
        target = (workspace / "artifacts/identification").relative_to(ROOT).as_posix()
        commands = {"python -m agenticsim capabilities"} | {
            f"python -m agenticsim identify {phase} --directory {target}"
            for phase in ("ingest", "model", "fit", "validate")}
    if case == "screw-benchmark":
        target = (workspace / "artifacts/benchmark").relative_to(ROOT).as_posix()
        commands = {"python -m agenticsim capabilities"} | {
            f"python -m agenticsim benchmark {phase} --directory {target}"
            for phase in ("prepare", "fit", "score")}
    if case == "semantic-benchmark":
        target = (workspace / "artifacts/semantic").relative_to(ROOT).as_posix()
        submission = (workspace / "artifacts/semantic-submission.json").relative_to(ROOT).as_posix()
        commands = {f"python -m agenticsim semantic {phase} --directory {target}" for phase in ("prepare", "check")}
        commands.add(f"python -m agenticsim semantic score --directory {target} --submission {submission}")
    if case in ("factoryflow-benchmark", "chronobench-benchmark"):
        suite = case.split("-")[0]
        target = (workspace / "artifacts/modelbench").relative_to(ROOT).as_posix()
        submission = (workspace / "artifacts/model-submission.json").relative_to(ROOT).as_posix()
        commands = {f"python -m agenticsim modelbench prepare --suite {suite} --directory {target}",
                    f"python -m agenticsim modelbench score --suite {suite} --directory {target} --submission {submission}"}
    if case == "plaid-tensile2d":
        target = (workspace / "artifacts/plaid").relative_to(ROOT).as_posix()
        recipe = (workspace / "artifacts/plaid-recipe.json").relative_to(ROOT).as_posix()
        commands = {f"python -m agenticsim plaid {phase} --directory {target}" for phase in ("prepare", "score", "export")}
        commands.add(f"python -m agenticsim plaid fit --directory {target} --recipe {recipe}")
    if case == "tensile2d-benchmark":
        target = (workspace / "artifacts/tensile").relative_to(ROOT).as_posix()
        specification = (workspace / "artifacts/tensile-model.json").relative_to(ROOT).as_posix()
        commands = {f"python -m agenticsim tensile {phase} --directory {target}"
                    for phase in ("prepare", "fit", "score", "export")}
        commands.add(f"python -m agenticsim tensile model --directory {target} --model-spec {specification}")
    # Only verified Study locations from this session can be exported.
    for artifact in (workspace / "artifacts/studies").glob("*/study.json"):
        if not re.fullmatch(r"[a-f0-9-]{36}", artifact.parent.name):
            continue
        commands.add('python -m agenticsim visualize export '
                     + artifact.relative_to(ROOT).as_posix() + ' --output '
                     + (workspace / "artifacts/visualizations").relative_to(ROOT).as_posix())
    with Ledger(workspace / "ledger.db", workspace.name, actor="delegated_operator") as ledger:
        tools = DialogueTools(ledger, workspace, commands)
        extra = (
            "This is a local modeling and monitoring exploration, with no live plant endpoint. "
            "The operator supplied context is delegated, not measured. Use ask_user for missing facts. "
            "Write deliverables under " + (workspace / "artifacts").relative_to(ROOT).as_posix()
            + ". Only the following exact shell commands are enabled:\n"
            + "\n".join(sorted(commands))
        )
        if case == "exploration":
            extra += "\nAfter a Study, finish this turn with its artifact path; export is enabled on the next turn."
        if case == "tensile2d-benchmark":
            extra += (
                "\nUse prepare/model/fit/score/export exactly once in order. Write artifacts/tensile-model.json "
                "using the exact operator-supplied JSON specification in the brief before model. "
                "The worker generates TRI3 physics and identifies constants from training-only elastic cases. "
                "Do not modify the grid or model after scoring. CLI-owned tensile evidence and private data "
                "cannot be written or read. Report frozen local holdout comparisons, identification results, "
                "and the official-format submission path. No official hidden score or physical validation is "
                "available; plastic corrections are learned, not a recovered nonlinear constitutive law. "
                "Write artifacts/tensile-summary.md and finish."
            )
        if case == "kuka-calibration":
            extra += (
                "\nThis case is the synthetic KUKA joint2 calibration workflow, not a Study. "
                "Use generate, model, fit, validate in that order, once each. Read examples/kuka-calibration.json. "
                "Truth and holdout samples are unavailable to the Agent; generated calibration files are CLI-owned. "
                "Report held-out synthetic metrics and physical_validation=not_established. "
                "Write your explanation/intake notes outside artifacts/calibration; do not invent missing measurements."
            )
        if case == "panda-identification":
            extra += (
                "\nThis is public MERL Panda offline identification, not a Study or synthetic KUKA fixture. "
                "Use ingest, model, fit, validate once each in order. Read examples/panda-identification.json. "
                "The held-out measurements and CLI-owned identification evidence are protected. "
                "Report actual dataset metrics, active bounds and physical_validation=not_established. "
                "Provisional units/frame/target mapping and nominal reference assumptions remain explicit."
            )
        if case == "screw-benchmark":
            extra += (
                "\nRun the frozen real screw-driving benchmark: prepare, fit, score once each in order. "
                "All context is supplied in examples/screw-benchmark.json and the brief. "
                "CLI-owned artifacts/benchmark and its private .evaluation data cannot be written. "
                "Read only compact dataset/fit/score evidence; held-out records and labels are unavailable. "
                "Do not revise the specification, grid, model or threshold in response to test results. "
                "Report actual acceptance, distinguish measurements from learned predictions, and compare baselines. "
                "Write artifacts/benchmark-summary.md. Do not claim live OPC/MES, early warning or physical acceptance."
            )
        if case == "semantic-benchmark":
            extra += (
                "\nThis is a synthetic AAS mapping challenge. Use semantic prepare/check, write the sixteen-case "
                "submission outside artifacts/semantic, then score once. The check output is authoritative tool "
                "evidence, not the hidden gold rubric. Do not read private gold or alter cases. "
                "Report actual tool evidence, compare schema-only acceptance, and keep missing facts unknown. "
                "This custom submodel uses project semantic URNs; no official IDTA-template or live API claim."
            )
        if case in ("factoryflow-benchmark", "chronobench-benchmark"):
            extra += (
                "\nGenerate the model yourself from the prepared original prompts and this operator brief. "
                "Prepare once, read artifacts/modelbench/cases.json, write artifacts/model-submission.json, score once. "
                "The JSON submission is an object mapping each task_id to its model. Keep exactly the supplied model fields. "
                "Do not write under artifacts/modelbench or inspect hidden references. Report native residuals and "
                "topology/parameter match separately. These are constrained reference simulation pilots, "
                "not published benchmark scores, physical validation, or verified Irrlicht rendering. "
                "Write artifacts/modelbench-summary.md after scoring."
            )
        prompt = build_system_prompt(ROOT / "components/harness/a4s/skills",
                                     include_synopsis=False, extra=extra)
        loop = AgentLoop(provider=provider, model=config, tools=tools, ledger=ledger,
                         system_prompt=prompt, config=LoopConfig(max_steps=max_steps))
        try:
            if restored:
                result = loop.resume(ledger.replay(workspace.name, prompt), instruction)
            else:
                result = loop.run(instruction)
        except Exception as exc:
            # Provider exceptions may carry private endpoint/request details.
            # Keep the resumable record, and report only the exception type.
            from a4s.kernel.loop import RunResult
            summary = "Model/tool execution failed: " + type(exc).__name__
            ledger.append(Kind.ERROR, {"stage": "dialogue", "error": summary})
            result = RunResult("failed", summary, 0, messages=loop.messages)
        valid, bad = ledger.verify()
        report = {"session": workspace.name, "status": result.status,
                  "summary": result.summary, "steps": result.steps,
                  "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                  "operator_source": "delegated_operator", "ledger_valid": valid,
                  "case": case,
                  "ledger_error": bad, "scope": "local exploration; no live equipment"}
        (workspace / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (workspace / "conversation.json").write_text(json.dumps(result.messages, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Inspect configuration presence without a model call")
    parser.add_argument("--env-file", help="Private config path; values are never printed")
    parser.add_argument("--session", help="Continue the recorded dialogue")
    parser.add_argument("--reply-file", type=Path, help="UTF-8 delegated operator answer, recorded as a new user turn")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--case", choices=tuple(BRIEFS), help="Choose Study exploration, synthetic KUKA or public Panda identification")
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_steps > 40:
        parser.error("max-steps must be between 1 and 40")
    if args.env_file and not Path(args.env_file).is_file():
        parser.error("private configuration file does not exist")
    load_env(args.env_file)
    config = ModelConfig.from_env()
    gaps = config_gaps(config)
    case = args.case or "exploration"
    if not BRIEFS[case].is_file():
        gaps.append("checkout task brief " + str(BRIEFS[case].relative_to(ROOT)))
    if args.check:
        print(json.dumps({"provider": config.provider, "model": config.model,
                          "missing": gaps, "brief_exists": BRIEFS[case].is_file(), "case": case,
                          "mode": "configuration inspection; no LLM call"}, ensure_ascii=False))
        return 0 if not gaps else 2
    if gaps:
        print(json.dumps({"status": "configuration_required", "missing": gaps}))
        return 2
    if bool(args.session) != bool(args.reply_file):
        parser.error("continuing requires both --session and --reply-file")
    session = args.session or "twin-" + uuid4().hex[:12]
    if not re.fullmatch(r"twin-[a-zA-Z0-9_-]{1,64}", session):
        parser.error("invalid session ID")
    workspace = ROOT / "work/runs/twin-dialogues" / session
    if args.session and not (workspace / "ledger.db").is_file():
        parser.error("session does not exist")
    if args.session and (workspace / "report.json").is_file():
        saved_case = json.loads((workspace / "report.json").read_text(encoding="utf-8")).get("case", "exploration")
        if args.case and args.case != saved_case:
            parser.error("case cannot change when continuing a session")
        case = saved_case
    workspace.mkdir(parents=True, exist_ok=True)
    if args.reply_file:
        text = args.reply_file.read_text(encoding="utf-8").strip()
        if not text:
            parser.error("operator answer is empty")
        instruction = "[delegated_operator answer]\n" + text
    else:
        instruction = BRIEFS[case].read_text(encoding="utf-8")
    report = execute_turn(build(config), config, workspace, instruction,
                          restored=bool(args.session), max_steps=args.max_steps, case=case)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"finished", "awaiting_operator"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

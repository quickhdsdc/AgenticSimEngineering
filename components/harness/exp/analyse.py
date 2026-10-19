"""Read an `a4s experiment` result file and report the comparison per task.

`a4s experiment` prints a summary table as it finishes. This reads the JSON it
writes and asks the questions the summary cannot: which tasks separate the
arms, in which direction, and whether the separation survives being looked at
per task rather than averaged.

The averaging is the thing to be careful about. Phase 2 reported 5.4 steps
against 5.8 across eight tasks, which reads as a small uniform effect and was
not one -- retrieval favoured the CLI by 43% of input tokens while
`aas_list_templates` answered in one call what the CLI arm assembled from
several. A mean over tasks that differ in sign is a number with no referent.

Usage:

    python exp/analyse.py exp/results-11tasks.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # `to_json` has wrapped the outcomes differently across versions; accept
    # either a bare list or a dict carrying one, rather than failing on shape.
    if isinstance(raw, dict):
        for key in ("outcomes", "results", "runs"):
            if isinstance(raw.get(key), list):
                return raw[key]
        raise SystemExit(f"{path}: no outcome list in {sorted(raw)}")
    return raw


def by_task(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """task -> arm -> the mean of its repeats."""
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["task"], {}).setdefault(row["arm"], []).append(row)

    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for task, arms in grouped.items():
        out[task] = {}
        for arm, repeats in arms.items():
            n = len(repeats)
            out[task][arm] = {
                "passed": sum(1 for r in repeats if r["passed"]),
                "n": n,
                "steps": sum(r["steps"] for r in repeats) / n,
                "input": sum(r["input_tokens"] for r in repeats) / n,
                "output": sum(r["output_tokens"] for r in repeats) / n,
                "seconds": sum(r["seconds"] for r in repeats) / n,
                "detail": repeats[0]["detail"],
                "status": repeats[0]["status"],
            }
    return out


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "exp/results-11tasks.json")
    rows = load(path)
    tasks = by_task(rows)

    print(f"{len(rows)} runs over {len(tasks)} tasks\n")
    header = (
        f"{'task':<10}{'pass':>10}{'steps':>16}{'input tokens':>22}{'output':>16}"
    )
    print(header)
    print(f"{'':<10}{'base/cli':>10}{'base -> cli':>16}{'base -> cli':>22}{'base -> cli':>16}")

    wins = {"cli": [], "baseline": [], "tie": []}
    for task in sorted(tasks):
        arms = tasks[task]
        b, c = arms.get("baseline"), arms.get("cli")
        if not b or not c:
            print(f"{task:<10}  incomplete ({', '.join(arms)})")
            continue
        pass_cell = f"{b['passed']}/{c['passed']}"
        steps = f"{b['steps']:.1f} -> {c['steps']:.1f}"
        inp = f"{b['input']:,.0f} -> {c['input']:,.0f}"
        outp = f"{b['output']:,.0f} -> {c['output']:,.0f}"
        print(f"{task:<10}{pass_cell:>10}{steps:>16}{inp:>22}{outp:>16}")

        # Attributed on input tokens rather than steps: a step is a round trip
        # whose cost varies with what came back, and the resident tool schemas
        # -- the thing under test -- are paid on input.
        margin = (b["input"] - c["input"]) / max(b["input"], 1)
        if abs(margin) < 0.10:
            wins["tie"].append((task, margin))
        elif margin > 0:
            wins["cli"].append((task, margin))
        else:
            wins["baseline"].append((task, margin))

    print()
    total_b = sum(t["baseline"]["input"] for t in tasks.values() if "baseline" in t)
    total_c = sum(t["cli"]["input"] for t in tasks.values() if "cli" in t)
    failed = [r for r in rows if not r["passed"]]
    print(f"total input tokens: baseline {total_b:,.0f}   cli {total_c:,.0f}   "
          f"({(total_b - total_c) / max(total_b, 1):+.1%} for cli)")
    print(f"failures: {len(failed)}" + (
        "  " + ", ".join(f"{r['arm']}/{r['task']} ({r['status']})" for r in failed[:8])
        if failed else ""))

    print("\nper-task direction, on input tokens, 10% band counted as a tie:")
    for name, label in (("cli", "cli cheaper"), ("baseline", "baseline cheaper"),
                        ("tie", "within 10%")):
        entries = sorted(wins[name], key=lambda e: -abs(e[1]))
        rendered = ", ".join(f"{task} ({margin:+.0%})" for task, margin in entries)
        print(f"  {label:<18} {len(entries):>2}   {rendered}")

    batch = tasks.get("batch")
    if batch and "baseline" in batch and "cli" in batch:
        print("\nthe task the suite was built for (open question 5):")
        b, c = batch["baseline"], batch["cli"]
        print(f"  batch  baseline {b['steps']:.0f} steps / {b['input']:,.0f} in   "
              f"cli {c['steps']:.0f} steps / {c['input']:,.0f} in")
        print(f"         passed: baseline {bool(b['passed'])}, cli {bool(c['passed'])}")
        print("  This is the condition the original suite lacked: thirty values to")
        print("  write, where a composed pipeline can pay for itself and separate")
        print("  calls cannot amortise anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One fixed registered backend per process; JSON in/out, no dynamic entry paths."""
import json
import sys
from contextlib import redirect_stdout
from agenticsim.runtime.registry import evaluator_for
from agenticsim.domain import EngineeringTask


def main():
    payload = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        result = evaluator_for(payload["task"]).evaluate(EngineeringTask(payload["task"]), payload["candidate"])
    print(json.dumps(result.to_dict(), allow_nan=False))


if __name__ == "__main__":
    main()

"""Thin entry point for the MES coordination CLI."""

import argparse
import json
import sys
from .client import Config, MesClient, MesError
from .commands import build_parser, describe_actions
from .executor import execute


def main(argv=None):
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        if args.command_name == "describe":
            result = describe_actions(args.action_words)
        else:
            from ..kernel.config import load_env

            load_env()
            with MesClient(Config.from_env()) as client:
                result = client.clean(execute(args, client))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if isinstance(result, dict) and result.get("verified") is False else 0
    except (MesError, ValueError, argparse.ArgumentTypeError) as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "status": getattr(exc, "status", None),
                    "details": getattr(exc, "details", None),
                    "execution": getattr(exc, "execution", None),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

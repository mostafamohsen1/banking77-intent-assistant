#!/usr/bin/env python3
"""Interactive CLI demo against the served model (vLLM endpoint)."""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "src")

from llm_structured_output.inference.client import IntentClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="banking77-structured")
    parser.add_argument(
        "message", nargs="?", help="One message to classify; omit for interactive mode"
    )
    args = parser.parse_args()

    client = IntentClient(base_url=args.base_url, model=args.model)
    if not client.health():
        raise SystemExit(
            f"Endpoint not reachable at {args.base_url}. Start the server with ./scripts/serve.sh"
        )

    if args.message:
        print(json.dumps(client.predict(args.message).model_dump(), indent=2))
        return

    print("Interactive intent classification (Ctrl-D or 'quit' to exit)")
    while True:
        try:
            message = input("message> ").strip()
        except EOFError:
            break
        if not message or message.lower() in {"quit", "exit"}:
            break
        try:
            print(json.dumps(client.predict(message).model_dump(), indent=2))
        except ValueError as exc:
            print(f"error: {exc}")


if __name__ == "__main__":
    main()

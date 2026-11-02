"""Console entry point: ``miniserve`` — serves the API via uvicorn.

Deliberately small. The app factory stays in :mod:`miniserve.server`; this
module only maps CLI flags to it so ``[project.scripts] miniserve`` works.
Args beyond host/port are rejected (the serving knobs live in the engine env)
so the CLI never drifts from what the server actually honors.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="miniserve",
        description="miniserve — continuous-batching LLM serving engine",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max-queue", type=int, default=None,
        help="max queued requests (default: server MAX_QUEUE)",
    )
    parser.add_argument(
        "--max-inflight", type=int, default=None,
        help="max in-flight generations (default: server MAX_INFLIGHT)",
    )
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover - uvicorn is a hard dep
        raise SystemExit(f"miniserve: uvicorn not installed: {e}")

    args = parser.parse_args()

    import miniserve.server as server

    uvicorn.run(
        server.create_app(
            **({"max_queue": args.max_queue} if args.max_queue else {}),
            **({"max_inflight": args.max_inflight} if args.max_inflight else {}),
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()

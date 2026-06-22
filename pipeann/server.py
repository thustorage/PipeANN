"""Launcher for the PipeANN Milvus-compatible gRPC server.

The gRPC service itself is implemented by the native ``pipeann_milvus_server``
binary (see ``src/server/``). ``setup.py`` builds it and bundles it inside the
installed package at ``pipeann/_bin/``, so the server can be started with a
friendly command once PipeANN is installed::

    pipeann-server --data_dir ./data --port 19530 --threads 8

or equivalently::

    python -m pipeann.server --port 19530

Set ``$PIPEANN_SERVER_BIN`` to point at a different binary (handy for dev
checkouts that build into ``build/`` instead of reinstalling the wheel).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

BINARY_NAME = "pipeann_milvus_server"


def find_server_binary() -> Optional[Path]:
    """Locate the native server binary, or ``None`` if it cannot be found.

    The binary is bundled in ``pipeann/_bin/`` by the build; ``$PIPEANN_SERVER_BIN``
    overrides it for development.
    """
    env = os.environ.get("PIPEANN_SERVER_BIN")
    if env:
        path = Path(env)
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()

    bundled = Path(__file__).resolve().parent / "_bin" / BINARY_NAME
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return bundled.resolve()

    return None


def _binary_not_found_message() -> str:
    return (
        f"Could not find the bundled '{BINARY_NAME}' binary at "
        f"{Path(__file__).resolve().parent / '_bin' / BINARY_NAME}.\n\n"
        "Reinstall PipeANN from source so the server is rebuilt:\n"
        "  pip install -e .\n\n"
        "or point PIPEANN_SERVER_BIN at an existing binary."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeann-server",
        description="Start the PipeANN Milvus-compatible gRPC server.",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory for persisted collections and indexes (default: ./data).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=19530,
        help="Listen port (default: 19530).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Search worker threads; 0 uses the server default (default: 0).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    binary = find_server_binary()
    if binary is None:
        print(_binary_not_found_message(), file=sys.stderr)
        return 1

    cmd = [
        str(binary),
        "--data_dir", args.data_dir,
        "--host", args.host,
        "--port", str(args.port),
        "--threads", str(args.threads),
    ]

    # Replace this process with the server so SIGINT/SIGTERM (which the binary
    # already handles for graceful shutdown) reach it directly. execv only
    # returns on failure.
    if os.name == "posix":
        try:
            os.execv(cmd[0], cmd)
        except OSError as exc:
            print(f"Failed to exec {binary}: {exc}", file=sys.stderr)
            return 1

    # Non-POSIX fallback: spawn and forward the exit code.
    import subprocess

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

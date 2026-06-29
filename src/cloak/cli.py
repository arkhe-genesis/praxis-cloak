"""Console entry point for the cloak package.

``cloak serve`` boots the local FastAPI backend and opens a browser to it. The local
model endpoint is configured via ``--ollama-url``, which is exported as
``CLOAK_OLLAMA_URL`` before the backend imports the library.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import webbrowser
from pathlib import Path

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"


def _load_backend_app():
    """Import and return the FastAPI app object (``backend.main:app``).

    The backend lives in the repo's ``app/`` directory, two levels up from this file
    (``<repo>/src/cloak/cli.py`` -> ``<repo>/app``). We add ``app/`` to
    ``sys.path`` and import the ``backend.main`` module. Raises ``RuntimeError`` with a
    clear message if it cannot be located or imported.
    """
    repo_root = Path(__file__).resolve().parents[2]
    app_dir = repo_root / "app"
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    try:
        module = importlib.import_module("backend.main")
    except Exception as exc:  # ImportError, or anything the backend raises on import
        raise RuntimeError(
            f"Could not import the cloak backend (backend.main) from {app_dir}. "
            "Ensure the app backend is present and the server dependencies are "
            "installed (pip install 'cloak[server]')."
        ) from exc
    app = getattr(module, "app", None)
    if app is None:
        raise RuntimeError("backend.main was imported but exposes no `app` attribute.")
    return app


def _serve(args: argparse.Namespace) -> int:
    # Set the local model endpoint before the backend (and the library) import-load,
    # so span_finder/relevance pick it up from the environment.
    os.environ["CLOAK_OLLAMA_URL"] = args.ollama_url

    try:
        import uvicorn
    except ImportError:
        print(
            "ERROR: uvicorn is not installed. Install the server extras with:\n"
            "    pip install 'cloak[server]'",
            file=sys.stderr,
        )
        return 1

    try:
        app = _load_backend_app()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{args.port}"
    print(f"Starting cloak on {url} (Ollama: {args.ollama_url})")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass  # a missing browser is not fatal
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cloak", description="Cloak PII scrubbing service"
    )
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Run the local backend and open the browser")
    serve.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default 127.0.0.1)"
    )
    serve.add_argument(
        "--port", type=int, default=8765, help="Port to bind to (default 8765)"
    )
    serve.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama chat endpoint (default {DEFAULT_OLLAMA_URL})",
    )
    serve.add_argument(
        "--no-browser", action="store_true", help="Do not open the browser automatically"
    )

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

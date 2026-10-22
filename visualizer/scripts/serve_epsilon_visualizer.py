#!/usr/bin/env python3
"""Serve the pre-built epsilon visualizer (static Vite output) without Node/npm.

Usage (from the repository root):

    npm --prefix visualizer/web ci
    npm --prefix visualizer/web run build
    python3 visualizer/scripts/serve_epsilon_visualizer.py

Then open http://127.0.0.1:8765/ in your browser.

The app loads epsilon_visualizer_data.json from the same directory as index.html
(public/ is copied into dist/ at build time).
"""
import argparse
import http.server
import os
import socketserver
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dist = repo_root / "visualizer" / "web" / "dist"
    if not (dist / "index.html").is_file():
        print(
            "Missing visualizer/web/dist/index.html.\n"
            "Build once (on a machine with Node):\n"
            "  npm --prefix visualizer/web ci\n"
            "  npm --prefix visualizer/web run build\n"
            "Then re-run this script. dist/ is generated locally and is not versioned."
        )
        return 1

    os.chdir(dist)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving {dist} at http://{args.host}:{args.port}/")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

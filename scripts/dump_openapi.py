"""Dump the FastAPI OpenAPI schema to frontend/openapi.json.

Imports the app WITHOUT starting a server, so the `lifespan` never runs and no
DB or credentials are needed (see api/main.py). 
openapi-typescript turns the emitted JSON into TS interfaces that the 
vanilla-TS client imports.

Run from anywhere (needs the Python backend env, e.g. via uv):
    uv run python scripts/dump_openapi.py
"""

import json
from pathlib import Path

from api.main import app

OUTPUT = Path(__file__).resolve().parent.parent / "frontend" / "openapi.json"


def main() -> None:
    schema = app.openapi()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"Wrote OpenAPI schema ({len(schema['paths'])} paths) to {OUTPUT}")


if __name__ == "__main__":
    main()
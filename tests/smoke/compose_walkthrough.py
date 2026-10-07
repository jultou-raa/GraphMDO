"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Replays docs/user-guide/running-optimization.md against a running
`docker compose up` stack. Run with: python tests/smoke/compose_walkthrough.py
"""

import os
import sys

import httpx

GRAPH = os.getenv("GRAPH_URL", "http://localhost:8001")
OPTIMIZATION = os.getenv("OPTIMIZATION_URL", "http://localhost:8003")
VARIABLES = (
    {"kind": "range", "name": "x", "lower": 0.0, "upper": 10.0},
    {"kind": "range", "name": "y", "lower": 0.0, "upper": 10.0},
    {"kind": "state", "name": "f_xy"},
)
CONNECTIONS = (
    ("input", "x", "Paraboloid"),
    ("input", "y", "Paraboloid"),
    ("output", "Paraboloid", "f_xy"),
)


def main() -> int:
    with httpx.Client(timeout=300.0) as client:
        client.post(f"{GRAPH}/clear").raise_for_status()
        for variable in VARIABLES:
            client.post(f"{GRAPH}/variables", json=variable).raise_for_status()
        client.post(f"{GRAPH}/tools", json={"name": "Paraboloid"}).raise_for_status()
        for kind, source, target in CONNECTIONS:
            client.post(
                f"{GRAPH}/connections/{kind}",
                json={"source": source, "target": target},
            ).raise_for_status()

        response = client.post(
            f"{OPTIMIZATION}/optimize",
            json={
                "objectives": [{"name": "f_xy", "minimize": True}],
                "n_init": 2,
                "n_steps": 2,
            },
        )
    if response.status_code != 200:
        print(f"POST /optimize -> {response.status_code}: {response.text}")
        return 1
    body = response.json()
    if "f_xy" not in body["best_objectives"] or len(body["history"]) < 1:
        print(f"Unexpected /optimize response: {body}")
        return 1
    print("Walkthrough OK:", body["best_parameters"], body["best_objectives"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Static checks of the container deployment (Dockerfile, docker-compose.yml).
The stack itself is started by the compose-smoke CI job.
"""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
SERVICE_PORTS = {
    "graph-service": 8001,
    "execution-service": 8002,
    "optimization-service": 8003,
}


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text(encoding="utf-8")


def environment(service: dict) -> dict[str, str]:
    env = service.get("environment", {})
    if isinstance(env, dict):
        return env
    return dict(item.split("=", 1) for item in env)


@pytest.mark.parametrize(("name", "port"), SERVICE_PORTS.items())
def test_services_run_venv_uvicorn_without_uv(compose, name, port):
    command = compose["services"][name]["command"]
    assert isinstance(command, list), "use exec form"
    assert command[0] == "uvicorn"
    assert "uv" not in command
    assert command[command.index("--port") + 1] == str(port)


def test_optimization_service_reaches_graph_and_execution(compose):
    env = environment(compose["services"]["optimization-service"])
    assert env["GRAPH_SERVICE_URL"] == "http://graph-service:8001"
    assert env["EXECUTION_SERVICE_URL"] == "http://execution-service:8002"
    execution_env = environment(compose["services"]["execution-service"])
    assert execution_env["GRAPH_SERVICE_URL"] == "http://graph-service:8001"


def test_every_service_has_a_healthcheck(compose):
    for name, service in compose["services"].items():
        assert "test" in service.get("healthcheck", {}), name
    for name, port in SERVICE_PORTS.items():
        probe = " ".join(compose["services"][name]["healthcheck"]["test"])
        assert f"localhost:{port}/health" in probe


def test_dependencies_wait_for_healthy_services(compose):
    for name in SERVICE_PORTS:
        depends_on = compose["services"][name]["depends_on"]
        assert isinstance(depends_on, dict), name
        for dependency in depends_on.values():
            assert dependency["condition"] == "service_healthy"


def test_image_python_matches_project_python(dockerfile):
    project_python = (ROOT / ".python-version").read_text().strip()
    bases = re.findall(r"^FROM python:(\S+?)-slim", dockerfile, re.MULTILINE)
    assert bases and set(bases) == {project_python}


def test_image_excludes_dev_dependencies(dockerfile):
    syncs = re.findall(r"uv sync[^\n]*", dockerfile)
    assert syncs
    assert all("--no-dev" in sync for sync in syncs)


def test_image_exposes_service_ports(dockerfile):
    exposed = re.findall(r"^EXPOSE (.+)$", dockerfile, re.MULTILINE)
    assert set(" ".join(exposed).split()) >= {str(p) for p in SERVICE_PORTS.values()}

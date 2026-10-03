from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "forge_bench" / "openshell_runtime.py"
SPEC = importlib.util.spec_from_file_location("forge_bench_openshell_runtime_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNTIME = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNTIME
SPEC.loader.exec_module(RUNTIME)

NetworkEndpoint = RUNTIME.NetworkEndpoint
OpenShellPolicy = RUNTIME.OpenShellPolicy
OpenShellRuntime = RUNTIME.OpenShellRuntime
OpenShellSandboxSpec = RUNTIME.OpenShellSandboxSpec
RestRule = RUNTIME.RestRule


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.kwargs: list[dict[str, object]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        self.kwargs.append(dict(kwargs))
        stdout = ""
        if argv[1:4] == ["sandbox", "create", "--name"]:
            stdout = json.dumps({"name": argv[4], "phase": "Ready"})
        elif "--policy-only" in argv:
            stdout = "version: 1\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")


def main() -> int:
    broker = NetworkEndpoint(
        name="robot_broker",
        host="host.openshell.internal",
        port=8765,
        binaries=("/usr/bin/python3",),
        rules=(
            RestRule("GET", "/v1/capabilities"),
            RestRule("GET", "/v1/state"),
            RestRule("POST", "/v1/capture"),
        ),
    )
    policy = OpenShellPolicy(endpoints=(broker,))
    payload = policy.as_dict()
    assert payload["version"] == 1
    assert payload["landlock"]["compatibility"] == "hard_requirement"
    assert payload["process"] == {
        "run_as_user": "sandbox",
        "run_as_group": "sandbox",
    }
    numeric = OpenShellPolicy(user=10000, group=10000).as_dict()
    assert numeric["process"] == {
        "run_as_user": "10000",
        "run_as_group": "10000",
    }
    network = payload["network_policies"]["robot_broker"]
    assert network["binaries"] == [{"path": "/usr/bin/python3"}]
    endpoint = network["endpoints"][0]
    assert endpoint["host"] == "host.openshell.internal"
    assert endpoint["port"] == 8765
    assert endpoint["protocol"] == "rest"
    assert endpoint["enforcement"] == "enforce"
    assert endpoint["rules"][0] == {
        "allow": {"method": "GET", "path": "/v1/capabilities"}
    }

    try:
        NetworkEndpoint(
            name="bad",
            host="example.com",
            port=443,
            binaries=("python",),
            rules=(RestRule("GET", "/"),),
        ).validated()
    except ValueError:
        pass
    else:
        raise AssertionError("relative binary path should be rejected")

    try:
        NetworkEndpoint(
            name="unsupported",
            host="example.com",
            port=443,
            binaries=("/usr/bin/python3",),
            protocol="graphql",
            rules=(RestRule("GET", "/"),),
        ).validated()
    except ValueError as exc:
        assert "protocol-specific rule schemas" in str(exc)
    else:
        raise AssertionError("GraphQL must not use the REST-style policy helper")


    fake = FakeRunner()
    runtime = OpenShellRuntime(executable="openshell", runner=fake)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        policy_path = root / "policy.yaml"
        spec = OpenShellSandboxSpec(
            name="forge-test",
            image="example/hermes:test",
            policy=policy,
            cpu="2",
            memory="4Gi",
            labels={"purpose": "test"},
            providers=("openrouter",),
        )
        created = runtime.create(spec, policy_path=policy_path)
        assert created["name"] == "forge-test"
        written = yaml.safe_load(policy_path.read_text())
        assert written["network_policies"]["robot_broker"]["name"] == "robot_broker"

        runtime.upload("forge-test", root, "/workspace/input")
        proc = runtime.exec(
            "forge-test",
            ["/bin/echo", "hello"],
            workdir="/workspace",
            env={"EXAMPLE": "1"},
            timeout=60,
        )
        assert proc.returncode == 0
        runtime.download("forge-test", "output", root / "download")
        runtime.effective_policy("forge-test", root / "effective.yaml")
        runtime.logs("forge-test", root / "logs.txt")
        runtime.delete("forge-test")

    create = fake.calls[0]
    assert create[:4] == ["openshell", "sandbox", "create", "--name"]
    assert "--policy" in create
    assert "--output" in create and "json" in create
    assert "--detach" in create
    assert "--cpu" in create and "2" in create
    assert "--memory" in create and "4Gi" in create
    assert "--provider" in create and "openrouter" in create
    assert ["--label", "purpose=test"] == create[
        create.index("--label") : create.index("--label") + 2
    ]

    exec_call = next(call for call in fake.calls if call[1:3] == ["sandbox", "exec"])
    assert "--no-login-shell" in exec_call
    exec_index = fake.calls.index(exec_call)
    exec_env = fake.kwargs[exec_index]["env"]
    assert isinstance(exec_env, dict)
    assert exec_env["EXAMPLE"] == "1"
    assert "PATH" in exec_env

    assert "--workdir" in exec_call
    assert "/workspace" in exec_call
    assert "--env" in exec_call
    assert "EXAMPLE=1" in exec_call

    assert fake.calls[-1] == ["openshell", "sandbox", "delete", "forge-test"]
    print("OpenShell runtime contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

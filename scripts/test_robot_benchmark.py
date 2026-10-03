from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from forge_bench.robot_benchmark import (
    BROKER_ROUTES,
    RobotBenchmarkScore,
    VisionJudgment,
    latest_overhead_capture_sha,
    robot_broker_policy,
    score_robot_completion,
)


class FakeJudge:
    def __init__(self, *, inside: bool = True, clear: bool = True) -> None:
        self.inside = inside
        self.clear = clear
        self.calls: list[Path] = []

    def judge(self, image_path: Path) -> VisionJudgment:
        self.calls.append(image_path)
        return VisionJudgment(
            clear=self.clear,
            object_inside_container=self.inside,
            reason="synthetic judgment",
            model="fake-vision",
        )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    policy = robot_broker_policy(
        port=8765,
        python_binaries=("/usr/bin/python3.12",),
    ).as_dict()
    assert policy["process"] == {
        "run_as_user": "1000",
        "run_as_group": "1000",
    }
    network = policy["network_policies"]["soarm101_robot_broker"]
    endpoint = network["endpoints"][0]
    assert endpoint["host"] == "host.openshell.internal"
    assert endpoint["port"] == 8765
    assert endpoint["protocol"] == "rest"
    assert endpoint["enforcement"] == "enforce"
    observed_rules = {
        (row["allow"]["method"], row["allow"]["path"])
        for row in endpoint["rules"]
    }
    expected_rules = {(rule.method, rule.path) for rule in BROKER_ROUTES}
    assert observed_rules == expected_rules
    assert ("POST", "/v1/arm") not in observed_rules
    assert ("POST", "/v1/relax") not in observed_rules
    assert ("POST", "/v1/exec") not in observed_rules

    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        observations = run / "observations"
        observations.mkdir()
        old_image = b"old-overhead"
        final_image = b"fresh-final-overhead"
        (observations / "old.jpg").write_bytes(old_image)
        final_path = observations / "final.jpg"
        final_path.write_bytes(final_image)

        events = run / "broker-events.jsonl"
        rows = [
            {
                "schema_version": 1,
                "request_id": "old",
                "timestamp": 10.0,
                "action": "capture_evidence",
                "request": {"camera": "overhead"},
                "ok": True,
                "duration_s": 0.0,
                "result": {
                    "name": "overhead",
                    "sha256": _sha(old_image),
                },
            },
            {
                "schema_version": 1,
                "request_id": "wrist",
                "timestamp": 11.0,
                "action": "capture_evidence",
                "request": {"camera": "wrist"},
                "ok": True,
                "duration_s": 0.0,
                "result": {
                    "name": "wrist",
                    "sha256": _sha(b"wrist"),
                },
            },
            {
                "schema_version": 1,
                "request_id": "final",
                "timestamp": 12.0,
                "action": "capture_evidence",
                "request": {"camera": "overhead"},
                "ok": True,
                "duration_s": 0.0,
                "result": {
                    "name": "overhead",
                    "sha256": _sha(final_image),
                },
            },
        ]
        events.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        assert latest_overhead_capture_sha(events) == _sha(final_image)

        result = run / "task-result.json"
        result.write_text(
            json.dumps(
                {
                    "task_status": "finished",
                    "evidence_camera": "overhead",
                    "evidence_path": "observations/final.jpg",
                }
            ),
            encoding="utf-8",
        )
        judge = FakeJudge()
        score = score_robot_completion(
            run_dir=run,
            result_path=result,
            broker_events_path=events,
            judge=judge,
        )
        assert isinstance(score, RobotBenchmarkScore)
        assert score.fresh_overhead_evidence is True
        assert score.success is True
        assert score.semantic_judgment is not None
        assert judge.calls == [final_path]

        # A prior overhead image is not acceptable even if the agent cites it.
        result.write_text(
            json.dumps(
                {
                    "task_status": "finished",
                    "evidence_camera": "overhead",
                    "evidence_path": "observations/old.jpg",
                }
            ),
            encoding="utf-8",
        )
        stale_judge = FakeJudge()
        stale = score_robot_completion(
            run_dir=run,
            result_path=result,
            broker_events_path=events,
            judge=stale_judge,
        )
        assert stale.fresh_overhead_evidence is False
        assert stale.success is False
        assert stale_judge.calls == []

        # No semantic judge means evidence can be verified without inventing success.
        result.write_text(
            json.dumps(
                {
                    "task_status": "finished",
                    "evidence_camera": "overhead",
                    "evidence_path": "/sandbox/observations/final.jpg",
                }
            ),
            encoding="utf-8",
        )
        evidence_only = score_robot_completion(
            run_dir=run,
            result_path=result,
            broker_events_path=events,
            judge=None,
        )
        assert evidence_only.fresh_overhead_evidence is True
        assert evidence_only.success is None

    skill = Path("robot_tasks/skills/soarm101-robot-camera/SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "python3 robotctl.py capture overhead" in skill
    assert "vision_analyze" in skill
    assert "task-result.json" in skill
    assert "soarm101 agent arm" not in skill
    assert "soarm101 move-" not in skill

    print("Robot benchmark contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

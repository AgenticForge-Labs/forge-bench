from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .harbor_execution import (
    SUPPORTED_HARBOR_ENVIRONMENTS,
    execution_from_plan,
    require_execution_dependencies,
)
from .harbor_runtime import (
    load_harbor_plan,
    materialize_harbor_trial_configs,
    run_harbor_plan,
)
from .telemetry_archive import read_manifest
from .telemetry_contracts import CaptureProfile
from .local_resources import (
    LocalProcessResourceSampler,
    record_local_resources_not_applicable,
)
from .telemetry_recorder import ExperimentTelemetryRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute an explicit Forge Bench run plan through Harbor on local "
            "Docker or Modal without changing Forge cell identity."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trials-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--environment",
        choices=SUPPORTED_HARBOR_ENVIRONMENTS,
        default=None,
        help="Execution environment override. Defaults to the plan value (docker).",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=1,
        help="Maximum explicit Forge cells to execute concurrently.",
    )
    parser.add_argument("--cpus", type=int, help="CPU override per Harbor trial.")
    parser.add_argument("--memory-mb", type=int, help="Memory override per Harbor trial.")
    parser.add_argument("--storage-mb", type=int, help="Storage override per Harbor trial.")
    parser.add_argument("--gpus", type=int, help="GPU count override per Harbor trial.")
    parser.add_argument(
        "--telemetry-dir",
        type=Path,
        help=(
            "Raw telemetry archive root. By default a unique experiment archive "
            "is created beside the normalized Harbor results."
        ),
    )
    parser.add_argument(
        "--experiment-id",
        help=(
            "Stable experiment identifier for telemetry. If omitted, Forge "
            "creates one from UTC time plus the source plan hash."
        ),
    )
    parser.add_argument(
        "--telemetry-profile",
        choices=[profile.value for profile in CaptureProfile],
        default=CaptureProfile.MAXIMAL.value,
        help="Telemetry capture profile recorded in the raw archive.",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable the raw lifecycle telemetry recorder for this execution.",
    )
    parser.add_argument(
        "--resource-sample-interval",
        type=float,
        help=(
            "Local host/process CPU/RAM sampling interval in seconds. Defaults "
            "to 1.0 for maximal, 2.0 for standard, and disabled for minimal. "
            "Only applies to local Docker execution in this layer."
        ),
    )
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help=(
            "Resolve Harbor TrialConfig objects and execution provenance, then "
            "exit without provisioning Docker/Modal or running agents."
        ),
    )
    return parser.parse_args()


def _execution_manifest(
    *,
    plan: dict,
    plan_path: Path,
    output: Path,
    execution,
    materialize_only: bool,
    telemetry: dict | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "kind": "forge-bench-harbor-execution",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": str(plan_path.resolve()),
        "plan_run_plan_sha256": plan.get("run_plan_sha256"),
        "trial_count": int(plan.get("trial_count", 0)),
        "scientific_cell_identity_changed": False,
        "mode": "materialize-only" if materialize_only else "execute",
        "execution": execution.as_dict(),
        "telemetry": telemetry,
        "results": str(output),
    }


async def _main_async(args: argparse.Namespace) -> int:
    plan = load_harbor_plan(args.plan)
    execution = execution_from_plan(
        plan,
        environment=args.environment,
        n_concurrent=args.n_concurrent,
        cpus=args.cpus,
        memory_mb=args.memory_mb,
        storage_mb=args.storage_mb,
        gpus=args.gpus,
    )
    output = (
        args.output or args.trials_dir.parent / "forge-harbor-results.json"
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_name(output.stem + "-execution.json")

    telemetry_settings = (
        None
        if args.materialize_only or args.no_telemetry
        else _resolve_telemetry_settings(args, plan, output)
    )
    execution_manifest = _execution_manifest(
        plan=plan,
        plan_path=args.plan,
        output=output,
        execution=execution,
        materialize_only=args.materialize_only,
        telemetry=telemetry_settings,
    )
    _write_json(manifest_path, execution_manifest)

    if args.materialize_only:
        materialized = await materialize_harbor_trial_configs(
            plan,
            trials_dir=args.trials_dir,
            execution=execution,
        )
        payload = [
            {
                "forge_cell_id": intent["forge_cell_id"],
                "forge_run_index": intent["forge_run_index"],
                "trial_config": config.model_dump(mode="json"),
            }
            for intent, config in materialized
        ]
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Materialized {len(payload)} Harbor trial configs: {output}")
        print("Execution manifest:", manifest_path)
        print("No Docker/Modal environment or agent/model call was started.")
        return 0

    require_execution_dependencies(execution)

    recorder = None
    if telemetry_settings is not None:
        try:
            recorder = ExperimentTelemetryRecorder(
                telemetry_settings["root"],
                experiment_id=telemetry_settings["experiment_id"],
                capture_profile=CaptureProfile(telemetry_settings["capture_profile"]),
            )
            telemetry_settings["active"] = True
        except Exception as exc:
            telemetry_settings["active"] = False
            telemetry_settings["initialization_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            print(
                "WARNING: telemetry initialization failed; benchmark execution "
                "will continue without raw lifecycle telemetry:",
                telemetry_settings["initialization_error"],
            )

    try:
        if recorder is None:
            _harbor, forge_results = await run_harbor_plan(
                plan,
                trials_dir=args.trials_dir,
                execution=execution,
            )
        else:
            with recorder:
                sampler = _start_local_resource_sampler(
                    recorder,
                    execution_environment=execution.environment,
                    requested_interval=args.resource_sample_interval,
                    telemetry_settings=telemetry_settings,
                )
                try:
                    _harbor, forge_results = await run_harbor_plan(
                        plan,
                        trials_dir=args.trials_dir,
                        execution=execution,
                        telemetry=recorder,
                    )
                finally:
                    if sampler is not None:
                        stats = sampler.stop()
                        telemetry_settings["resource_sampling"]["stats"] = asdict(stats)
    finally:
        if telemetry_settings is not None:
            if recorder is not None:
                telemetry_settings["degraded"] = recorder.degraded
                telemetry_settings["issue_count"] = len(recorder.issues)
                telemetry_settings["sealed_state"] = (
                    recorder.sealed_state.value
                    if recorder.sealed_state is not None
                    else None
                )
            execution_manifest["telemetry"] = telemetry_settings
            _write_json(manifest_path, execution_manifest)

    output.write_text(
        json.dumps([asdict(result) for result in forge_results], indent=2) + "\n",
        encoding="utf-8",
    )
    usable = sum(result.valid for result in forge_results)
    solved = sum(result.resolved for result in forge_results if result.valid)
    print(
        f"Harbor {execution.environment} execution complete: "
        f"{usable}/{len(forge_results)} usable; {solved} resolved"
    )
    print("Concurrency:", execution.n_concurrent)
    print("Forge-normalized results:", output)
    print("Execution manifest:", manifest_path)
    if telemetry_settings is not None:
        print("Telemetry archive:", telemetry_settings["root"])
        if telemetry_settings.get("degraded"):
            print(
                "WARNING: telemetry archive was sealed partial because recorder "
                f"issues occurred ({telemetry_settings.get('issue_count', 0)} issue(s))."
            )
    return 0 if usable == len(forge_results) else 1


def _resolve_telemetry_settings(
    args: argparse.Namespace,
    plan: dict,
    output: Path,
) -> dict:
    capture_profile = CaptureProfile(args.telemetry_profile)
    requested_root = args.telemetry_dir.resolve() if args.telemetry_dir else None

    if requested_root is not None and (requested_root / "archive_manifest.json").exists():
        existing = read_manifest(requested_root / "archive_manifest.json")
        if args.experiment_id and args.experiment_id != existing.experiment_id:
            raise SystemExit(
                "--experiment-id does not match the existing telemetry archive: "
                f"{existing.experiment_id}"
            )
        if capture_profile != existing.capture_profile:
            raise SystemExit(
                "--telemetry-profile does not match the existing telemetry archive: "
                f"{existing.capture_profile.value}"
            )
        experiment_id = existing.experiment_id
        root = requested_root
    else:
        experiment_id = args.experiment_id or _new_experiment_id(plan)
        root = (
            requested_root
            if requested_root is not None
            else output.parent / "telemetry" / experiment_id
        )

    return {
        "requested": True,
        "active": False,
        "root": str(root.resolve()),
        "experiment_id": experiment_id,
        "capture_profile": capture_profile.value,
        "durability": "lifecycle_fsync_each_record; resource_streams_periodic_fsync",
        "resource_sampling": {
            "requested_interval_seconds": args.resource_sample_interval,
            "active": False,
        },
    }


def _start_local_resource_sampler(
    recorder: ExperimentTelemetryRecorder,
    *,
    execution_environment: str,
    requested_interval: float | None,
    telemetry_settings: dict,
) -> LocalProcessResourceSampler | None:
    settings = telemetry_settings["resource_sampling"]
    if execution_environment != "docker":
        reason = (
            "local host/process telemetry is not used for Modal because the "
            "orchestrator host would not represent the remote trial sandbox; "
            "Modal resource parity is implemented in a later layer"
        )
        try:
            record_local_resources_not_applicable(recorder, reason=reason)
        except Exception as exc:
            recorder.issue("local_resources.not_applicable", exc)
        settings.update(
            {
                "active": False,
                "scope": "remote_modal_not_sampled",
                "reason": reason,
            }
        )
        return None

    interval = _resource_sampling_interval(
        recorder.capture_profile,
        requested_interval=requested_interval,
    )
    if interval is None:
        reason = "minimal capture profile records lifecycle evidence only"
        settings.update(
            {
                "active": False,
                "scope": "local_host_and_forge_process_tree",
                "reason": reason,
            }
        )
        return None

    try:
        sampler = LocalProcessResourceSampler(
            recorder,
            interval_seconds=interval,
        )
        sampler.start()
    except Exception as exc:
        recorder.issue("local_resources.initialize", exc)
        settings.update(
            {
                "active": False,
                "scope": "local_host_and_forge_process_tree",
                "initialization_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return None

    settings.update(
        {
            "active": True,
            "scope": "local_host_and_forge_process_tree",
            "interval_seconds": interval,
            "container_workload_included": False,
        }
    )
    return sampler


def _resource_sampling_interval(
    profile: CaptureProfile,
    *,
    requested_interval: float | None,
) -> float | None:
    if requested_interval is not None:
        if requested_interval <= 0:
            raise SystemExit("--resource-sample-interval must be > 0")
        if profile == CaptureProfile.MINIMAL:
            raise SystemExit(
                "--resource-sample-interval cannot be used with "
                "--telemetry-profile minimal"
            )
        return float(requested_interval)
    if profile == CaptureProfile.MAXIMAL:
        return 1.0
    if profile == CaptureProfile.STANDARD:
        return 2.0
    return None


def _new_experiment_id(plan: dict) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    plan_hash = str(plan.get("run_plan_sha256") or "unhashed")[:12]
    return f"forge-{timestamp}-{plan_hash}"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    return asyncio.run(_main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

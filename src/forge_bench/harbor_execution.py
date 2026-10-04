from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SUPPORTED_HARBOR_ENVIRONMENTS = ("docker", "modal")


@dataclass(frozen=True)
class HarborExecutionConfig:
    """Execution-only settings applied beneath the Forge scientific plan."""

    environment: str = "docker"
    n_concurrent: int = 1
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int | None = None

    def validated(self) -> "HarborExecutionConfig":
        environment = str(self.environment).strip().lower()
        if environment not in SUPPORTED_HARBOR_ENVIRONMENTS:
            raise ValueError(
                "Forge Bench Harbor execution currently supports environment "
                f"{SUPPORTED_HARBOR_ENVIRONMENTS}; got {self.environment!r}"
            )
        if int(self.n_concurrent) < 1:
            raise ValueError("n_concurrent must be >= 1")
        for name, value in (
            ("cpus", self.cpus),
            ("memory_mb", self.memory_mb),
            ("storage_mb", self.storage_mb),
            ("gpus", self.gpus),
        ):
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be >= 1 when supplied")
        return HarborExecutionConfig(
            environment=environment,
            n_concurrent=int(self.n_concurrent),
            cpus=None if self.cpus is None else int(self.cpus),
            memory_mb=None if self.memory_mb is None else int(self.memory_mb),
            storage_mb=None if self.storage_mb is None else int(self.storage_mb),
            gpus=None if self.gpus is None else int(self.gpus),
        )

    def environment_kwargs(self) -> dict[str, Any]:
        config = self.validated()
        return {
            "override_cpus": config.cpus,
            "override_memory_mb": config.memory_mb,
            "override_storage_mb": config.storage_mb,
            "override_gpus": config.gpus,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self.validated())


def execution_from_plan(
    plan: dict[str, Any],
    *,
    environment: str | None = None,
    n_concurrent: int = 1,
    cpus: int | None = None,
    memory_mb: int | None = None,
    storage_mb: int | None = None,
    gpus: int | None = None,
) -> HarborExecutionConfig:
    return HarborExecutionConfig(
        environment=str(environment or plan.get("environment") or "docker"),
        n_concurrent=n_concurrent,
        cpus=cpus,
        memory_mb=memory_mb,
        storage_mb=storage_mb,
        gpus=gpus,
    ).validated()


def require_execution_dependencies(config: HarborExecutionConfig) -> None:
    """Fail early before a paid/cloud run when optional runtime pieces are absent."""
    resolved = config.validated()
    if resolved.environment != "modal":
        return

    try:
        import modal  # noqa: F401
        import dockerfile_parse  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Modal execution requires Harbor's modal extra. Run with "
            "'uv run --with \"harbor[modal]==0.23.0\" forge-bench-harbor ...' "
            "or install harbor[modal]==0.23.0 in the execution environment."
        ) from exc

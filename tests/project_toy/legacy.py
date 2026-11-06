"""A project of a subclass which overrides `run_experiments` with another signature."""

from pathlib import Path
from typing import Any

from sbmlsim.project import Project
from tests.project_toy import PROJECT


class LegacyProject(Project):
    """A project whose `run_experiments` has its own signature and result."""

    def run_experiments(  # ty: ignore[invalid-method-override]
        self, experiments: Any = None, output_dir: Path | None = None
    ) -> Path:
        """Fail: the run layer must call the base implementation."""
        raise AssertionError("the override of the subclass was called")


LEGACY = LegacyProject(
    **{name: getattr(PROJECT, name) for name in type(PROJECT).model_fields}
)

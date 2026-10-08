"""The source of a model, a file or the SBML itself."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Class for keeping track of the resolved sources."""

    source: str
    path: Path | None = None  # if source is a path
    content: str | None = None  # if source is the SBML itself

    def is_content(self) -> bool:
        """Check if the source is Content."""
        return self.content is not None

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dict.

        Used for serialization.
        """
        return {
            "source": str(self.source),
            "path": str(self.path) if self.path else None,
            "content": str(self.content),
        }

    @classmethod
    def from_source(
        cls, source: Union["Source", str, Path], base_dir: Path | None = None
    ) -> "Source":
        """Resolve the source string.

        Args:
            source: path of the model or the SBML itself.
            base_dir: directory a relative path is resolved against, the
                current working directory by default.

        Returns:
            The resolved source.

        Raises:
            OSError: if the path of the source does not exist.
        """
        if isinstance(source, Source):
            return source

        path: Path | None = None
        content: str | None = None

        if isinstance(source, str) and source.lstrip().startswith("<"):
            # the SBML itself
            content = source

        # is path
        if content is None:
            # without a base_dir the current working directory is the base
            path = Path(base_dir) / Path(source) if base_dir else Path(source)
            path = path.resolve()
            if not path.exists():
                raise OSError(
                    f"Path '{path}' for model source '{source}' does not exist."
                )

        return Source(str(source), path, content)

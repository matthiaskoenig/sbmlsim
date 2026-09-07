"""Reports."""

import logging

logger = logging.getLogger(__name__)


class Report:
    """Reports of simulation experiments.

    Collections of data generators.
    """

    def __init__(
        self, sid: str, name: str | None = None, datasets: dict[str, str] | None = None
    ):
        """Construct report."""
        self.sid: str = sid
        self.name: str | None = name
        self.datasets: dict[str, str] = datasets if datasets is not None else {}

    def add_dataset(self, label: str, data_id: str) -> None:
        """Add dataset for given label."""
        if label in self.datasets:
            logger.warning(
                "label '%s' does already exist in report '%s'", label, self.sid
            )
        self.datasets[label] = data_id

"""Reports."""

import logging

logger = logging.getLogger(__name__)


class Report:
    """Reports of simulation experiments.

    Collections of data generators.
    """

    def __init__(
        self,
        sid: str,
        name: str | None = None,
        datasets: dict[str, str] | None = None,
    ):
        """Construct report.

        Args:
            sid: Identifier of the report.
            name: Name of the report.
            datasets: Mapping of labels to data identifiers.
        """
        self.sid: str = sid
        self.name: str | None = name
        self.datasets: dict[str, str] = datasets if datasets is not None else {}

    def add_dataset(self, label: str, data_id: str) -> None:
        """Add dataset for given label.

        Args:
            label: Label of the dataset in the report.
            data_id: Identifier of the data.
        """
        if label in self.datasets:
            logger.warning(
                "label '%s' does already exist in report '%s'", label, self.sid
            )
        self.datasets[label] = data_id

"""Reports."""

from pymetadata import log


logger = log.get_logger(__name__)


class Report:
    """Reports of simulation experiments.

    Collections of data generators.
    """

    def __init__(self, sid: str, name: str = None, datasets: dict[str, str] = None):
        """Construct report."""
        self.sid: str = sid
        self.name: str = name

        if datasets is None:
            self.datasets = {}

        self.datasets: dict[str, str] = datasets

    def add_dataset(self, label: str, data_id: str) -> None:
        """Add dataset for given label."""
        if label in self.datasets:
            logger.warning(f"label '{label}' does already exist in report '{self.sid}'")
        self.datasets[label] = data_id

"""Helpers related to logging."""
import coloredlogs  # type: ignore


coloredlogs.install(level="INFO", fmt="%(levelname)s %(message)s")


class bcolors:
    """Colors for styling log."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BGWHITE = "\033[47m"
    BGBLACK = "\033[49m"
    WHITE = "\033[37m"
    BLACK = "\033[30m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

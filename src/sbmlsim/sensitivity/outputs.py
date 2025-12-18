from dataclasses import dataclass
from typing import Optional


@dataclass
class SensitivityOutput:
    """Output measurement for SensitivityAnalysis."""
    uid: str
    name: str
    unit: Optional[str]

"""ECG-PACE: adaptive, hypothesis-guided ECG evidence acquisition."""

from .agent import PACEAgent
from .config import PACEConfig
from .models import Action, Evidence, ECGRecord, InitialInterpretation, ReportEdit, RunResult

__version__ = "0.1.0"
__all__ = [
    "PACEAgent", "PACEConfig", "Action", "Evidence", "ECGRecord",
    "InitialInterpretation", "ReportEdit", "RunResult",
]

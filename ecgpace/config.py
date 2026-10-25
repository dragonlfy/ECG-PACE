"""Search and safety settings. Defaults follow the paper's configuration table."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PACEConfig:
    max_actions: int = 4
    cost_weight: float = 0.10
    reliability_weight: float = 0.20
    stop_threshold: float = 0.05
    revision_threshold: float = 0.15
    minimum_evidence_reliability: float = 0.50

    def __post_init__(self) -> None:
        if not 1 <= self.max_actions <= 4:
            raise ValueError("max_actions must be between 1 and 4")
        for name in ("cost_weight", "reliability_weight", "stop_threshold", "revision_threshold"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.minimum_evidence_reliability <= 1:
            raise ValueError("minimum_evidence_reliability must be between 0 and 1")

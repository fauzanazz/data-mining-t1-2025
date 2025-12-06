"""Core module containing interfaces and base classes for SP mining."""

from sp_mining.core.interfaces import (
    SPAlgorithm,
    SequenceLoader,
    SequenceTransformer,
    SPEvaluator,
    SPPipeline,
    Sequence,
    SequentialPattern,
    SequentialRule,
    SPAlgorithmResult,
    SPEvaluationResult,
)
from sp_mining.core.container import SPContainer
from sp_mining.core.pipeline import SPMiningPipeline

__all__ = [
    "SPAlgorithm",
    "SequenceLoader",
    "SequenceTransformer",
    "SPEvaluator",
    "SPPipeline",
    "Sequence",
    "SequentialPattern",
    "SequentialRule",
    "SPAlgorithmResult",
    "SPEvaluationResult",
    "SPContainer",
    "SPMiningPipeline",
]

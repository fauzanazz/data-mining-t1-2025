"""Sequential Pattern Mining Pipeline Framework."""

from sp_mining.core.interfaces import (
    SPAlgorithm,
    SequenceLoader,
    SequenceTransformer,
    SPEvaluator,
    SPPipeline,
)
from sp_mining.core.container import SPContainer
from sp_mining.core.pipeline import SPMiningPipeline

__version__ = "0.1.0"

__all__ = [
    "SPAlgorithm",
    "SequenceLoader",
    "SequenceTransformer",
    "SPEvaluator",
    "SPPipeline",
    "SPContainer",
    "SPMiningPipeline",
]

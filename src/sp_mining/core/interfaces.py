"""Core interfaces for the Sequential Pattern Mining Pipeline.

This module defines the abstract base classes and protocols for
sequential pattern mining, enabling dependency injection and
interchangeable components.

Sequential Pattern Mining differs from Frequent Pattern Mining in that
it considers the ORDER of items/events, making it suitable for:
- Customer purchase sequences over time
- Web clickstream analysis
- Event log analysis
- DNA sequence analysis
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class Itemset:
    """Represents a set of items occurring together in a sequence element.

    In sequential patterns, each element of a sequence can contain
    multiple items that occur simultaneously.
    """
    items: frozenset[str]

    def __str__(self) -> str:
        return "{" + ", ".join(sorted(self.items)) + "}"

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class Sequence:
    """Represents a sequence of itemsets.

    A sequence is an ordered list of itemsets, where each itemset
    represents items that occur together at a point in time.

    Example:
        Customer purchase sequence:
        <{bread, milk}, {eggs}, {bread, butter, jam}>
        - First visit: bought bread and milk together
        - Second visit: bought eggs
        - Third visit: bought bread, butter, and jam together
    """
    sequence_id: str | int
    elements: tuple[Itemset, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.elements)

    def __str__(self) -> str:
        return "<" + ", ".join(str(e) for e in self.elements) + ">"

    def __hash__(self) -> int:
        return hash((self.sequence_id, self.elements))

    def total_items(self) -> int:
        """Return total number of items across all elements."""
        return sum(len(e) for e in self.elements)

    def flatten(self) -> list[str]:
        """Flatten sequence to list of all items (losing order info)."""
        items = []
        for element in self.elements:
            items.extend(element.items)
        return items


@dataclass(frozen=True)
class SequentialPattern:
    """Represents a frequent sequential pattern with its support.

    A sequential pattern is a sequence that appears frequently
    across the database of sequences.
    """
    elements: tuple[Itemset, ...]
    support: float
    support_count: int = 0

    def __str__(self) -> str:
        return "<" + ", ".join(str(e) for e in self.elements) + ">"

    def __len__(self) -> int:
        return len(self.elements)

    def __hash__(self) -> int:
        return hash((self.elements, self.support))


@dataclass
class SequentialRule:
    """Represents a sequential rule with metrics.

    A sequential rule X -> Y means that if pattern X appears,
    pattern Y is likely to follow.

    Example:
        <{phone}> -> <{case}, {charger}>
        If customer buys phone, they'll likely buy case then charger.
    """
    antecedent: tuple[Itemset, ...]
    consequent: tuple[Itemset, ...]
    support: float
    confidence: float
    lift: float = 1.0

    def __str__(self) -> str:
        ant = "<" + ", ".join(str(e) for e in self.antecedent) + ">"
        con = "<" + ", ".join(str(e) for e in self.consequent) + ">"
        return f"{ant} -> {con}"


@dataclass
class SPAlgorithmResult:
    """Result container for SP algorithm execution."""
    patterns: list[SequentialPattern]
    rules: list[SequentialRule]
    execution_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SPEvaluationResult:
    """Result container for SP evaluation metrics."""
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SequenceLoader(Protocol):
    """Protocol for loading sequence datasets.

    Implementations should handle loading data from various sources
    and return a pandas DataFrame suitable for sequence transformation.
    """

    def load(self) -> pd.DataFrame:
        """Load and return the dataset as a DataFrame."""
        ...

    @property
    def name(self) -> str:
        """Return the name/identifier of this data loader."""
        ...


@runtime_checkable
class SequenceTransformer(Protocol):
    """Protocol for transforming data into sequence format.

    SP mining algorithms require data in sequence format
    (list of Sequence objects). This transformer handles the conversion.
    """

    def transform(self, df: pd.DataFrame) -> list[Sequence]:
        """Transform DataFrame to list of Sequences."""
        ...

    @property
    def name(self) -> str:
        """Return the name/identifier of this transformer."""
        ...


class SPAlgorithm(ABC):
    """Abstract base class for Sequential Pattern mining algorithms.

    All SP mining algorithms (PrefixSpan, GSP, SPADE, etc.)
    must inherit from this class and implement the required methods.
    """

    def __init__(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.5,
        max_pattern_length: Optional[int] = None,
    ) -> None:
        """Initialize algorithm with minimum thresholds.

        Args:
            min_support: Minimum support threshold (0.0 to 1.0)
            min_confidence: Minimum confidence for rules (0.0 to 1.0)
            max_pattern_length: Maximum length of patterns. None for no limit.
        """
        self._min_support = min_support
        self._min_confidence = min_confidence
        self._max_pattern_length = max_pattern_length

    @property
    def min_support(self) -> float:
        """Get minimum support threshold."""
        return self._min_support

    @property
    def min_confidence(self) -> float:
        """Get minimum confidence threshold."""
        return self._min_confidence

    @property
    def max_pattern_length(self) -> Optional[int]:
        """Get maximum pattern length."""
        return self._max_pattern_length

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the algorithm."""
        ...

    @abstractmethod
    def find_sequential_patterns(
        self, sequences: list[Sequence]
    ) -> list[SequentialPattern]:
        """Find all frequent sequential patterns in the sequences.

        Args:
            sequences: List of Sequence objects.

        Returns:
            List of SequentialPattern objects.
        """
        ...

    @abstractmethod
    def generate_rules(
        self, patterns: list[SequentialPattern]
    ) -> list[SequentialRule]:
        """Generate sequential rules from patterns.

        Args:
            patterns: List of sequential patterns.

        Returns:
            List of SequentialRule objects.
        """
        ...

    def run(self, sequences: list[Sequence]) -> SPAlgorithmResult:
        """Execute the full algorithm pipeline.

        Args:
            sequences: List of sequences.

        Returns:
            SPAlgorithmResult containing patterns, rules, and metadata.
        """
        import time

        start_time = time.perf_counter()
        patterns = self.find_sequential_patterns(sequences)
        rules = self.generate_rules(patterns)
        execution_time = time.perf_counter() - start_time

        return SPAlgorithmResult(
            patterns=patterns,
            rules=rules,
            execution_time=execution_time,
            metadata={
                "algorithm": self.name,
                "min_support": self.min_support,
                "min_confidence": self.min_confidence,
                "max_pattern_length": self.max_pattern_length,
                "num_sequences": len(sequences),
            },
        )


class SPEvaluator(ABC):
    """Abstract base class for evaluating SP mining results.

    Evaluators compute various metrics to assess the quality
    of discovered sequential patterns and rules.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this evaluator."""
        ...

    @abstractmethod
    def evaluate(
        self,
        result: SPAlgorithmResult,
        sequences: list[Sequence],
    ) -> SPEvaluationResult:
        """Evaluate the algorithm result.

        Args:
            result: The algorithm execution result.
            sequences: Original sequences for validation.

        Returns:
            SPEvaluationResult with computed metrics.
        """
        ...


class SPPipeline(ABC):
    """Abstract base class for SP mining pipelines.

    A pipeline orchestrates the execution of data loading,
    transformation, algorithm execution, and evaluation.
    """

    @abstractmethod
    def add_dataset(
        self, loader: SequenceLoader, transformer: SequenceTransformer
    ) -> "SPPipeline":
        """Add a dataset with its transformer to the pipeline."""
        ...

    @abstractmethod
    def add_algorithm(self, algorithm: SPAlgorithm) -> "SPPipeline":
        """Add an algorithm to the pipeline."""
        ...

    @abstractmethod
    def add_evaluator(self, evaluator: SPEvaluator) -> "SPPipeline":
        """Add an evaluator to the pipeline."""
        ...

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the complete pipeline."""
        ...

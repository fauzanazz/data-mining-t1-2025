"""Core interfaces for the FP Mining Pipeline.

This module defines the abstract base classes and protocols that all
implementations must follow, enabling dependency injection and
interchangeable components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@dataclass
class FrequentItemset:
    """Represents a frequent itemset with its support."""

    items: frozenset[str]
    support: float

    def __hash__(self) -> int:
        return hash((self.items, self.support))


@dataclass
class AssociationRule:
    """Represents an association rule with metrics."""

    antecedent: frozenset[str]
    consequent: frozenset[str]
    support: float
    confidence: float
    lift: float = 1.0
    conviction: float = float("inf")
    leverage: float = 0.0


@dataclass
class AlgorithmResult:
    """Result container for algorithm execution."""

    itemsets: list[FrequentItemset]
    rules: list[AssociationRule]
    execution_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result container for evaluation metrics."""

    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DataLoader(Protocol):
    """Protocol for loading datasets.

    Implementations should handle loading data from various sources
    (CSV, database, API, etc.) and return a pandas DataFrame.
    """

    def load(self) -> pd.DataFrame:
        """Load and return the dataset as a DataFrame."""
        ...

    @property
    def name(self) -> str:
        """Return the name/identifier of this data loader."""
        ...


@runtime_checkable
class DataTransformer(Protocol):
    """Protocol for transforming data into transaction format.

    FP mining algorithms require data in transaction format
    (list of itemsets). This transformer handles the conversion.
    """

    def transform(self, df: pd.DataFrame) -> list[list[str]]:
        """Transform DataFrame to list of transactions."""
        ...

    @property
    def name(self) -> str:
        """Return the name/identifier of this transformer."""
        ...


class Algorithm(ABC):
    """Abstract base class for FP mining algorithms.

    All FP mining algorithms (Apriori, FP-Growth, ECLAT, etc.)
    must inherit from this class and implement the required methods.
    """

    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.5) -> None:
        """Initialize algorithm with minimum thresholds.

        Args:
            min_support: Minimum support threshold (0.0 to 1.0)
            min_confidence: Minimum confidence for association rules (0.0 to 1.0)
        """
        self._min_support = min_support
        self._min_confidence = min_confidence

    @property
    def min_support(self) -> float:
        """Get minimum support threshold."""
        return self._min_support

    @property
    def min_confidence(self) -> float:
        """Get minimum confidence threshold."""
        return self._min_confidence

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the algorithm."""
        ...

    @abstractmethod
    def find_frequent_itemsets(
        self, transactions: list[list[str]]
    ) -> list[FrequentItemset]:
        """Find all frequent itemsets in the transactions.

        Args:
            transactions: List of transactions, where each transaction
                         is a list of items.

        Returns:
            List of FrequentItemset objects.
        """
        ...

    @abstractmethod
    def generate_rules(
        self, itemsets: list[FrequentItemset]
    ) -> list[AssociationRule]:
        """Generate association rules from frequent itemsets.

        Args:
            itemsets: List of frequent itemsets.

        Returns:
            List of AssociationRule objects.
        """
        ...

    def run(self, transactions: list[list[str]]) -> AlgorithmResult:
        """Execute the full algorithm pipeline.

        Args:
            transactions: List of transactions.

        Returns:
            AlgorithmResult containing itemsets, rules, and metadata.
        """
        import time

        start_time = time.perf_counter()
        itemsets = self.find_frequent_itemsets(transactions)
        rules = self.generate_rules(itemsets)
        execution_time = time.perf_counter() - start_time

        return AlgorithmResult(
            itemsets=itemsets,
            rules=rules,
            execution_time=execution_time,
            metadata={
                "algorithm": self.name,
                "min_support": self.min_support,
                "min_confidence": self.min_confidence,
                "num_transactions": len(transactions),
            },
        )


class Evaluator(ABC):
    """Abstract base class for evaluating algorithm results.

    Evaluators compute various metrics to assess the quality
    of discovered patterns and rules.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this evaluator."""
        ...

    @abstractmethod
    def evaluate(
        self,
        result: AlgorithmResult,
        transactions: list[list[str]],
    ) -> EvaluationResult:
        """Evaluate the algorithm result.

        Args:
            result: The algorithm execution result.
            transactions: Original transactions for validation.

        Returns:
            EvaluationResult with computed metrics.
        """
        ...


class Pipeline(ABC):
    """Abstract base class for mining pipelines.

    A pipeline orchestrates the execution of data loading,
    transformation, algorithm execution, and evaluation.
    """

    @abstractmethod
    def add_dataset(self, loader: DataLoader, transformer: DataTransformer) -> "Pipeline":
        """Add a dataset with its transformer to the pipeline.

        Args:
            loader: DataLoader implementation.
            transformer: DataTransformer implementation.

        Returns:
            Self for method chaining.
        """
        ...

    @abstractmethod
    def add_algorithm(self, algorithm: Algorithm) -> "Pipeline":
        """Add an algorithm to the pipeline.

        Args:
            algorithm: Algorithm implementation.

        Returns:
            Self for method chaining.
        """
        ...

    @abstractmethod
    def add_evaluator(self, evaluator: Evaluator) -> "Pipeline":
        """Add an evaluator to the pipeline.

        Args:
            evaluator: Evaluator implementation.

        Returns:
            Self for method chaining.
        """
        ...

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the complete pipeline.

        Returns:
            Dictionary containing all results organized by
            dataset, algorithm, and evaluator.
        """
        ...

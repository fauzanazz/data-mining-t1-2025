"""Pipeline orchestrator for SP Mining.

This module provides the concrete pipeline implementation that
coordinates the execution of multiple datasets, algorithms, and
evaluators for sequential pattern mining.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import logging

import pandas as pd
from tqdm import tqdm

from sp_mining.core.interfaces import (
    SPAlgorithm,
    SPAlgorithmResult,
    SequenceLoader,
    SequenceTransformer,
    SPEvaluator,
    SPEvaluationResult,
    SPPipeline,
    Sequence,
)


logger = logging.getLogger(__name__)


@dataclass
class SPDatasetConfig:
    """Configuration for a dataset in the SP pipeline."""

    loader: SequenceLoader
    transformer: SequenceTransformer
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.loader.name


@dataclass
class SPPipelineResult:
    """Complete result of an SP pipeline execution."""

    results: dict[str, dict[str, dict[str, Any]]]
    summary: dict[str, Any] = field(default_factory=dict)

    def get_result(
        self,
        dataset: str,
        algorithm: str,
    ) -> Optional[SPAlgorithmResult]:
        """Get algorithm result for a specific dataset and algorithm."""
        try:
            return self.results[dataset][algorithm]["result"]
        except KeyError:
            return None

    def get_evaluations(
        self,
        dataset: str,
        algorithm: str,
    ) -> dict[str, SPEvaluationResult]:
        """Get all evaluation results for a dataset/algorithm combination."""
        try:
            return self.results[dataset][algorithm].get("evaluations", {})
        except KeyError:
            return {}


class SPMiningPipeline(SPPipeline):
    """Concrete pipeline for Sequential Pattern mining.

    This pipeline allows running multiple algorithms on multiple
    datasets, evaluating results with multiple metrics.

    Example:
        >>> pipeline = SPMiningPipeline()
        >>> pipeline.add_dataset(csv_loader, temporal_transformer)
        >>> pipeline.add_algorithm(prefixspan)
        >>> pipeline.add_algorithm(gsp)
        >>> pipeline.add_evaluator(coverage_evaluator)
        >>> results = pipeline.run()
    """

    def __init__(self, verbose: bool = True) -> None:
        """Initialize the pipeline.

        Args:
            verbose: If True, show progress bars during execution.
        """
        self._datasets: list[SPDatasetConfig] = []
        self._algorithms: list[SPAlgorithm] = []
        self._evaluators: list[SPEvaluator] = []
        self._cached_sequences: dict[str, list[Sequence]] = {}
        self._verbose = verbose

    def add_dataset(
        self,
        loader: SequenceLoader,
        transformer: SequenceTransformer,
        name: Optional[str] = None,
    ) -> "SPMiningPipeline":
        """Add a dataset with its transformer to the pipeline.

        Args:
            loader: SequenceLoader implementation.
            transformer: SequenceTransformer implementation.
            name: Optional custom name for the dataset.

        Returns:
            Self for method chaining.
        """
        config = SPDatasetConfig(
            loader=loader,
            transformer=transformer,
            name=name or loader.name,
        )
        self._datasets.append(config)
        logger.info(f"Added dataset: {config.name}")
        return self

    def add_algorithm(self, algorithm: SPAlgorithm) -> "SPMiningPipeline":
        """Add an algorithm to the pipeline.

        Args:
            algorithm: SPAlgorithm implementation.

        Returns:
            Self for method chaining.
        """
        self._algorithms.append(algorithm)
        logger.info(f"Added algorithm: {algorithm.name}")
        return self

    def add_evaluator(self, evaluator: SPEvaluator) -> "SPMiningPipeline":
        """Add an evaluator to the pipeline.

        Args:
            evaluator: SPEvaluator implementation.

        Returns:
            Self for method chaining.
        """
        self._evaluators.append(evaluator)
        logger.info(f"Added evaluator: {evaluator.name}")
        return self

    def set_verbose(self, verbose: bool) -> "SPMiningPipeline":
        """Set verbose mode for progress bars.

        Args:
            verbose: If True, show progress bars.

        Returns:
            Self for method chaining.
        """
        self._verbose = verbose
        return self

    def _load_and_transform(
        self,
        config: SPDatasetConfig,
        pbar: Optional[tqdm] = None,
    ) -> list[Sequence]:
        """Load and transform a dataset, with caching."""
        if config.name in self._cached_sequences:
            logger.debug(f"Using cached sequences for {config.name}")
            return self._cached_sequences[config.name]

        if pbar:
            pbar.set_description(f"Loading {config.name}")

        logger.info(f"Loading dataset: {config.name}")
        df: pd.DataFrame = config.loader.load()
        logger.info(f"Loaded {len(df)} rows from {config.name}")

        if pbar:
            pbar.set_description(f"Transforming {config.name}")

        logger.info(f"Transforming with {config.transformer.name}")
        sequences = config.transformer.transform(df)
        logger.info(f"Created {len(sequences)} sequences")

        self._cached_sequences[config.name] = sequences
        return sequences

    def _run_algorithm(
        self,
        algorithm: SPAlgorithm,
        sequences: list[Sequence],
        pbar: Optional[tqdm] = None,
    ) -> SPAlgorithmResult:
        """Run a single algorithm on sequences."""
        if pbar:
            pbar.set_description(f"Running {algorithm.name}")

        logger.info(
            f"Running {algorithm.name} with min_support={algorithm.min_support}, "
            f"min_confidence={algorithm.min_confidence}"
        )
        result = algorithm.run(sequences)
        logger.info(
            f"{algorithm.name} found {len(result.patterns)} patterns, "
            f"{len(result.rules)} rules in {result.execution_time:.4f}s"
        )
        return result

    def _evaluate_result(
        self,
        evaluator: SPEvaluator,
        result: SPAlgorithmResult,
        sequences: list[Sequence],
        pbar: Optional[tqdm] = None,
    ) -> SPEvaluationResult:
        """Evaluate algorithm result with a single evaluator."""
        if pbar:
            pbar.set_description(f"Evaluating with {evaluator.name}")

        logger.debug(f"Evaluating with {evaluator.name}")
        return evaluator.evaluate(result, sequences)

    def run(self) -> SPPipelineResult:
        """Execute the complete pipeline.

        Returns:
            SPPipelineResult containing all results organized by
            dataset, algorithm, and evaluator.
        """
        if not self._datasets:
            raise ValueError("No datasets added to pipeline")
        if not self._algorithms:
            raise ValueError("No algorithms added to pipeline")

        results: dict[str, dict[str, dict[str, Any]]] = {}
        total_patterns = 0
        total_rules = 0
        total_time = 0.0

        # Calculate total steps for progress bar
        total_steps = len(self._datasets) * (
            1 + len(self._algorithms) * (1 + len(self._evaluators))
        )

        with tqdm(
            total=total_steps,
            desc="SP Mining Pipeline",
            disable=not self._verbose,
            unit="step",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for dataset_config in self._datasets:
                dataset_name = dataset_config.name
                results[dataset_name] = {}

                # Load and transform
                sequences = self._load_and_transform(dataset_config, pbar)
                pbar.update(1)

                for algorithm in self._algorithms:
                    algo_name = algorithm.name

                    # Run algorithm
                    algo_result = self._run_algorithm(algorithm, sequences, pbar)
                    pbar.update(1)

                    results[dataset_name][algo_name] = {
                        "result": algo_result,
                        "evaluations": {},
                    }

                    total_patterns += len(algo_result.patterns)
                    total_rules += len(algo_result.rules)
                    total_time += algo_result.execution_time

                    # Run evaluators
                    for evaluator in self._evaluators:
                        eval_result = self._evaluate_result(
                            evaluator, algo_result, sequences, pbar
                        )
                        results[dataset_name][algo_name]["evaluations"][
                            evaluator.name
                        ] = eval_result
                        pbar.update(1)

            pbar.set_description("Pipeline complete")

        summary = {
            "num_datasets": len(self._datasets),
            "num_algorithms": len(self._algorithms),
            "num_evaluators": len(self._evaluators),
            "total_patterns_found": total_patterns,
            "total_rules_generated": total_rules,
            "total_execution_time": total_time,
        }

        logger.info(f"SP Pipeline complete: {summary}")
        return SPPipelineResult(results=results, summary=summary)

    def clear(self) -> "SPMiningPipeline":
        """Clear all datasets, algorithms, and evaluators."""
        self._datasets.clear()
        self._algorithms.clear()
        self._evaluators.clear()
        self._cached_sequences.clear()
        return self

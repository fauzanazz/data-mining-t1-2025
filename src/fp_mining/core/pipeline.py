"""Pipeline orchestrator for FP Mining.

This module provides the concrete pipeline implementation that
coordinates the execution of multiple datasets, algorithms, and
evaluators in a systematic manner.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import logging

import pandas as pd
from tqdm import tqdm

from fp_mining.core.interfaces import (
    Algorithm,
    AlgorithmResult,
    DataLoader,
    DataTransformer,
    Evaluator,
    EvaluationResult,
    Pipeline,
)


logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a dataset in the pipeline."""

    loader: DataLoader
    transformer: DataTransformer
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.loader.name


@dataclass
class PipelineResult:
    """Complete result of a pipeline execution."""

    results: dict[str, dict[str, dict[str, Any]]]
    summary: dict[str, Any] = field(default_factory=dict)

    def get_result(
        self,
        dataset: str,
        algorithm: str,
    ) -> Optional[AlgorithmResult]:
        """Get algorithm result for a specific dataset and algorithm."""
        try:
            return self.results[dataset][algorithm]["result"]
        except KeyError:
            return None

    def get_evaluations(
        self,
        dataset: str,
        algorithm: str,
    ) -> dict[str, EvaluationResult]:
        """Get all evaluation results for a dataset/algorithm combination."""
        try:
            return self.results[dataset][algorithm].get("evaluations", {})
        except KeyError:
            return {}


class FPMiningPipeline(Pipeline):
    """Concrete pipeline for Frequent Pattern mining.

    This pipeline allows running multiple algorithms on multiple
    datasets, evaluating results with multiple metrics.

    Example:
        >>> pipeline = FPMiningPipeline()
        >>> pipeline.add_dataset(csv_loader, transaction_transformer)
        >>> pipeline.add_algorithm(apriori)
        >>> pipeline.add_algorithm(fpgrowth)
        >>> pipeline.add_evaluator(coverage_evaluator)
        >>> results = pipeline.run()
    """

    def __init__(self, verbose: bool = True) -> None:
        """Initialize the pipeline.

        Args:
            verbose: If True, show progress bars during execution.
        """
        self._datasets: list[DatasetConfig] = []
        self._algorithms: list[Algorithm] = []
        self._evaluators: list[Evaluator] = []
        self._cached_transactions: dict[str, list[list[str]]] = {}
        self._verbose = verbose

    def add_dataset(
        self,
        loader: DataLoader,
        transformer: DataTransformer,
        name: Optional[str] = None,
    ) -> "FPMiningPipeline":
        """Add a dataset with its transformer to the pipeline.

        Args:
            loader: DataLoader implementation.
            transformer: DataTransformer implementation.
            name: Optional custom name for the dataset.

        Returns:
            Self for method chaining.
        """
        config = DatasetConfig(
            loader=loader,
            transformer=transformer,
            name=name or loader.name,
        )
        self._datasets.append(config)
        logger.info(f"Added dataset: {config.name}")
        return self

    def add_algorithm(self, algorithm: Algorithm) -> "FPMiningPipeline":
        """Add an algorithm to the pipeline.

        Args:
            algorithm: Algorithm implementation.

        Returns:
            Self for method chaining.
        """
        self._algorithms.append(algorithm)
        logger.info(f"Added algorithm: {algorithm.name}")
        return self

    def add_evaluator(self, evaluator: Evaluator) -> "FPMiningPipeline":
        """Add an evaluator to the pipeline.

        Args:
            evaluator: Evaluator implementation.

        Returns:
            Self for method chaining.
        """
        self._evaluators.append(evaluator)
        logger.info(f"Added evaluator: {evaluator.name}")
        return self

    def set_verbose(self, verbose: bool) -> "FPMiningPipeline":
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
        config: DatasetConfig,
        pbar: Optional[tqdm] = None,
    ) -> list[list[str]]:
        """Load and transform a dataset, with caching."""
        if config.name in self._cached_transactions:
            logger.debug(f"Using cached transactions for {config.name}")
            return self._cached_transactions[config.name]

        if pbar:
            pbar.set_description(f"Loading {config.name}")

        logger.info(f"Loading dataset: {config.name}")
        df: pd.DataFrame = config.loader.load()
        logger.info(f"Loaded {len(df)} rows from {config.name}")

        if pbar:
            pbar.set_description(f"Transforming {config.name}")

        logger.info(f"Transforming with {config.transformer.name}")
        transactions = config.transformer.transform(df)
        logger.info(f"Created {len(transactions)} transactions")

        self._cached_transactions[config.name] = transactions
        return transactions

    def _run_algorithm(
        self,
        algorithm: Algorithm,
        transactions: list[list[str]],
        pbar: Optional[tqdm] = None,
    ) -> AlgorithmResult:
        """Run a single algorithm on transactions."""
        if pbar:
            pbar.set_description(f"Running {algorithm.name}")

        logger.info(
            f"Running {algorithm.name} with min_support={algorithm.min_support}, "
            f"min_confidence={algorithm.min_confidence}"
        )
        result = algorithm.run(transactions)
        logger.info(
            f"{algorithm.name} found {len(result.itemsets)} itemsets, "
            f"{len(result.rules)} rules in {result.execution_time:.4f}s"
        )
        return result

    def _evaluate_result(
        self,
        evaluator: Evaluator,
        result: AlgorithmResult,
        transactions: list[list[str]],
        pbar: Optional[tqdm] = None,
    ) -> EvaluationResult:
        """Evaluate algorithm result with a single evaluator."""
        if pbar:
            pbar.set_description(f"Evaluating with {evaluator.name}")

        logger.debug(f"Evaluating with {evaluator.name}")
        return evaluator.evaluate(result, transactions)

    def run(self) -> PipelineResult:
        """Execute the complete pipeline.

        Returns:
            PipelineResult containing all results organized by
            dataset, algorithm, and evaluator.
        """
        if not self._datasets:
            raise ValueError("No datasets added to pipeline")
        if not self._algorithms:
            raise ValueError("No algorithms added to pipeline")

        results: dict[str, dict[str, dict[str, Any]]] = {}
        total_itemsets = 0
        total_rules = 0
        total_time = 0.0

        # Calculate total steps for progress bar
        total_steps = len(self._datasets) * (
            1 + len(self._algorithms) * (1 + len(self._evaluators))
        )

        with tqdm(
            total=total_steps,
            desc="FP Mining Pipeline",
            disable=not self._verbose,
            unit="step",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for dataset_config in self._datasets:
                dataset_name = dataset_config.name
                results[dataset_name] = {}

                # Load and transform
                transactions = self._load_and_transform(dataset_config, pbar)
                pbar.update(1)

                for algorithm in self._algorithms:
                    algo_name = algorithm.name

                    # Run algorithm
                    algo_result = self._run_algorithm(algorithm, transactions, pbar)
                    pbar.update(1)

                    results[dataset_name][algo_name] = {
                        "result": algo_result,
                        "evaluations": {},
                    }

                    total_itemsets += len(algo_result.itemsets)
                    total_rules += len(algo_result.rules)
                    total_time += algo_result.execution_time

                    # Run evaluators
                    for evaluator in self._evaluators:
                        eval_result = self._evaluate_result(
                            evaluator, algo_result, transactions, pbar
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
            "total_itemsets_found": total_itemsets,
            "total_rules_generated": total_rules,
            "total_execution_time": total_time,
        }

        logger.info(f"Pipeline complete: {summary}")
        return PipelineResult(results=results, summary=summary)

    def clear(self) -> "FPMiningPipeline":
        """Clear all datasets, algorithms, and evaluators."""
        self._datasets.clear()
        self._algorithms.clear()
        self._evaluators.clear()
        self._cached_transactions.clear()
        return self

"""Main entry point for the SP Mining Pipeline.

This module provides the CLI interface and demonstrates how to
configure and run the SP pipeline using dependency injection.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from sp_mining.core.container import SPContainer
from sp_mining.core.pipeline import SPMiningPipeline
from sp_mining.algorithms import PrefixSpanAlgorithm, GSPAlgorithm
from sp_mining.loaders import SequenceCSVLoader, TemporalTransactionTransformer
from sp_mining.evaluators import (
    SPCoverageEvaluator,
    SPRuleQualityEvaluator,
    SPPerformanceEvaluator,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_default_container(
    data_path: str,
    min_support: float = 0.01,
    min_confidence: float = 0.5,
    max_pattern_length: Optional[int] = None,
) -> SPContainer:
    """Create a container with default registrations.

    Args:
        data_path: Path to the dataset.
        min_support: Minimum support threshold.
        min_confidence: Minimum confidence threshold.
        max_pattern_length: Maximum pattern length.

    Returns:
        Configured SPContainer instance.
    """
    container = SPContainer()

    # Register data loader
    container.register_loader(
        "retail_csv",
        lambda c: SequenceCSVLoader(
            data_path,
            name="RetailSequences",
            parse_dates=["TransactionDate"],
        ),
        singleton=True,
    )

    # Register transformer
    container.register_transformer(
        "temporal_transformer",
        lambda c: TemporalTransactionTransformer(
            sequence_col="CustomerID",
            item_col="ProductCategory",
            time_col="TransactionDate",
        ),
        singleton=True,
    )

    # Register algorithms
    container.register_algorithm(
        "prefixspan",
        lambda c: PrefixSpanAlgorithm(
            min_support=min_support,
            min_confidence=min_confidence,
            max_pattern_length=max_pattern_length,
        ),
    )

    container.register_algorithm(
        "gsp",
        lambda c: GSPAlgorithm(
            min_support=min_support,
            min_confidence=min_confidence,
            max_pattern_length=max_pattern_length,
        ),
    )

    # Register evaluators
    container.register_evaluator(
        "coverage",
        lambda c: SPCoverageEvaluator(),
        singleton=True,
    )

    container.register_evaluator(
        "quality",
        lambda c: SPRuleQualityEvaluator(),
        singleton=True,
    )

    container.register_evaluator(
        "performance",
        lambda c: SPPerformanceEvaluator(),
        singleton=True,
    )

    return container


def run_pipeline(
    container: SPContainer,
    algorithms: Optional[list[str]] = None,
) -> None:
    """Run the SP mining pipeline with registered components.

    Args:
        container: Configured dependency injection container.
        algorithms: List of algorithm names to run. None runs all.
    """
    # Build pipeline
    pipeline = SPMiningPipeline()

    # Add datasets
    loader = container.resolve_loader("retail_csv")
    transformer = container.resolve_transformer("temporal_transformer")
    pipeline.add_dataset(loader, transformer)

    # Add algorithms
    if algorithms:
        for algo_name in algorithms:
            pipeline.add_algorithm(container.resolve_algorithm(algo_name))
    else:
        for algo in container.get_all_algorithms().values():
            pipeline.add_algorithm(algo)

    # Add evaluators
    for evaluator in container.get_all_evaluators().values():
        pipeline.add_evaluator(evaluator)

    # Run pipeline
    result = pipeline.run()

    # Print results
    print_results(result)


def print_results(result) -> None:
    """Print pipeline results in a formatted manner."""
    print("\n" + "=" * 60)
    print("SEQUENTIAL PATTERN MINING PIPELINE RESULTS")
    print("=" * 60)

    print(f"\nSummary:")
    print(f"  Datasets: {result.summary['num_datasets']}")
    print(f"  Algorithms: {result.summary['num_algorithms']}")
    print(f"  Evaluators: {result.summary['num_evaluators']}")
    print(f"  Total Patterns: {result.summary['total_patterns_found']}")
    print(f"  Total Rules: {result.summary['total_rules_generated']}")
    print(f"  Total Time: {result.summary['total_execution_time']:.4f}s")

    for dataset_name, dataset_results in result.results.items():
        print(f"\n{'─' * 60}")
        print(f"Dataset: {dataset_name}")
        print("─" * 60)

        for algo_name, algo_results in dataset_results.items():
            algo_result = algo_results["result"]
            print(f"\n  Algorithm: {algo_name}")
            print(f"    Patterns: {len(algo_result.patterns)}")
            print(f"    Rules: {len(algo_result.rules)}")
            print(f"    Time: {algo_result.execution_time:.4f}s")

            # Print top 5 patterns by support
            if algo_result.patterns:
                print("\n    Top 5 Patterns (by support):")
                sorted_patterns = sorted(
                    algo_result.patterns,
                    key=lambda p: p.support,
                    reverse=True,
                )[:5]
                for i, pattern in enumerate(sorted_patterns, 1):
                    print(f"      {i}. {pattern} (support={pattern.support:.4f})")

            # Print top 5 rules by confidence
            if algo_result.rules:
                print("\n    Top 5 Rules (by confidence):")
                sorted_rules = sorted(
                    algo_result.rules,
                    key=lambda r: r.confidence,
                    reverse=True,
                )[:5]
                for i, rule in enumerate(sorted_rules, 1):
                    print(
                        f"      {i}. {rule} "
                        f"(conf={rule.confidence:.3f}, lift={rule.lift:.3f})"
                    )

            # Print evaluations
            evaluations = algo_results.get("evaluations", {})
            if evaluations:
                print("\n    Evaluations:")
                for eval_name, eval_result in evaluations.items():
                    print(f"      {eval_name}:")
                    for metric, value in list(eval_result.metrics.items())[:4]:
                        if isinstance(value, float):
                            print(f"        {metric}: {value:.4f}")
                        else:
                            print(f"        {metric}: {value}")

    print("\n" + "=" * 60)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sequential Pattern Mining Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d", "--data",
        type=str,
        default="datasets/Retail_Transaction_Dataset.csv",
        help="Path to the dataset CSV file",
    )

    parser.add_argument(
        "-s", "--min-support",
        type=float,
        default=0.01,
        help="Minimum support threshold (0.0 to 1.0)",
    )

    parser.add_argument(
        "-c", "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (0.0 to 1.0)",
    )

    parser.add_argument(
        "-m", "--max-length",
        type=int,
        default=None,
        help="Maximum pattern length (default: no limit)",
    )

    parser.add_argument(
        "-a", "--algorithms",
        type=str,
        nargs="+",
        choices=["prefixspan", "gsp"],
        help="Algorithms to run (default: all)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Check if data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return 1

    try:
        container = create_default_container(
            str(data_path),
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            max_pattern_length=args.max_length,
        )

        run_pipeline(container, args.algorithms)
        return 0

    except Exception as e:
        logging.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Main entry point for the FP Mining Pipeline.

This module provides the CLI interface and demonstrates how to
configure and run the pipeline using dependency injection.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from fp_mining.core.container import Container
from fp_mining.core.pipeline import FPMiningPipeline
from fp_mining.algorithms import AprioriAlgorithm, FPGrowthAlgorithm
from fp_mining.loaders import CSVLoader, RetailTransactionTransformer
from fp_mining.evaluators import (
    CoverageEvaluator,
    RuleQualityEvaluator,
    PerformanceEvaluator,
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
) -> Container:
    """Create a container with default registrations.

    Args:
        data_path: Path to the dataset.
        min_support: Minimum support threshold.
        min_confidence: Minimum confidence threshold.

    Returns:
        Configured Container instance.
    """
    container = Container()

    # Register data loader
    container.register_loader(
        "retail_csv",
        lambda c: CSVLoader(data_path, name="RetailDataset"),
        singleton=True,
    )

    # Register transformer
    container.register_transformer(
        "retail_transformer",
        lambda c: RetailTransactionTransformer(
            group_col="CustomerID",
            item_col="ProductCategory",
        ),
        singleton=True,
    )

    # Register algorithms
    container.register_algorithm(
        "apriori",
        lambda c: AprioriAlgorithm(
            min_support=min_support,
            min_confidence=min_confidence,
        ),
    )

    container.register_algorithm(
        "fpgrowth",
        lambda c: FPGrowthAlgorithm(
            min_support=min_support,
            min_confidence=min_confidence,
        ),
    )

    # Register evaluators
    container.register_evaluator(
        "coverage",
        lambda c: CoverageEvaluator(),
        singleton=True,
    )

    container.register_evaluator(
        "quality",
        lambda c: RuleQualityEvaluator(),
        singleton=True,
    )

    container.register_evaluator(
        "performance",
        lambda c: PerformanceEvaluator(),
        singleton=True,
    )

    return container


def run_pipeline(container: Container, algorithms: Optional[list[str]] = None) -> None:
    """Run the FP mining pipeline with registered components.

    Args:
        container: Configured dependency injection container.
        algorithms: List of algorithm names to run. None runs all.
    """
    # Build pipeline
    pipeline = FPMiningPipeline()

    # Add datasets
    loader = container.resolve_loader("retail_csv")
    transformer = container.resolve_transformer("retail_transformer")
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
    print("FP MINING PIPELINE RESULTS")
    print("=" * 60)

    print(f"\nSummary:")
    print(f"  Datasets: {result.summary['num_datasets']}")
    print(f"  Algorithms: {result.summary['num_algorithms']}")
    print(f"  Evaluators: {result.summary['num_evaluators']}")
    print(f"  Total Itemsets: {result.summary['total_itemsets_found']}")
    print(f"  Total Rules: {result.summary['total_rules_generated']}")
    print(f"  Total Time: {result.summary['total_execution_time']:.4f}s")

    for dataset_name, dataset_results in result.results.items():
        print(f"\n{'─' * 60}")
        print(f"Dataset: {dataset_name}")
        print("─" * 60)

        for algo_name, algo_results in dataset_results.items():
            algo_result = algo_results["result"]
            print(f"\n  Algorithm: {algo_name}")
            print(f"    Itemsets: {len(algo_result.itemsets)}")
            print(f"    Rules: {len(algo_result.rules)}")
            print(f"    Time: {algo_result.execution_time:.4f}s")

            # Print top 5 rules by confidence
            if algo_result.rules:
                print("\n    Top 5 Rules (by confidence):")
                sorted_rules = sorted(
                    algo_result.rules,
                    key=lambda r: r.confidence,
                    reverse=True,
                )[:5]
                for i, rule in enumerate(sorted_rules, 1):
                    antecedent = ", ".join(sorted(rule.antecedent))
                    consequent = ", ".join(sorted(rule.consequent))
                    print(
                        f"      {i}. {{{antecedent}}} → {{{consequent}}} "
                        f"(conf={rule.confidence:.3f}, lift={rule.lift:.3f})"
                    )

            # Print evaluations
            evaluations = algo_results.get("evaluations", {})
            if evaluations:
                print("\n    Evaluations:")
                for eval_name, eval_result in evaluations.items():
                    print(f"      {eval_name}:")
                    for metric, value in eval_result.metrics.items():
                        if isinstance(value, float):
                            print(f"        {metric}: {value:.4f}")
                        else:
                            print(f"        {metric}: {value}")

    print("\n" + "=" * 60)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Frequent Pattern Mining Pipeline",
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
        "-a", "--algorithms",
        type=str,
        nargs="+",
        choices=["apriori", "fpgrowth"],
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
        )

        run_pipeline(container, args.algorithms)
        return 0

    except Exception as e:
        logging.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Example: Running Sequential Pattern mining on datasets.

This example demonstrates:
1. Using dependency injection container for SP mining
2. Adding datasets with temporal transformers
3. Running multiple SP algorithms (PrefixSpan, GSP)
4. Using multiple evaluators
5. Accessing results programmatically
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sp_mining.core.container import SPContainer
from sp_mining.core.pipeline import SPMiningPipeline
from sp_mining.algorithms import PrefixSpanAlgorithm, GSPAlgorithm
from sp_mining.loaders import SequenceCSVLoader, TemporalTransactionTransformer
from sp_mining.evaluators import (
    SPCoverageEvaluator,
    SPRuleQualityEvaluator,
    SPPerformanceEvaluator,
)


def main():
    """Run example SP mining pipeline."""
    # =========================================================
    # 1. Create Dependency Injection Container
    # =========================================================
    container = SPContainer()

    # =========================================================
    # 2. Register Components in Container
    # =========================================================

    # Register data loaders
    container.register_loader(
        "retail",
        lambda c: SequenceCSVLoader(
            "datasets/Retail_Transaction_Dataset.csv",
            name="Retail Sequences",
            parse_dates=["TransactionDate"],
        ),
    )

    # Register transformers
    container.register_transformer(
        "temporal_by_category",
        lambda c: TemporalTransactionTransformer(
            sequence_col="CustomerID",
            item_col="ProductCategory",
            time_col="TransactionDate",
            name="CategorySequenceTransformer",
        ),
    )

    # Register algorithms with different configurations
    container.register_algorithm(
        "prefixspan_default",
        lambda c: PrefixSpanAlgorithm(
            min_support=0.01,
            min_confidence=0.5,
            max_pattern_length=5,
        ),
    )

    container.register_algorithm(
        "gsp_default",
        lambda c: GSPAlgorithm(
            min_support=0.01,
            min_confidence=0.5,
            max_pattern_length=5,
        ),
    )

    # Register evaluators
    container.register_evaluator("coverage", lambda c: SPCoverageEvaluator())
    container.register_evaluator("quality", lambda c: SPRuleQualityEvaluator())
    container.register_evaluator("performance", lambda c: SPPerformanceEvaluator())

    # =========================================================
    # 3. Build and Configure Pipeline
    # =========================================================
    pipeline = SPMiningPipeline()

    # Add dataset with temporal transformer
    pipeline.add_dataset(
        loader=container.resolve_loader("retail"),
        transformer=container.resolve_transformer("temporal_by_category"),
        name="Retail-CategorySequences",
    )

    # Add algorithms
    pipeline.add_algorithm(container.resolve_algorithm("prefixspan_default"))
    pipeline.add_algorithm(container.resolve_algorithm("gsp_default"))

    # Add evaluators
    pipeline.add_evaluator(container.resolve_evaluator("coverage"))
    pipeline.add_evaluator(container.resolve_evaluator("quality"))
    pipeline.add_evaluator(container.resolve_evaluator("performance"))

    # =========================================================
    # 4. Run Pipeline
    # =========================================================
    print("Running Sequential Pattern Mining Pipeline...")
    print("This may take a while for large datasets...\n")

    result = pipeline.run()

    # =========================================================
    # 5. Access Results Programmatically
    # =========================================================
    print("\n" + "=" * 60)
    print("SEQUENTIAL PATTERN MINING RESULTS")
    print("=" * 60)

    # Summary
    print(f"\nPipeline Summary:")
    for key, value in result.summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Access specific results
    for dataset_name in result.results:
        print(f"\n--- {dataset_name} ---")

        for algo_name in result.results[dataset_name]:
            algo_result = result.get_result(dataset_name, algo_name)
            evaluations = result.get_evaluations(dataset_name, algo_name)

            print(f"\n  {algo_name}:")
            print(f"    Patterns found: {len(algo_result.patterns)}")
            print(f"    Rules generated: {len(algo_result.rules)}")
            print(f"    Execution time: {algo_result.execution_time:.4f}s")

            # Show top 3 patterns by support
            if algo_result.patterns:
                print("\n    Top 3 Sequential Patterns:")
                sorted_patterns = sorted(
                    algo_result.patterns,
                    key=lambda x: x.support,
                    reverse=True,
                )[:3]
                for pattern in sorted_patterns:
                    print(f"      {pattern} (support={pattern.support:.4f})")

            # Show top 3 rules by confidence
            if algo_result.rules:
                print("\n    Top 3 Sequential Rules:")
                sorted_rules = sorted(
                    algo_result.rules,
                    key=lambda x: x.confidence,
                    reverse=True,
                )[:3]
                for rule in sorted_rules:
                    print(
                        f"      {rule}\n"
                        f"        confidence={rule.confidence:.3f}, lift={rule.lift:.3f}"
                    )

            # Show evaluation metrics
            if evaluations:
                print("\n    Evaluation Metrics:")
                for eval_name, eval_result in evaluations.items():
                    print(f"      {eval_name}:")
                    for metric, value in list(eval_result.metrics.items())[:3]:
                        if isinstance(value, float):
                            print(f"        {metric}: {value:.4f}")
                        else:
                            print(f"        {metric}: {value}")

    # =========================================================
    # 6. Compare Algorithms
    # =========================================================
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON")
    print("=" * 60)

    for dataset_name in result.results:
        print(f"\nDataset: {dataset_name}")
        print("-" * 50)

        comparison_data = []
        for algo_name in result.results[dataset_name]:
            algo_result = result.get_result(dataset_name, algo_name)
            perf_eval = result.get_evaluations(dataset_name, algo_name).get(
                "SPPerformance"
            )

            comparison_data.append({
                "algorithm": algo_name,
                "patterns": len(algo_result.patterns),
                "rules": len(algo_result.rules),
                "time": algo_result.execution_time,
                "throughput": (
                    perf_eval.metrics["sequences_per_second"] if perf_eval else 0
                ),
            })

        # Print comparison table
        print(
            f"{'Algorithm':<20} {'Patterns':>10} {'Rules':>10} "
            f"{'Time (s)':>12} {'Seq/s':>12}"
        )
        print("-" * 64)
        for row in comparison_data:
            print(
                f"{row['algorithm']:<20} "
                f"{row['patterns']:>10} "
                f"{row['rules']:>10} "
                f"{row['time']:>12.4f} "
                f"{row['throughput']:>12.2f}"
            )


if __name__ == "__main__":
    main()

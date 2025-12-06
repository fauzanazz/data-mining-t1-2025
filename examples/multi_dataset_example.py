"""Example: Running FP mining on multiple datasets with multiple algorithms.

This example demonstrates:
1. Using dependency injection container
2. Adding multiple datasets
3. Running multiple algorithms
4. Using multiple evaluators
5. Accessing results programmatically
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fp_mining.core.container import Container
from fp_mining.core.pipeline import FPMiningPipeline
from fp_mining.algorithms import AprioriAlgorithm, FPGrowthAlgorithm
from fp_mining.loaders import CSVLoader, RetailTransactionTransformer, GenericTransformer
from fp_mining.evaluators import (
    CoverageEvaluator,
    RuleQualityEvaluator,
    PerformanceEvaluator,
)


def main():
    """Run example pipeline."""
    # =========================================================
    # 1. Create Dependency Injection Container
    # =========================================================
    container = Container()

    # =========================================================
    # 2. Register Components in Container
    # =========================================================

    # Register data loaders (you can add more datasets here)
    container.register_loader(
        "retail",
        lambda c: CSVLoader(
            "datasets/Retail_Transaction_Dataset.csv",
            name="Retail Transactions"
        ),
    )

    # Register transformers
    container.register_transformer(
        "retail_by_category",
        lambda c: RetailTransactionTransformer(
            group_col="CustomerID",
            item_col="ProductCategory",
            name="CategoryTransformer"
        ),
    )

    container.register_transformer(
        "retail_by_product",
        lambda c: RetailTransactionTransformer(
            group_col="CustomerID",
            item_col="ProductID",
            name="ProductTransformer"
        ),
    )

    # Register algorithms with different configurations
    container.register_algorithm(
        "apriori_strict",
        lambda c: AprioriAlgorithm(min_support=0.05, min_confidence=0.7),
    )

    container.register_algorithm(
        "apriori_relaxed",
        lambda c: AprioriAlgorithm(min_support=0.01, min_confidence=0.5),
    )

    container.register_algorithm(
        "fpgrowth_strict",
        lambda c: FPGrowthAlgorithm(min_support=0.05, min_confidence=0.7),
    )

    container.register_algorithm(
        "fpgrowth_relaxed",
        lambda c: FPGrowthAlgorithm(min_support=0.01, min_confidence=0.5),
    )

    # Register evaluators
    container.register_evaluator("coverage", lambda c: CoverageEvaluator())
    container.register_evaluator("quality", lambda c: RuleQualityEvaluator())
    container.register_evaluator("performance", lambda c: PerformanceEvaluator())

    # =========================================================
    # 3. Build and Configure Pipeline
    # =========================================================
    pipeline = FPMiningPipeline()

    # Add dataset with category-based transactions
    pipeline.add_dataset(
        loader=container.resolve_loader("retail"),
        transformer=container.resolve_transformer("retail_by_category"),
        name="Retail-Categories"
    )

    # Add same dataset with product-based transactions (different view)
    # Uncomment below to add product-level analysis (may be slower)
    # pipeline.add_dataset(
    #     loader=container.resolve_loader("retail"),
    #     transformer=container.resolve_transformer("retail_by_product"),
    #     name="Retail-Products"
    # )

    # Add algorithms (choose which to run)
    pipeline.add_algorithm(container.resolve_algorithm("apriori_relaxed"))
    pipeline.add_algorithm(container.resolve_algorithm("fpgrowth_relaxed"))

    # Add evaluators
    pipeline.add_evaluator(container.resolve_evaluator("coverage"))
    pipeline.add_evaluator(container.resolve_evaluator("quality"))
    pipeline.add_evaluator(container.resolve_evaluator("performance"))

    # =========================================================
    # 4. Run Pipeline
    # =========================================================
    print("Running FP Mining Pipeline...")
    result = pipeline.run()

    # =========================================================
    # 5. Access Results Programmatically
    # =========================================================
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)

    # Summary
    print(f"\nPipeline Summary:")
    for key, value in result.summary.items():
        print(f"  {key}: {value}")

    # Access specific results
    for dataset_name in result.results:
        print(f"\n--- {dataset_name} ---")

        for algo_name in result.results[dataset_name]:
            algo_result = result.get_result(dataset_name, algo_name)
            evaluations = result.get_evaluations(dataset_name, algo_name)

            print(f"\n  {algo_name}:")
            print(f"    Itemsets found: {len(algo_result.itemsets)}")
            print(f"    Rules generated: {len(algo_result.rules)}")
            print(f"    Execution time: {algo_result.execution_time:.4f}s")

            # Show top 3 itemsets by support
            if algo_result.itemsets:
                print("\n    Top 3 Itemsets:")
                sorted_itemsets = sorted(
                    algo_result.itemsets,
                    key=lambda x: x.support,
                    reverse=True
                )[:3]
                for itemset in sorted_itemsets:
                    items_str = ", ".join(sorted(itemset.items))
                    print(f"      {{{items_str}}} (support={itemset.support:.4f})")

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
        print("-" * 40)

        comparison_data = []
        for algo_name in result.results[dataset_name]:
            algo_result = result.get_result(dataset_name, algo_name)
            perf_eval = result.get_evaluations(dataset_name, algo_name).get("Performance")

            comparison_data.append({
                "algorithm": algo_name,
                "itemsets": len(algo_result.itemsets),
                "rules": len(algo_result.rules),
                "time": algo_result.execution_time,
                "throughput": perf_eval.metrics["transactions_per_second"] if perf_eval else 0,
            })

        # Print comparison table
        print(f"{'Algorithm':<20} {'Itemsets':>10} {'Rules':>10} {'Time (s)':>12} {'Trans/s':>12}")
        print("-" * 64)
        for row in comparison_data:
            print(
                f"{row['algorithm']:<20} "
                f"{row['itemsets']:>10} "
                f"{row['rules']:>10} "
                f"{row['time']:>12.4f} "
                f"{row['throughput']:>12.2f}"
            )


if __name__ == "__main__":
    main()

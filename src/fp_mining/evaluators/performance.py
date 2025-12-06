"""Performance evaluator for FP mining results."""

import statistics

from fp_mining.core.interfaces import (
    AlgorithmResult,
    Evaluator,
    EvaluationResult,
)


class PerformanceEvaluator(Evaluator):
    """Evaluates the performance characteristics of the mining result.

    Performance metrics help understand the efficiency and scalability
    of the algorithm execution.

    Metrics computed:
    - execution_time: Total execution time in seconds
    - itemsets_per_second: Rate of itemset discovery
    - rules_per_second: Rate of rule generation
    - avg_itemset_size: Average size of discovered itemsets
    - max_itemset_size: Maximum itemset size
    - memory_efficiency: Ratio of unique items to total itemset elements
    """

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "Performance"

    def evaluate(
        self,
        result: AlgorithmResult,
        transactions: list[list[str]],
    ) -> EvaluationResult:
        """Evaluate performance metrics.

        Args:
            result: Algorithm execution result.
            transactions: Original transactions.

        Returns:
            EvaluationResult with performance metrics.
        """
        exec_time = result.execution_time
        num_itemsets = len(result.itemsets)
        num_rules = len(result.rules)

        # Calculate rates
        itemsets_per_second = num_itemsets / exec_time if exec_time > 0 else 0.0
        rules_per_second = num_rules / exec_time if exec_time > 0 else 0.0

        # Calculate itemset size statistics
        itemset_sizes = [len(itemset.items) for itemset in result.itemsets]
        avg_itemset_size = statistics.mean(itemset_sizes) if itemset_sizes else 0.0
        max_itemset_size = max(itemset_sizes) if itemset_sizes else 0

        # Size distribution
        size_distribution: dict[int, int] = {}
        for size in itemset_sizes:
            size_distribution[size] = size_distribution.get(size, 0) + 1

        # Memory efficiency: unique items vs total elements in itemsets
        all_items_in_itemsets: set[str] = set()
        total_elements = 0
        for itemset in result.itemsets:
            all_items_in_itemsets.update(itemset.items)
            total_elements += len(itemset.items)

        memory_efficiency = (
            len(all_items_in_itemsets) / total_elements
            if total_elements > 0
            else 0.0
        )

        # Throughput relative to data size
        transactions_per_second = len(transactions) / exec_time if exec_time > 0 else 0.0

        return EvaluationResult(
            metrics={
                "execution_time": exec_time,
                "itemsets_per_second": itemsets_per_second,
                "rules_per_second": rules_per_second,
                "avg_itemset_size": avg_itemset_size,
                "max_itemset_size": float(max_itemset_size),
                "memory_efficiency": memory_efficiency,
                "transactions_per_second": transactions_per_second,
            },
            details={
                "total_itemsets": num_itemsets,
                "total_rules": num_rules,
                "total_transactions": len(transactions),
                "size_distribution": size_distribution,
                "unique_items_in_itemsets": len(all_items_in_itemsets),
                "total_itemset_elements": total_elements,
            },
        )

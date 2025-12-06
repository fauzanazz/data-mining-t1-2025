"""Coverage evaluator for FP mining results."""

from fp_mining.core.interfaces import (
    AlgorithmResult,
    Evaluator,
    EvaluationResult,
)


class CoverageEvaluator(Evaluator):
    """Evaluates the coverage of discovered patterns.

    Coverage metrics measure how well the discovered itemsets and rules
    cover the original transaction data.

    Metrics computed:
    - item_coverage: Fraction of unique items covered by itemsets
    - transaction_coverage: Fraction of transactions containing at least one itemset
    - rule_coverage: Fraction of transactions covered by at least one rule
    """

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "Coverage"

    def evaluate(
        self,
        result: AlgorithmResult,
        transactions: list[list[str]],
    ) -> EvaluationResult:
        """Evaluate coverage metrics.

        Args:
            result: Algorithm execution result.
            transactions: Original transactions.

        Returns:
            EvaluationResult with coverage metrics.
        """
        # Get all unique items from transactions
        all_items: set[str] = set()
        for transaction in transactions:
            all_items.update(transaction)

        # Get items covered by itemsets
        covered_items: set[str] = set()
        for itemset in result.itemsets:
            covered_items.update(itemset.items)

        # Calculate item coverage
        item_coverage = len(covered_items) / len(all_items) if all_items else 0.0

        # Calculate transaction coverage (transactions with at least one itemset)
        transactions_with_itemsets = 0
        itemsets_as_sets = [itemset.items for itemset in result.itemsets]

        for transaction in transactions:
            transaction_set = set(transaction)
            for itemset in itemsets_as_sets:
                if itemset.issubset(transaction_set):
                    transactions_with_itemsets += 1
                    break

        transaction_coverage = transactions_with_itemsets / len(transactions) if transactions else 0.0

        # Calculate rule coverage
        transactions_with_rules = 0
        for transaction in transactions:
            transaction_set = set(transaction)
            for rule in result.rules:
                if rule.antecedent.issubset(transaction_set):
                    transactions_with_rules += 1
                    break

        rule_coverage = transactions_with_rules / len(transactions) if transactions else 0.0

        return EvaluationResult(
            metrics={
                "item_coverage": item_coverage,
                "transaction_coverage": transaction_coverage,
                "rule_coverage": rule_coverage,
            },
            details={
                "total_items": len(all_items),
                "covered_items": len(covered_items),
                "total_transactions": len(transactions),
                "transactions_with_itemsets": transactions_with_itemsets,
                "transactions_with_rules": transactions_with_rules,
            },
        )

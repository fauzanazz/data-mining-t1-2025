"""FP-Growth algorithm implementation."""

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

from fp_mining.core.interfaces import (
    Algorithm,
    AssociationRule,
    FrequentItemset,
)


class FPGrowthAlgorithm(Algorithm):
    """FP-Growth algorithm for frequent itemset mining.

    FP-Growth is generally more efficient than Apriori as it uses
    a compressed data structure (FP-tree) and doesn't require
    candidate generation.

    Example:
        >>> algo = FPGrowthAlgorithm(min_support=0.05)
        >>> result = algo.run(transactions)
        >>> print(f"Found {len(result.itemsets)} itemsets")
    """

    def __init__(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.5,
        max_len: int | None = None,
        use_colnames: bool = True,
    ) -> None:
        """Initialize FP-Growth algorithm.

        Args:
            min_support: Minimum support threshold (0.0 to 1.0).
            min_confidence: Minimum confidence for rules (0.0 to 1.0).
            max_len: Maximum length of itemsets. None for no limit.
            use_colnames: Use column names instead of indices.
        """
        super().__init__(min_support, min_confidence)
        self._max_len = max_len
        self._use_colnames = use_colnames
        self._te = TransactionEncoder()
        self._itemsets_df: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        """Return the algorithm name."""
        return "FP-Growth"

    def _encode_transactions(
        self, transactions: list[list[str]]
    ) -> pd.DataFrame:
        """Encode transactions to one-hot format."""
        te_array = self._te.fit_transform(transactions)
        return pd.DataFrame(te_array, columns=self._te.columns_)

    def find_frequent_itemsets(
        self, transactions: list[list[str]]
    ) -> list[FrequentItemset]:
        """Find frequent itemsets using FP-Growth algorithm.

        Args:
            transactions: List of transactions.

        Returns:
            List of FrequentItemset objects.
        """
        # Encode transactions
        df_encoded = self._encode_transactions(transactions)

        # Run FP-Growth
        self._itemsets_df = fpgrowth(
            df_encoded,
            min_support=self.min_support,
            use_colnames=self._use_colnames,
            max_len=self._max_len,
        )

        # Convert to FrequentItemset objects
        itemsets: list[FrequentItemset] = []
        for _, row in self._itemsets_df.iterrows():
            itemsets.append(
                FrequentItemset(
                    items=frozenset(row["itemsets"]),
                    support=float(row["support"]),
                )
            )

        return itemsets

    def generate_rules(
        self, itemsets: list[FrequentItemset]
    ) -> list[AssociationRule]:
        """Generate association rules from frequent itemsets.

        Args:
            itemsets: List of frequent itemsets (unused, uses cached DataFrame).

        Returns:
            List of AssociationRule objects.
        """
        if self._itemsets_df is None or len(self._itemsets_df) == 0:
            return []

        # Need at least 2-itemsets to generate rules
        if self._itemsets_df["itemsets"].apply(len).max() < 2:
            return []

        try:
            rules_df = association_rules(
                self._itemsets_df,
                metric="confidence",
                min_threshold=self.min_confidence,
            )
        except ValueError:
            # No rules could be generated
            return []

        # Convert to AssociationRule objects
        rules: list[AssociationRule] = []
        for _, row in rules_df.iterrows():
            rules.append(
                AssociationRule(
                    antecedent=frozenset(row["antecedents"]),
                    consequent=frozenset(row["consequents"]),
                    support=float(row["support"]),
                    confidence=float(row["confidence"]),
                    lift=float(row["lift"]),
                    conviction=float(row["conviction"]) if pd.notna(row["conviction"]) else float("inf"),
                    leverage=float(row["leverage"]),
                )
            )

        return rules

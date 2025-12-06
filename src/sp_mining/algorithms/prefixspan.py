"""PrefixSpan algorithm implementation for sequential pattern mining.

PrefixSpan (Prefix-projected Sequential pattern mining) is an efficient
algorithm that uses prefix projection to reduce the search space.
"""

from collections import defaultdict
from typing import Any

from tqdm import tqdm

from sp_mining.core.interfaces import (
    SPAlgorithm,
    Sequence,
    Itemset,
    SequentialPattern,
    SequentialRule,
)


class PrefixSpanAlgorithm(SPAlgorithm):
    """PrefixSpan algorithm for sequential pattern mining.

    PrefixSpan is a pattern-growth approach that:
    1. Finds all frequent items (length-1 patterns)
    2. For each frequent item, creates a projected database
    3. Recursively mines patterns in projected databases

    Example:
        >>> algo = PrefixSpanAlgorithm(min_support=0.1)
        >>> result = algo.run(sequences)
        >>> print(f"Found {len(result.patterns)} patterns")
    """

    def __init__(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.5,
        max_pattern_length: int | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize PrefixSpan algorithm.

        Args:
            min_support: Minimum support threshold (0.0 to 1.0).
            min_confidence: Minimum confidence for rules.
            max_pattern_length: Maximum pattern length. None for no limit.
            verbose: If True, show progress bars.
        """
        super().__init__(min_support, min_confidence, max_pattern_length)
        self._patterns: list[SequentialPattern] = []
        self._num_sequences: int = 0
        self._verbose = verbose
        self._pbar: tqdm | None = None

    @property
    def name(self) -> str:
        return "PrefixSpan"

    def _get_frequent_items(
        self, sequences: list[Sequence]
    ) -> dict[str, int]:
        """Find all frequent single items."""
        item_counts: dict[str, int] = defaultdict(int)

        for seq in sequences:
            # Count each item once per sequence
            seen_items: set[str] = set()
            for element in seq.elements:
                for item in element.items:
                    if item not in seen_items:
                        item_counts[item] += 1
                        seen_items.add(item)

        min_count = self.min_support * self._num_sequences
        return {
            item: count
            for item, count in item_counts.items()
            if count >= min_count
        }

    def _project_database(
        self,
        sequences: list[tuple[Sequence, int, int]],
        prefix_item: str,
    ) -> list[tuple[Sequence, int, int]]:
        """Create projected database for a prefix item.

        Args:
            sequences: List of (sequence, element_idx, item_idx) tuples.
            prefix_item: The item to project on.

        Returns:
            Projected database as list of (sequence, element_idx, item_idx).
        """
        projected: list[tuple[Sequence, int, int]] = []

        for seq, start_elem, start_item in sequences:
            found = False
            for elem_idx in range(start_elem, len(seq.elements)):
                element = seq.elements[elem_idx]
                items = sorted(element.items)

                # Determine starting item index
                item_start = start_item if elem_idx == start_elem else 0

                for item_idx, item in enumerate(items):
                    if item_idx < item_start:
                        continue

                    if item == prefix_item:
                        # Project from the position after this item
                        if item_idx + 1 < len(items):
                            # More items in same element
                            projected.append((seq, elem_idx, item_idx + 1))
                        elif elem_idx + 1 < len(seq.elements):
                            # Next element
                            projected.append((seq, elem_idx + 1, 0))
                        found = True
                        break

                if found:
                    break

        return projected

    def _count_items_in_projected(
        self,
        projected_db: list[tuple[Sequence, int, int]],
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Count items in projected database.

        Returns:
            Tuple of (same_element_counts, subsequent_element_counts)
        """
        same_element: dict[str, int] = defaultdict(int)
        subsequent: dict[str, int] = defaultdict(int)

        for seq, elem_idx, item_idx in projected_db:
            seen_same: set[str] = set()
            seen_subsequent: set[str] = set()

            # Items in same element (after current position)
            if elem_idx < len(seq.elements):
                element = seq.elements[elem_idx]
                items = sorted(element.items)
                for idx, item in enumerate(items):
                    if idx >= item_idx and item not in seen_same:
                        same_element[item] += 1
                        seen_same.add(item)

            # Items in subsequent elements
            for e_idx in range(elem_idx + 1, len(seq.elements)):
                for item in seq.elements[e_idx].items:
                    if item not in seen_subsequent:
                        subsequent[item] += 1
                        seen_subsequent.add(item)

        return same_element, subsequent

    def _prefixspan_recursive(
        self,
        prefix: list[Itemset],
        projected_db: list[tuple[Sequence, int, int]],
    ) -> None:
        """Recursively mine patterns using PrefixSpan.

        Args:
            prefix: Current pattern prefix.
            projected_db: Current projected database.
        """
        if self.max_pattern_length and len(prefix) >= self.max_pattern_length:
            return

        min_count = self.min_support * self._num_sequences
        same_element, subsequent = self._count_items_in_projected(projected_db)

        # Process items that can extend the last element
        if prefix:
            last_itemset = prefix[-1]
            for item, count in same_element.items():
                if count >= min_count:
                    # Extend last element
                    new_items = frozenset(last_itemset.items | {item})
                    new_prefix = prefix[:-1] + [Itemset(items=new_items)]

                    pattern = SequentialPattern(
                        elements=tuple(new_prefix),
                        support=count / self._num_sequences,
                        support_count=count,
                    )
                    self._patterns.append(pattern)

                    # Update progress bar
                    if self._pbar:
                        self._pbar.set_postfix(patterns=len(self._patterns))

                    # Create new projected database for this extension
                    new_projected = self._project_database(projected_db, item)
                    if new_projected:
                        self._prefixspan_recursive(new_prefix, new_projected)

        # Process items for new elements
        for item, count in subsequent.items():
            if count >= min_count:
                new_prefix = prefix + [Itemset(items=frozenset([item]))]

                pattern = SequentialPattern(
                    elements=tuple(new_prefix),
                    support=count / self._num_sequences,
                    support_count=count,
                )
                self._patterns.append(pattern)

                # Update progress bar
                if self._pbar:
                    self._pbar.set_postfix(patterns=len(self._patterns))

                # Create new projected database
                new_projected: list[tuple[Sequence, int, int]] = []
                for seq, elem_idx, item_idx in projected_db:
                    for e_idx in range(elem_idx + 1, len(seq.elements)):
                        if item in seq.elements[e_idx].items:
                            items = sorted(seq.elements[e_idx].items)
                            i_idx = items.index(item)
                            if i_idx + 1 < len(items):
                                new_projected.append((seq, e_idx, i_idx + 1))
                            elif e_idx + 1 < len(seq.elements):
                                new_projected.append((seq, e_idx + 1, 0))
                            break

                if new_projected:
                    self._prefixspan_recursive(new_prefix, new_projected)

    def find_sequential_patterns(
        self, sequences: list[Sequence]
    ) -> list[SequentialPattern]:
        """Find frequent sequential patterns using PrefixSpan.

        Args:
            sequences: List of Sequence objects.

        Returns:
            List of SequentialPattern objects.
        """
        self._patterns = []
        self._num_sequences = len(sequences)

        if not sequences:
            return []

        # Find frequent 1-patterns
        frequent_items = self._get_frequent_items(sequences)

        with tqdm(
            total=len(frequent_items),
            desc="PrefixSpan mining",
            disable=not self._verbose,
            unit="prefix",
        ) as pbar:
            self._pbar = pbar

            for item, count in frequent_items.items():
                pbar.set_description(f"Mining prefix: {item}")

                pattern = SequentialPattern(
                    elements=(Itemset(items=frozenset([item])),),
                    support=count / self._num_sequences,
                    support_count=count,
                )
                self._patterns.append(pattern)

                # Create initial projected database
                projected_db: list[tuple[Sequence, int, int]] = []
                for seq in sequences:
                    for elem_idx, element in enumerate(seq.elements):
                        if item in element.items:
                            items = sorted(element.items)
                            item_idx = items.index(item)
                            if item_idx + 1 < len(items):
                                projected_db.append((seq, elem_idx, item_idx + 1))
                            elif elem_idx + 1 < len(seq.elements):
                                projected_db.append((seq, elem_idx + 1, 0))
                            break

                if projected_db:
                    prefix = [Itemset(items=frozenset([item]))]
                    self._prefixspan_recursive(prefix, projected_db)

                pbar.update(1)
                pbar.set_postfix(patterns=len(self._patterns))

            self._pbar = None

        return self._patterns

    def generate_rules(
        self, patterns: list[SequentialPattern]
    ) -> list[SequentialRule]:
        """Generate sequential rules from patterns.

        For each pattern, generate rules by splitting it into
        antecedent and consequent subsequences.

        Args:
            patterns: List of sequential patterns.

        Returns:
            List of SequentialRule objects.
        """
        rules: list[SequentialRule] = []

        # Build pattern support lookup
        pattern_supports: dict[tuple[Itemset, ...], float] = {
            p.elements: p.support for p in patterns
        }

        patterns_to_process = [p for p in patterns if len(p.elements) >= 2]

        for pattern in tqdm(
            patterns_to_process,
            desc="Generating rules",
            disable=not self._verbose,
            unit="pattern",
        ):
            # Generate rules by splitting at each position
            for split_pos in range(1, len(pattern.elements)):
                antecedent = pattern.elements[:split_pos]
                consequent = pattern.elements[split_pos:]

                # Get antecedent support
                ant_support = pattern_supports.get(antecedent)
                if ant_support is None or ant_support == 0:
                    continue

                confidence = pattern.support / ant_support

                if confidence >= self.min_confidence:
                    # Calculate lift
                    con_support = pattern_supports.get(consequent, 0)
                    lift = (
                        confidence / con_support
                        if con_support > 0
                        else float("inf")
                    )

                    rules.append(
                        SequentialRule(
                            antecedent=antecedent,
                            consequent=consequent,
                            support=pattern.support,
                            confidence=confidence,
                            lift=lift,
                        )
                    )

        return rules

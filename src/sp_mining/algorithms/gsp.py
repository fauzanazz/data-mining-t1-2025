"""GSP (Generalized Sequential Pattern) algorithm implementation.

GSP is an Apriori-like algorithm for sequential pattern mining that
uses a level-wise candidate generation approach.
"""

from collections import defaultdict
from itertools import combinations

from tqdm import tqdm

from sp_mining.core.interfaces import (
    SPAlgorithm,
    Sequence,
    Itemset,
    SequentialPattern,
    SequentialRule,
)


class GSPAlgorithm(SPAlgorithm):
    """GSP algorithm for sequential pattern mining.

    GSP (Generalized Sequential Pattern) mining:
    1. Find all frequent 1-sequences
    2. Generate candidate k-sequences from (k-1)-sequences
    3. Scan database to count support
    4. Repeat until no more frequent sequences

    Example:
        >>> algo = GSPAlgorithm(min_support=0.1)
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
        """Initialize GSP algorithm.

        Args:
            min_support: Minimum support threshold (0.0 to 1.0).
            min_confidence: Minimum confidence for rules.
            max_pattern_length: Maximum pattern length. None for no limit.
            verbose: If True, show progress bars.
        """
        super().__init__(min_support, min_confidence, max_pattern_length)
        self._num_sequences: int = 0
        self._verbose = verbose

    @property
    def name(self) -> str:
        return "GSP"

    def _is_subsequence(
        self,
        pattern: tuple[Itemset, ...],
        sequence: Sequence,
    ) -> bool:
        """Check if pattern is a subsequence of sequence.

        Args:
            pattern: The pattern to check.
            sequence: The sequence to check against.

        Returns:
            True if pattern is contained in sequence.
        """
        if not pattern:
            return True
        if not sequence.elements:
            return False

        pattern_idx = 0
        pattern_itemset = pattern[pattern_idx]

        for element in sequence.elements:
            # Check if pattern itemset is subset of current element
            if pattern_itemset.items.issubset(element.items):
                pattern_idx += 1
                if pattern_idx >= len(pattern):
                    return True
                pattern_itemset = pattern[pattern_idx]

        return False

    def _count_support(
        self,
        candidates: list[tuple[Itemset, ...]],
        sequences: list[Sequence],
        pbar: tqdm | None = None,
    ) -> dict[tuple[Itemset, ...], int]:
        """Count support for candidate patterns.

        Args:
            candidates: List of candidate patterns.
            sequences: List of sequences.
            pbar: Optional progress bar.

        Returns:
            Dictionary mapping patterns to their support counts.
        """
        counts: dict[tuple[Itemset, ...], int] = defaultdict(int)

        for candidate in candidates:
            for sequence in sequences:
                if self._is_subsequence(candidate, sequence):
                    counts[candidate] += 1
            if pbar:
                pbar.update(1)

        return counts

    def _generate_candidates_join(
        self,
        frequent_patterns: list[tuple[Itemset, ...]],
    ) -> list[tuple[Itemset, ...]]:
        """Generate candidate sequences by joining frequent patterns.

        Uses GSP candidate generation: join two (k-1) patterns if
        the subsequence obtained by dropping the first item of the first
        equals the subsequence obtained by dropping the last item of the second.

        Args:
            frequent_patterns: List of frequent (k-1)-patterns.

        Returns:
            List of candidate k-patterns.
        """
        candidates: list[tuple[Itemset, ...]] = []

        for i, p1 in enumerate(frequent_patterns):
            for p2 in frequent_patterns[i:]:
                # Try to join p1 and p2
                joined = self._try_join(p1, p2)
                if joined:
                    candidates.extend(joined)

        return list(set(candidates))

    def _try_join(
        self,
        p1: tuple[Itemset, ...],
        p2: tuple[Itemset, ...],
    ) -> list[tuple[Itemset, ...]]:
        """Try to join two patterns.

        Args:
            p1: First pattern.
            p2: Second pattern.

        Returns:
            List of joined patterns (may be empty).
        """
        results: list[tuple[Itemset, ...]] = []

        # Get the suffix of p1 (dropping first item)
        # Get the prefix of p2 (dropping last item)
        # If they match, we can join

        # Flatten to compare
        p1_items = []
        for itemset in p1:
            for item in sorted(itemset.items):
                p1_items.append(item)

        p2_items = []
        for itemset in p2:
            for item in sorted(itemset.items):
                p2_items.append(item)

        if len(p1_items) != len(p2_items):
            return results

        # Check if p1[1:] == p2[:-1]
        if p1_items[1:] == p2_items[:-1]:
            # Join by appending last item of p2
            last_item = p2_items[-1]

            # Two cases: add to last itemset or create new itemset
            # Case 1: New itemset
            new_pattern = p1 + (Itemset(items=frozenset([last_item])),)
            results.append(new_pattern)

            # Case 2: Extend last itemset (if last itemset of p1 has 1 item)
            if len(p1[-1].items) == 1 and len(p2[-1].items) == 1:
                first_item = list(p1[-1].items)[0]
                if first_item < last_item:  # Maintain ordering
                    extended = Itemset(items=frozenset([first_item, last_item]))
                    new_pattern = p1[:-1] + (extended,)
                    results.append(new_pattern)

        return results

    def _prune_candidates(
        self,
        candidates: list[tuple[Itemset, ...]],
        frequent_patterns: set[tuple[Itemset, ...]],
    ) -> list[tuple[Itemset, ...]]:
        """Prune candidates whose (k-1) subsequences are not all frequent.

        Args:
            candidates: List of candidate patterns.
            frequent_patterns: Set of frequent (k-1)-patterns.

        Returns:
            Pruned list of candidates.
        """
        pruned: list[tuple[Itemset, ...]] = []

        for candidate in candidates:
            # Generate all (k-1) subsequences
            valid = True

            # Simple pruning: check if removing any single element creates
            # a pattern in frequent_patterns
            for i in range(len(candidate)):
                subseq = candidate[:i] + candidate[i + 1 :]
                if subseq and subseq not in frequent_patterns:
                    valid = False
                    break

            if valid:
                pruned.append(candidate)

        return pruned

    def find_sequential_patterns(
        self, sequences: list[Sequence]
    ) -> list[SequentialPattern]:
        """Find frequent sequential patterns using GSP.

        Args:
            sequences: List of Sequence objects.

        Returns:
            List of SequentialPattern objects.
        """
        self._num_sequences = len(sequences)
        if not sequences:
            return []

        all_patterns: list[SequentialPattern] = []
        min_count = self.min_support * self._num_sequences

        # Step 1: Find frequent 1-patterns
        item_counts: dict[str, int] = defaultdict(int)
        for seq in tqdm(
            sequences,
            desc="GSP: Counting items",
            disable=not self._verbose,
            unit="seq",
        ):
            seen: set[str] = set()
            for element in seq.elements:
                for item in element.items:
                    if item not in seen:
                        item_counts[item] += 1
                        seen.add(item)

        frequent_1: list[tuple[Itemset, ...]] = []
        for item, count in item_counts.items():
            if count >= min_count:
                pattern = (Itemset(items=frozenset([item])),)
                frequent_1.append(pattern)
                all_patterns.append(
                    SequentialPattern(
                        elements=pattern,
                        support=count / self._num_sequences,
                        support_count=count,
                    )
                )

        # Step 2: Level-wise generation
        current_frequent = frequent_1
        k = 2

        while current_frequent:
            if self.max_pattern_length and k > self.max_pattern_length:
                break

            # Generate candidates
            candidates = self._generate_candidates_join(current_frequent)

            # Prune candidates
            frequent_set = set(current_frequent)
            candidates = self._prune_candidates(candidates, frequent_set)

            if not candidates:
                break

            # Count support with progress bar
            with tqdm(
                total=len(candidates),
                desc=f"GSP: Level {k} ({len(candidates)} candidates)",
                disable=not self._verbose,
                unit="cand",
            ) as pbar:
                counts = self._count_support(candidates, sequences, pbar)

            # Filter frequent
            current_frequent = []
            for pattern, count in counts.items():
                if count >= min_count:
                    current_frequent.append(pattern)
                    all_patterns.append(
                        SequentialPattern(
                            elements=pattern,
                            support=count / self._num_sequences,
                            support_count=count,
                        )
                    )

            k += 1

        return all_patterns

    def generate_rules(
        self, patterns: list[SequentialPattern]
    ) -> list[SequentialRule]:
        """Generate sequential rules from patterns.

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

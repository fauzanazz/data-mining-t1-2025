"""Data transformers for converting DataFrames to sequence format."""

from typing import Callable

import pandas as pd

from sp_mining.core.interfaces import Sequence, Itemset


class TemporalTransactionTransformer:
    """Transform temporal transaction data into sequences.

    This transformer is designed for datasets where each row represents
    a transaction item with a timestamp, grouped by customer/entity ID.

    The transactions are ordered by time to create sequences.

    Example:
        >>> transformer = TemporalTransactionTransformer(
        ...     sequence_col="CustomerID",
        ...     item_col="ProductCategory",
        ...     time_col="TransactionDate"
        ... )
        >>> sequences = transformer.transform(df)
    """

    def __init__(
        self,
        sequence_col: str = "CustomerID",
        item_col: str = "ProductCategory",
        time_col: str = "TransactionDate",
        name: str = "TemporalTransformer",
    ) -> None:
        """Initialize the transformer.

        Args:
            sequence_col: Column to group sequences by (e.g., CustomerID).
            item_col: Column containing items.
            time_col: Column containing timestamps for ordering.
            name: Name identifier for this transformer.
        """
        self._sequence_col = sequence_col
        self._item_col = item_col
        self._time_col = time_col
        self._name = name

    @property
    def name(self) -> str:
        """Return the name of this transformer."""
        return self._name

    def transform(self, df: pd.DataFrame) -> list[Sequence]:
        """Transform DataFrame to list of Sequences.

        Args:
            df: DataFrame with temporal transaction data.

        Returns:
            List of Sequence objects ordered by time.
        """
        required_cols = [self._sequence_col, self._item_col, self._time_col]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

        # Sort by sequence ID and time
        df_sorted = df.sort_values([self._sequence_col, self._time_col])

        sequences: list[Sequence] = []

        # Group by sequence ID
        for seq_id, group in df_sorted.groupby(self._sequence_col):
            # Group items by time to create itemsets
            elements: list[Itemset] = []

            for _, time_group in group.groupby(self._time_col):
                items = frozenset(time_group[self._item_col].astype(str).unique())
                if items:
                    elements.append(Itemset(items=items))

            if elements:
                sequences.append(
                    Sequence(
                        sequence_id=seq_id,
                        elements=tuple(elements),
                    )
                )

        return sequences


class EventSequenceTransformer:
    """Transform event log data into sequences.

    Suitable for datasets where each row is an event with
    timestamp and event type/action.
    """

    def __init__(
        self,
        sequence_col: str = "SessionID",
        event_col: str = "EventType",
        time_col: str = "Timestamp",
        name: str = "EventTransformer",
    ) -> None:
        """Initialize event sequence transformer.

        Args:
            sequence_col: Column to group sequences by.
            event_col: Column containing event types.
            time_col: Column containing timestamps.
            name: Name identifier.
        """
        self._sequence_col = sequence_col
        self._event_col = event_col
        self._time_col = time_col
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def transform(self, df: pd.DataFrame) -> list[Sequence]:
        """Transform DataFrame to list of Sequences.

        Each event becomes a single-item itemset in the sequence.
        """
        required_cols = [self._sequence_col, self._event_col, self._time_col]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

        df_sorted = df.sort_values([self._sequence_col, self._time_col])
        sequences: list[Sequence] = []

        for seq_id, group in df_sorted.groupby(self._sequence_col):
            elements: list[Itemset] = []

            for _, row in group.iterrows():
                event = str(row[self._event_col])
                elements.append(Itemset(items=frozenset([event])))

            if elements:
                sequences.append(
                    Sequence(
                        sequence_id=seq_id,
                        elements=tuple(elements),
                    )
                )

        return sequences


class SessionTransformer:
    """Transform session-based data into sequences.

    Groups events by session with configurable time windows.
    """

    def __init__(
        self,
        user_col: str = "UserID",
        item_col: str = "Item",
        time_col: str = "Timestamp",
        session_gap_minutes: int = 30,
        name: str = "SessionTransformer",
    ) -> None:
        """Initialize session transformer.

        Args:
            user_col: Column for user identification.
            item_col: Column for items/actions.
            time_col: Column for timestamps.
            session_gap_minutes: Gap in minutes to split sessions.
            name: Name identifier.
        """
        self._user_col = user_col
        self._item_col = item_col
        self._time_col = time_col
        self._session_gap = pd.Timedelta(minutes=session_gap_minutes)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def transform(self, df: pd.DataFrame) -> list[Sequence]:
        """Transform DataFrame to list of session-based Sequences."""
        required_cols = [self._user_col, self._item_col, self._time_col]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

        # Ensure time column is datetime
        df = df.copy()
        df[self._time_col] = pd.to_datetime(df[self._time_col])
        df_sorted = df.sort_values([self._user_col, self._time_col])

        sequences: list[Sequence] = []
        seq_counter = 0

        for user_id, user_group in df_sorted.groupby(self._user_col):
            # Split into sessions based on time gap
            user_group = user_group.reset_index(drop=True)
            time_diff = user_group[self._time_col].diff()

            session_starts = time_diff > self._session_gap
            session_ids = session_starts.cumsum()

            for session_id, session_group in user_group.groupby(session_ids):
                elements: list[Itemset] = []

                for _, row in session_group.iterrows():
                    item = str(row[self._item_col])
                    elements.append(Itemset(items=frozenset([item])))

                if elements:
                    sequences.append(
                        Sequence(
                            sequence_id=f"{user_id}_{session_id}",
                            elements=tuple(elements),
                            metadata={"user_id": user_id, "session": session_id},
                        )
                    )
                    seq_counter += 1

        return sequences


class GenericSequenceTransformer:
    """Generic transformer with custom transformation function.

    Allows users to define custom transformation logic.
    """

    def __init__(
        self,
        transform_fn: Callable[[pd.DataFrame], list[Sequence]],
        name: str = "GenericSequenceTransformer",
    ) -> None:
        """Initialize with custom transform function.

        Args:
            transform_fn: Function that takes DataFrame and returns Sequences.
            name: Name identifier.
        """
        self._transform_fn = transform_fn
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def transform(self, df: pd.DataFrame) -> list[Sequence]:
        """Apply custom transformation."""
        return self._transform_fn(df)

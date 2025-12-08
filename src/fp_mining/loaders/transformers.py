"""Data transformers for converting DataFrames to transaction format."""

from typing import Callable, Optional

import pandas as pd


class RetailTransactionTransformer:
    """Transform retail transaction data into itemset format.

    This transformer is designed for retail datasets where each row
    represents a transaction item, and transactions are grouped by
    a transaction/customer ID.

    Example:
        >>> transformer = RetailTransactionTransformer(
        ...     group_col="CustomerID",
        ...     item_col="ProductCategory"
        ... )
        >>> transactions = transformer.transform(df)
    """

    def __init__(
        self,
        group_col: str = "CustomerID",
        item_col: str = "ProductCategory",
        name: str = "RetailTransformer",
    ) -> None:
        """Initialize the transformer.

        Args:
            group_col: Column name to group transactions by.
            item_col: Column name containing item identifiers.
            name: Name identifier for this transformer.
        """
        self._group_col = group_col
        self._item_col = item_col
        self._name = name

    @property
    def name(self) -> str:
        """Return the name of this transformer."""
        return self._name

    def transform(self, df: pd.DataFrame) -> list[list[str]]:
        """Transform DataFrame to list of transactions.

        Args:
            df: DataFrame with transaction data.

        Returns:
            List of transactions, where each transaction is a list of items.

        Raises:
            KeyError: If required columns are not present.
        """
        if self._group_col not in df.columns:
            raise KeyError(f"Group column '{self._group_col}' not found in DataFrame")
        if self._item_col not in df.columns:
            raise KeyError(f"Item column '{self._item_col}' not found in DataFrame")

        # Group by transaction and collect unique items
        grouped = df.groupby(self._group_col)[self._item_col].apply(
            lambda x: list(set(x.astype(str)))
        )

        return grouped.tolist()


class BasketTransformer:
    """Transform basket/market basket data.

    For datasets where each row is already a complete transaction
    with items in separate columns or as a delimited string.
    """

    def __init__(
        self,
        item_columns: Optional[list[str]] = None,
        item_string_col: Optional[str] = None,
        delimiter: str = ",",
        name: str = "BasketTransformer",
    ) -> None:
        """Initialize basket transformer.

        Args:
            item_columns: List of column names containing items.
            item_string_col: Column with comma-separated items.
            delimiter: Delimiter for item_string_col.
            name: Name identifier for this transformer.

        Note:
            Either item_columns or item_string_col must be provided.
        """
        if item_columns is None and item_string_col is None:
            raise ValueError("Either item_columns or item_string_col must be provided")

        self._item_columns = item_columns
        self._item_string_col = item_string_col
        self._delimiter = delimiter
        self._name = name

    @property
    def name(self) -> str:
        """Return the name of this transformer."""
        return self._name

    def transform(self, df: pd.DataFrame) -> list[list[str]]:
        """Transform DataFrame to list of transactions.

        Args:
            df: DataFrame with basket data.

        Returns:
            List of transactions.
        """
        transactions: list[list[str]] = []

        if self._item_string_col:
            # Parse delimited string column
            for _, row in df.iterrows():
                items_str = str(row[self._item_string_col])
                if pd.notna(items_str) and items_str.strip():
                    items = [
                        item.strip()
                        for item in items_str.split(self._delimiter)
                        if item.strip()
                    ]
                    if items:
                        transactions.append(items)
        else:
            # Collect items from multiple columns
            for _, row in df.iterrows():
                items = []
                for col in self._item_columns or []:
                    if col in df.columns:
                        val = row[col]
                        if pd.notna(val) and str(val).strip():
                            items.append(str(val).strip())
                if items:
                    transactions.append(items)

        return transactions


class GenericTransformer:
    """Generic transformer with custom transformation function.

    Allows users to define custom transformation logic.
    """

    def __init__(
        self,
        transform_fn: Callable[[pd.DataFrame], list[list[str]]],
        name: str = "GenericTransformer",
    ) -> None:
        """Initialize with custom transform function.

        Args:
            transform_fn: Function that takes a DataFrame and returns transactions.
            name: Name identifier for this transformer.
        """
        self._transform_fn = transform_fn
        self._name = name

    @property
    def name(self) -> str:
        """Return the name of this transformer."""
        return self._name

    def transform(self, df: pd.DataFrame) -> list[list[str]]:
        """Apply custom transformation.

        Args:
            df: Input DataFrame.

        Returns:
            List of transactions.
        """
        return self._transform_fn(df)

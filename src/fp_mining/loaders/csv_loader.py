"""CSV data loader implementation."""

from pathlib import Path
from typing import Any

import pandas as pd


class CSVLoader:
    """Load data from CSV files.

    This loader supports various CSV configurations and can handle
    large files efficiently.

    Example:
        >>> loader = CSVLoader("data/transactions.csv")
        >>> df = loader.load()
    """

    def __init__(
        self,
        file_path: str | Path,
        name: str | None = None,
        **pandas_kwargs: Any,
    ) -> None:
        """Initialize CSV loader.

        Args:
            file_path: Path to the CSV file.
            name: Optional name for this loader. Defaults to filename.
            **pandas_kwargs: Additional arguments passed to pd.read_csv().
        """
        self._file_path = Path(file_path)
        self._name = name or self._file_path.stem
        self._pandas_kwargs = pandas_kwargs

    @property
    def name(self) -> str:
        """Return the name of this loader."""
        return self._name

    @property
    def file_path(self) -> Path:
        """Return the file path."""
        return self._file_path

    def load(self) -> pd.DataFrame:
        """Load the CSV file and return as DataFrame.

        Returns:
            pandas DataFrame with the loaded data.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            pd.errors.EmptyDataError: If the file is empty.
        """
        if not self._file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self._file_path}")

        return pd.read_csv(self._file_path, **self._pandas_kwargs)

    def load_sample(self, n: int = 1000) -> pd.DataFrame:
        """Load a sample of rows from the CSV file.

        Useful for testing with large datasets.

        Args:
            n: Number of rows to sample.

        Returns:
            pandas DataFrame with sampled rows.
        """
        df = self.load()
        if len(df) <= n:
            return df
        return df.sample(n=n, random_state=42)

    def __repr__(self) -> str:
        return f"CSVLoader(file_path='{self._file_path}', name='{self._name}')"

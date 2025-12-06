"""CSV data loader for sequential pattern mining."""

from pathlib import Path
from typing import Any

import pandas as pd


class SequenceCSVLoader:
    """Load sequential data from CSV files.

    This loader supports various CSV configurations for temporal
    or sequential data.

    Example:
        >>> loader = SequenceCSVLoader("data/transactions.csv")
        >>> df = loader.load()
    """

    def __init__(
        self,
        file_path: str | Path,
        name: str | None = None,
        parse_dates: list[str] | None = None,
        **pandas_kwargs: Any,
    ) -> None:
        """Initialize CSV loader.

        Args:
            file_path: Path to the CSV file.
            name: Optional name for this loader. Defaults to filename.
            parse_dates: List of column names to parse as dates.
            **pandas_kwargs: Additional arguments passed to pd.read_csv().
        """
        self._file_path = Path(file_path)
        self._name = name or self._file_path.stem
        self._parse_dates = parse_dates
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
        """
        if not self._file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self._file_path}")

        kwargs = self._pandas_kwargs.copy()
        if self._parse_dates:
            kwargs["parse_dates"] = self._parse_dates

        return pd.read_csv(self._file_path, **kwargs)

    def load_sample(self, n: int = 1000) -> pd.DataFrame:
        """Load a sample of rows from the CSV file."""
        df = self.load()
        if len(df) <= n:
            return df
        return df.sample(n=n, random_state=42)

    def __repr__(self) -> str:
        return f"SequenceCSVLoader(file_path='{self._file_path}', name='{self._name}')"

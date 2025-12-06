# Frequent Pattern Mining Pipeline

A flexible, extensible framework for frequent pattern mining with support for multiple algorithms, datasets, and evaluation metrics.

## Features

- **Interface-based design**: Clean abstractions for algorithms, data loaders, transformers, and evaluators
- **Dependency Injection**: Configurable container for managing components
- **Multiple Algorithms**: Support for Apriori, FP-Growth, and easily extensible to add more
- **Pipeline Architecture**: Run multiple datasets through multiple algorithms with multiple evaluators
- **Comprehensive Evaluation**: Coverage, rule quality, and performance metrics

## Project Structure

```
data-mining-t1-2025/
├── src/
│   └── fp_mining/
│       ├── core/
│       │   ├── interfaces.py    # Abstract base classes and protocols
│       │   ├── container.py     # Dependency injection container
│       │   └── pipeline.py      # Pipeline orchestrator
│       ├── algorithms/
│       │   ├── apriori.py       # Apriori implementation
│       │   └── fpgrowth.py      # FP-Growth implementation
│       ├── loaders/
│       │   ├── csv_loader.py    # CSV data loader
│       │   └── transformers.py  # Data transformers
│       ├── evaluators/
│       │   ├── coverage.py      # Coverage metrics
│       │   ├── quality.py       # Rule quality metrics
│       │   └── performance.py   # Performance metrics
│       └── main.py              # CLI entry point
├── examples/
│   └── multi_dataset_example.py # Usage example
├── datasets/
│   └── Retail_Transaction_Dataset.csv
└── docs/
```

## Installation

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager written in Rust.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Install in development mode
uv pip install -e .

# Or do it all in one command
uv sync
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### CLI Usage

```bash
# Run with default settings
python -m fp_mining.main -d datasets/Retail_Transaction_Dataset.csv

# Run specific algorithms with custom thresholds
python -m fp_mining.main -d datasets/Retail_Transaction_Dataset.csv \
    -s 0.05 -c 0.7 \
    -a apriori fpgrowth \
    -v
```

### Programmatic Usage

```python
from fp_mining.core.container import Container
from fp_mining.core.pipeline import FPMiningPipeline
from fp_mining.algorithms import AprioriAlgorithm, FPGrowthAlgorithm
from fp_mining.loaders import CSVLoader, RetailTransactionTransformer
from fp_mining.evaluators import CoverageEvaluator, RuleQualityEvaluator

# Create container and register components
container = Container()

container.register_loader(
    "retail",
    lambda c: CSVLoader("datasets/Retail_Transaction_Dataset.csv")
)

container.register_transformer(
    "retail_tx",
    lambda c: RetailTransactionTransformer(
        group_col="CustomerID",
        item_col="ProductCategory"
    )
)

container.register_algorithm(
    "apriori",
    lambda c: AprioriAlgorithm(min_support=0.01, min_confidence=0.5)
)

# Build and run pipeline
pipeline = FPMiningPipeline()
pipeline.add_dataset(
    container.resolve_loader("retail"),
    container.resolve_transformer("retail_tx")
)
pipeline.add_algorithm(container.resolve_algorithm("apriori"))
pipeline.add_evaluator(CoverageEvaluator())

result = pipeline.run()

# Access results
for dataset_name, algos in result.results.items():
    for algo_name, algo_result in algos.items():
        print(f"{algo_name}: {len(algo_result['result'].rules)} rules")
```

## Extending the Framework

### Adding a New Algorithm

```python
from fp_mining.core.interfaces import Algorithm, FrequentItemset, AssociationRule

class MyCustomAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "MyAlgorithm"

    def find_frequent_itemsets(self, transactions: list[list[str]]) -> list[FrequentItemset]:
        # Your implementation here
        ...

    def generate_rules(self, itemsets: list[FrequentItemset]) -> list[AssociationRule]:
        # Your implementation here
        ...
```

### Adding a New Evaluator

```python
from fp_mining.core.interfaces import Evaluator, AlgorithmResult, EvaluationResult

class MyCustomEvaluator(Evaluator):
    @property
    def name(self) -> str:
        return "MyEvaluator"

    def evaluate(
        self,
        result: AlgorithmResult,
        transactions: list[list[str]]
    ) -> EvaluationResult:
        # Your evaluation logic here
        return EvaluationResult(
            metrics={"my_metric": 0.95},
            details={"additional_info": "..."}
        )
```

### Adding a New Data Loader

```python
from fp_mining.core.interfaces import DataLoader
import pandas as pd

class DatabaseLoader:
    def __init__(self, connection_string: str, query: str):
        self._conn_str = connection_string
        self._query = query

    @property
    def name(self) -> str:
        return "DatabaseLoader"

    def load(self) -> pd.DataFrame:
        # Your database loading logic here
        ...
```

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- mlxtend >= 0.23.0
- scikit-learn >= 1.3.0
- pyyaml >= 6.0

## License

MIT

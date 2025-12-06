# Data Mining Pipeline Framework

A flexible, extensible framework for pattern mining with support for multiple algorithms, datasets, and evaluation metrics. Includes both **Frequent Pattern (FP) Mining** and **Sequential Pattern (SP) Mining**.

## Features

- **Interface-based design**: Clean abstractions for algorithms, data loaders, transformers, and evaluators
- **Dependency Injection**: Configurable container for managing components
- **YAML Configuration**: Define pipelines declaratively with configuration files
- **Multiple Algorithms**:
  - FP Mining: Apriori, FP-Growth
  - SP Mining: PrefixSpan, GSP
- **Pipeline Architecture**: Run multiple datasets through multiple algorithms with multiple evaluators
- **Comprehensive Evaluation**: Coverage, rule quality, and performance metrics
- **Progress Visualization**: Real-time progress bars with tqdm

## Project Structure

```
data-mining-t1-2025/
├── src/
│   ├── common/                 # Shared utilities
│   │   ├── __init__.py
│   │   └── config.py           # YAML config loader and parser
│   │
│   ├── fp_mining/              # Frequent Pattern Mining
│   │   ├── core/
│   │   │   ├── interfaces.py   # Algorithm, DataLoader, Evaluator, Pipeline
│   │   │   ├── container.py    # Dependency injection container
│   │   │   └── pipeline.py     # FP Pipeline orchestrator
│   │   ├── algorithms/
│   │   │   ├── apriori.py      # Apriori implementation
│   │   │   └── fpgrowth.py     # FP-Growth implementation
│   │   ├── loaders/
│   │   │   ├── csv_loader.py   # CSV data loader
│   │   │   └── transformers.py # Transaction transformers
│   │   ├── evaluators/
│   │   │   ├── coverage.py     # Coverage metrics
│   │   │   ├── quality.py      # Rule quality metrics
│   │   │   └── performance.py  # Performance metrics
│   │   ├── config_runner.py    # Config-based FP runner
│   │   └── main.py             # CLI entry point
│   │
│   ├── sp_mining/              # Sequential Pattern Mining
│   │   ├── core/
│   │   │   ├── interfaces.py   # SPAlgorithm, SequenceLoader, SPEvaluator
│   │   │   ├── container.py    # SP dependency injection container
│   │   │   └── pipeline.py     # SP Pipeline orchestrator
│   │   ├── algorithms/
│   │   │   ├── prefixspan.py   # PrefixSpan implementation
│   │   │   └── gsp.py          # GSP implementation
│   │   ├── loaders/
│   │   │   ├── csv_loader.py   # Sequence CSV loader
│   │   │   └── transformers.py # Sequence transformers
│   │   ├── evaluators/
│   │   │   ├── coverage.py     # SP Coverage metrics
│   │   │   ├── quality.py      # SP Rule quality metrics
│   │   │   └── performance.py  # SP Performance metrics
│   │   ├── config_runner.py    # Config-based SP runner
│   │   └── main.py             # CLI entry point
│   │
│   └── run_pipeline.py         # Unified CLI runner
│
├── config/                     # YAML configuration files
│   ├── default.yaml            # Default settings
│   ├── fp_mining.yaml          # FP mining configuration
│   └── sp_mining.yaml          # SP mining configuration
│
├── examples/
│   ├── multi_dataset_example.py  # FP mining example
│   └── sp_mining_example.py      # SP mining example
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

### Configuration-Based Usage (Recommended)

The easiest way to run pipelines is using YAML configuration files:

```bash
# Run FP mining with configuration
python -m run_pipeline --type fp -c config/fp_mining.yaml

# Run SP mining with configuration
python -m run_pipeline --type sp -c config/sp_mining.yaml

# Auto-detect pipeline type from config
python -m run_pipeline -c config/fp_mining.yaml

# List available configurations
python -m run_pipeline --list-configs

# Validate configuration without running
python -m run_pipeline --validate -c config/fp_mining.yaml

# Override parameters via CLI
python -m run_pipeline -c config/fp_mining.yaml -s 0.05 -f 0.7 -q
```

#### Configuration File Structure

```yaml
# config/fp_mining.yaml
pipeline:
  name: "FP Mining Pipeline"
  verbose: true

# Default parameters for all algorithms
algorithm_defaults:
  min_support: 0.01
  min_confidence: 0.5

datasets:
  - name: "Retail-Categories"
    loader:
      type: "csv"
      path: "datasets/Retail_Transaction_Dataset.csv"
    transformer:
      type: "retail"
      options:
        group_col: "CustomerID"
        item_col: "ProductCategory"

algorithms:
  - name: "apriori"
    type: "apriori"
    params:
      min_support: 0.01
      min_confidence: 0.5

  - name: "fpgrowth"
    type: "fpgrowth"
    params:
      min_support: 0.01

evaluators:
  - name: "coverage"
    type: "coverage"

  - name: "quality"
    type: "quality"
    params:
      high_confidence_threshold: 0.8
      high_lift_threshold: 2.0

output:
  save_results: true
  output_dir: "output"
  formats: ["json"]
  print_summary: true
  top_rules: 10
  top_patterns: 10
```

#### Available Components

**Loaders:**
- `csv` - Load data from CSV files

**FP Transformers:**
- `retail` - Transform retail transaction data (group by customer)
- `basket` - Transform basket data (one transaction per row)

**SP Transformers:**
- `temporal` - Transform temporal transaction data into sequences
- `event` - Transform event log data into sequences
- `session` - Transform session-based data into sequences

**FP Algorithms:**
- `apriori` - Apriori algorithm
- `fpgrowth` - FP-Growth algorithm

**SP Algorithms:**
- `prefixspan` - PrefixSpan algorithm
- `gsp` - GSP algorithm

**Evaluators:**
- `coverage` - Transaction/sequence coverage metrics
- `quality` - Rule quality metrics (confidence, lift)
- `performance` - Execution time and memory metrics

### Direct CLI Usage

#### Frequent Pattern Mining

```bash
# Run with default settings
python -m fp_mining.main -d datasets/Retail_Transaction_Dataset.csv

# Run specific algorithms with custom thresholds
python -m fp_mining.main -d datasets/Retail_Transaction_Dataset.csv \
    -s 0.05 -c 0.7 \
    -a apriori fpgrowth \
    -v
```

#### Programmatic Usage

```python
from fp_mining.core.container import Container
from fp_mining.core.pipeline import FPMiningPipeline
from fp_mining.algorithms import AprioriAlgorithm, FPGrowthAlgorithm
from fp_mining.loaders import CSVLoader, RetailTransactionTransformer
from fp_mining.evaluators import CoverageEvaluator

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
```

### Sequential Pattern Mining

#### CLI Usage

```bash
# Run with default settings
python -m sp_mining.main -d datasets/Retail_Transaction_Dataset.csv

# Run specific algorithms with custom thresholds
python -m sp_mining.main -d datasets/Retail_Transaction_Dataset.csv \
    -s 0.05 -c 0.7 \
    -m 5 \
    -a prefixspan gsp \
    -v
```

#### Programmatic Usage

```python
from sp_mining.core.container import SPContainer
from sp_mining.core.pipeline import SPMiningPipeline
from sp_mining.algorithms import PrefixSpanAlgorithm, GSPAlgorithm
from sp_mining.loaders import SequenceCSVLoader, TemporalTransactionTransformer
from sp_mining.evaluators import SPCoverageEvaluator

# Create container and register components
container = SPContainer()

container.register_loader(
    "retail",
    lambda c: SequenceCSVLoader(
        "datasets/Retail_Transaction_Dataset.csv",
        parse_dates=["TransactionDate"]
    )
)

container.register_transformer(
    "temporal",
    lambda c: TemporalTransactionTransformer(
        sequence_col="CustomerID",
        item_col="ProductCategory",
        time_col="TransactionDate"
    )
)

container.register_algorithm(
    "prefixspan",
    lambda c: PrefixSpanAlgorithm(min_support=0.01, max_pattern_length=5)
)

# Build and run pipeline
pipeline = SPMiningPipeline()
pipeline.add_dataset(
    container.resolve_loader("retail"),
    container.resolve_transformer("temporal")
)
pipeline.add_algorithm(container.resolve_algorithm("prefixspan"))
pipeline.add_evaluator(SPCoverageEvaluator())

result = pipeline.run()
```

## Extending the Framework

### Adding a New FP Algorithm

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

### Adding a New SP Algorithm

```python
from sp_mining.core.interfaces import SPAlgorithm, Sequence, SequentialPattern, SequentialRule

class MySequentialAlgorithm(SPAlgorithm):
    @property
    def name(self) -> str:
        return "MySequentialAlgorithm"

    def find_sequential_patterns(self, sequences: list[Sequence]) -> list[SequentialPattern]:
        # Your implementation here
        ...

    def generate_rules(self, patterns: list[SequentialPattern]) -> list[SequentialRule]:
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
        return EvaluationResult(
            metrics={"my_metric": 0.95},
            details={"additional_info": "..."}
        )
```

### Adding a New Data Loader

```python
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
- tqdm >= 4.66.0

## License

MIT

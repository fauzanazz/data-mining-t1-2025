"""Config-based runner for SP Mining Pipeline.

This module provides functionality to run SP mining pipelines
based on YAML configuration files.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from common.config import (
    ConfigLoader,
    PipelineConfig,
    DatasetConfig as ConfigDatasetConfig,
    AlgorithmConfig,
    EvaluatorConfig,
)
from sp_mining.core.pipeline import SPMiningPipeline, SPPipelineResult
from sp_mining.algorithms import PrefixSpanAlgorithm, GSPAlgorithm
from sp_mining.loaders import (
    SequenceCSVLoader,
    TemporalTransactionTransformer,
    EventSequenceTransformer,
    SessionTransformer,
)
from sp_mining.evaluators import (
    SPCoverageEvaluator,
    SPRuleQualityEvaluator,
    SPPerformanceEvaluator,
)


logger = logging.getLogger(__name__)


class SPConfigRunner:
    """Run SP mining pipeline from YAML configuration.

    Example:
        >>> runner = SPConfigRunner()
        >>> result = runner.run("config/sp_mining.yaml")
    """

    # Registry of available loaders
    LOADER_TYPES = {
        "csv": SequenceCSVLoader,
    }

    # Registry of available transformers
    TRANSFORMER_TYPES = {
        "temporal": TemporalTransactionTransformer,
        "event": EventSequenceTransformer,
        "session": SessionTransformer,
    }

    # Registry of available algorithms
    ALGORITHM_TYPES = {
        "prefixspan": PrefixSpanAlgorithm,
        "gsp": GSPAlgorithm,
    }

    # Registry of available evaluators
    EVALUATOR_TYPES = {
        "coverage": SPCoverageEvaluator,
        "quality": SPRuleQualityEvaluator,
        "performance": SPPerformanceEvaluator,
    }

    def __init__(self, default_config_path: Optional[Path] = None) -> None:
        """Initialize the config runner.

        Args:
            default_config_path: Path to default configuration file.
        """
        self._config_loader = ConfigLoader(default_config_path)

    def _create_loader(self, config: ConfigDatasetConfig) -> Any:
        """Create a data loader from configuration."""
        loader_type = config.loader.type.lower()

        if loader_type not in self.LOADER_TYPES:
            raise ValueError(
                f"Unknown loader type: {loader_type}. "
                f"Available: {list(self.LOADER_TYPES.keys())}"
            )

        loader_class = self.LOADER_TYPES[loader_type]

        # Handle parse_dates option
        options = config.loader.options.copy()
        if "parse_dates" in options:
            # Ensure it's a list
            if isinstance(options["parse_dates"], str):
                options["parse_dates"] = [options["parse_dates"]]

        return loader_class(
            config.loader.path,
            name=config.name,
            **options,
        )

    def _create_transformer(self, config: ConfigDatasetConfig) -> Any:
        """Create a data transformer from configuration."""
        transformer_type = config.transformer.type.lower()

        if transformer_type not in self.TRANSFORMER_TYPES:
            raise ValueError(
                f"Unknown transformer type: {transformer_type}. "
                f"Available: {list(self.TRANSFORMER_TYPES.keys())}"
            )

        transformer_class = self.TRANSFORMER_TYPES[transformer_type]
        return transformer_class(**config.transformer.options)

    def _create_algorithm(self, config: AlgorithmConfig) -> Any:
        """Create an algorithm from configuration."""
        algo_type = config.type.lower()

        if algo_type not in self.ALGORITHM_TYPES:
            raise ValueError(
                f"Unknown algorithm type: {algo_type}. "
                f"Available: {list(self.ALGORITHM_TYPES.keys())}"
            )

        algo_class = self.ALGORITHM_TYPES[algo_type]

        # Map config params to algorithm params
        params = {}
        if "min_support" in config.params:
            params["min_support"] = config.params["min_support"]
        if "min_confidence" in config.params:
            params["min_confidence"] = config.params["min_confidence"]
        if "max_pattern_length" in config.params:
            params["max_pattern_length"] = config.params["max_pattern_length"]
        if "verbose" in config.params:
            params["verbose"] = config.params["verbose"]

        return algo_class(**params)

    def _create_evaluator(self, config: EvaluatorConfig) -> Any:
        """Create an evaluator from configuration."""
        eval_type = config.type.lower()

        if eval_type not in self.EVALUATOR_TYPES:
            raise ValueError(
                f"Unknown evaluator type: {eval_type}. "
                f"Available: {list(self.EVALUATOR_TYPES.keys())}"
            )

        eval_class = self.EVALUATOR_TYPES[eval_type]

        # Some evaluators have parameters
        if eval_type == "quality":
            return eval_class(
                high_confidence_threshold=config.params.get(
                    "high_confidence_threshold", 0.8
                ),
                high_lift_threshold=config.params.get("high_lift_threshold", 2.0),
            )

        return eval_class()

    def _build_pipeline(self, config: PipelineConfig) -> SPMiningPipeline:
        """Build pipeline from configuration."""
        pipeline = SPMiningPipeline(verbose=config.verbose)

        # Add datasets
        for dataset_config in config.datasets:
            loader = self._create_loader(dataset_config)
            transformer = self._create_transformer(dataset_config)
            pipeline.add_dataset(loader, transformer, name=dataset_config.name)

        # Add algorithms
        for algo_config in config.algorithms:
            algorithm = self._create_algorithm(algo_config)
            pipeline.add_algorithm(algorithm)

        # Add evaluators
        for eval_config in config.evaluators:
            evaluator = self._create_evaluator(eval_config)
            pipeline.add_evaluator(evaluator)

        return pipeline

    def _save_results(
        self,
        result: SPPipelineResult,
        config: PipelineConfig,
    ) -> None:
        """Save results to output directory."""
        output_dir = Path(config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for fmt in config.output.formats:
            if fmt.lower() == "json":
                self._save_json(result, output_dir, timestamp)

    def _save_json(
        self,
        result: SPPipelineResult,
        output_dir: Path,
        timestamp: str,
    ) -> None:
        """Save results as JSON."""
        output_file = output_dir / f"sp_results_{timestamp}.json"

        # Convert results to serializable format
        output_data = {
            "summary": result.summary,
            "results": {},
        }

        for dataset_name, algos in result.results.items():
            output_data["results"][dataset_name] = {}
            for algo_name, algo_data in algos.items():
                algo_result = algo_data["result"]
                output_data["results"][dataset_name][algo_name] = {
                    "num_patterns": len(algo_result.patterns),
                    "num_rules": len(algo_result.rules),
                    "execution_time": algo_result.execution_time,
                    "top_patterns": [
                        {
                            "pattern": str(p),
                            "support": p.support,
                        }
                        for p in sorted(
                            algo_result.patterns,
                            key=lambda x: x.support,
                            reverse=True,
                        )[:20]
                    ],
                    "top_rules": [
                        {
                            "rule": str(r),
                            "support": r.support,
                            "confidence": r.confidence,
                            "lift": r.lift if r.lift != float("inf") else "inf",
                        }
                        for r in sorted(
                            algo_result.rules,
                            key=lambda x: x.confidence,
                            reverse=True,
                        )[:20]
                    ],
                    "evaluations": {
                        eval_name: {
                            "metrics": eval_result.metrics,
                        }
                        for eval_name, eval_result in algo_data.get(
                            "evaluations", {}
                        ).items()
                    },
                }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")

    def _print_summary(
        self,
        result: SPPipelineResult,
        config: PipelineConfig,
    ) -> None:
        """Print summary to console."""
        print("\n" + "=" * 60)
        print(f"SP MINING RESULTS: {config.name}")
        print("=" * 60)

        print(f"\nSummary:")
        print(f"  Datasets: {result.summary['num_datasets']}")
        print(f"  Algorithms: {result.summary['num_algorithms']}")
        print(f"  Evaluators: {result.summary['num_evaluators']}")
        print(f"  Total Patterns: {result.summary['total_patterns_found']}")
        print(f"  Total Rules: {result.summary['total_rules_generated']}")
        print(f"  Total Time: {result.summary['total_execution_time']:.4f}s")

        for dataset_name, dataset_results in result.results.items():
            print(f"\n{'─' * 60}")
            print(f"Dataset: {dataset_name}")
            print("─" * 60)

            for algo_name, algo_results in dataset_results.items():
                algo_result = algo_results["result"]
                print(f"\n  Algorithm: {algo_name}")
                print(f"    Patterns: {len(algo_result.patterns)}")
                print(f"    Rules: {len(algo_result.rules)}")
                print(f"    Time: {algo_result.execution_time:.4f}s")

                # Print top patterns
                if algo_result.patterns:
                    top_n = config.output.top_patterns
                    print(f"\n    Top {top_n} Patterns (by support):")
                    sorted_patterns = sorted(
                        algo_result.patterns,
                        key=lambda p: p.support,
                        reverse=True,
                    )[:top_n]
                    for i, pattern in enumerate(sorted_patterns, 1):
                        print(f"      {i}. {pattern} (support={pattern.support:.4f})")

                # Print top rules
                if algo_result.rules:
                    top_n = config.output.top_rules
                    print(f"\n    Top {top_n} Rules (by confidence):")
                    sorted_rules = sorted(
                        algo_result.rules,
                        key=lambda r: r.confidence,
                        reverse=True,
                    )[:top_n]
                    for i, rule in enumerate(sorted_rules, 1):
                        print(
                            f"      {i}. {rule} "
                            f"(conf={rule.confidence:.3f}, lift={rule.lift:.3f})"
                        )

        print("\n" + "=" * 60)

    def run(
        self,
        config_path: Path,
        overrides: Optional[dict[str, Any]] = None,
    ) -> SPPipelineResult:
        """Run pipeline from configuration file.

        Args:
            config_path: Path to YAML configuration file.
            overrides: Optional dictionary of override values.

        Returns:
            SPPipelineResult containing all results.
        """
        # Load configuration
        if overrides:
            config = self._config_loader.load_with_overrides(config_path, overrides)
        else:
            config = self._config_loader.load(config_path)

        # Validate configuration
        errors = ConfigLoader.validate_config(config)
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            raise ValueError(f"Invalid configuration: {len(errors)} errors found")

        logger.info(f"Loaded configuration: {config.name}")
        logger.info(f"  Datasets: {config.dataset_names}")
        logger.info(f"  Algorithms: {config.algorithm_names}")
        logger.info(f"  Evaluators: {config.evaluator_names}")

        # Build and run pipeline
        pipeline = self._build_pipeline(config)
        result = pipeline.run()

        # Save results if configured
        if config.output.save_results:
            self._save_results(result, config)

        # Print summary if configured
        if config.output.print_summary:
            self._print_summary(result, config)

        return result


def main() -> int:
    """CLI entry point for config-based SP mining."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SP Mining Pipeline from YAML configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/sp_mining.yaml",
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--default-config",
        type=str,
        default="config/default.yaml",
        help="Path to default configuration file",
    )

    parser.add_argument(
        "-s", "--min-support",
        type=float,
        default=None,
        help="Override minimum support threshold",
    )

    parser.add_argument(
        "-m", "--max-length",
        type=int,
        default=None,
        help="Override maximum pattern length",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Disable progress bars",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Build overrides from CLI arguments
    overrides: dict[str, Any] = {}

    if args.min_support is not None:
        overrides["algorithm_defaults"] = {"min_support": args.min_support}

    if args.max_length is not None:
        if "algorithm_defaults" not in overrides:
            overrides["algorithm_defaults"] = {}
        overrides["algorithm_defaults"]["max_pattern_length"] = args.max_length

    if args.quiet:
        overrides["pipeline"] = {"verbose": False}

    # Check config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1

    # Check default config
    default_config = None
    if Path(args.default_config).exists():
        default_config = args.default_config

    try:
        runner = SPConfigRunner(default_config_path=default_config)
        runner.run(config_path, overrides=overrides if overrides else None)
        return 0

    except Exception as e:
        logging.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

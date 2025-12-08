#!/usr/bin/env python
"""Unified CLI runner for mining pipelines.

This script provides a single entry point to run either FP or SP
mining pipelines based on YAML configuration files.

Usage:
    python -m run_pipeline --type fp --config config/fp_mining.yaml
    python -m run_pipeline --type sp --config config/sp_mining.yaml
    python -m run_pipeline -c config/fp_mining.yaml  # auto-detect type
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def detect_pipeline_type(config_path: Path) -> str:
    """Detect pipeline type from configuration file.

    Args:
        config_path: Path to configuration file.

    Returns:
        'fp' or 'sp' based on configuration content.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    # Check pipeline name
    pipeline_name = config.get("pipeline", {}).get("name", "").lower()
    if "sequential" in pipeline_name or "sp" in pipeline_name:
        return "sp"
    if "frequent" in pipeline_name or "fp" in pipeline_name:
        return "fp"

    # Check algorithm types
    algorithms = config.get("algorithms", [])
    for algo in algorithms:
        algo_type = algo.get("type", "").lower()
        if algo_type in ("prefixspan", "gsp"):
            return "sp"
        if algo_type in ("apriori", "fpgrowth"):
            return "fp"

    # Check transformer types
    datasets = config.get("datasets", [])
    for dataset in datasets:
        transformer_type = dataset.get("transformer", {}).get("type", "").lower()
        if transformer_type in ("temporal", "event", "session"):
            return "sp"
        if transformer_type in ("retail", "basket"):
            return "fp"

    # Default to FP
    return "fp"


def run_fp_pipeline(
    config_path: Path,
    default_config: Optional[str],
    overrides: Optional[Dict[str, Any]],
) -> int:
    """Run FP mining pipeline."""
    from fp_mining.config_runner import FPConfigRunner

    try:
        runner = FPConfigRunner(default_config_path=default_config)
        runner.run(config_path, overrides=overrides if overrides else None)
        return 0
    except Exception as e:
        logging.exception(f"FP Pipeline failed: {e}")
        return 1


def run_sp_pipeline(
    config_path: Path,
    default_config: Optional[str],
    overrides: Optional[Dict[str, Any]],
) -> int:
    """Run SP mining pipeline."""
    from sp_mining.config_runner import SPConfigRunner

    try:
        runner = SPConfigRunner(default_config_path=default_config)
        runner.run(config_path, overrides=overrides if overrides else None)
        return 0
    except Exception as e:
        logging.exception(f"SP Pipeline failed: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run mining pipelines from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run FP mining with config
  python -m run_pipeline --type fp -c config/fp_mining.yaml

  # Run SP mining with config
  python -m run_pipeline --type sp -c config/sp_mining.yaml

  # Auto-detect pipeline type from config
  python -m run_pipeline -c config/fp_mining.yaml

  # Override parameters
  python -m run_pipeline -c config/fp_mining.yaml -s 0.05 -q

  # List available configs
  python -m run_pipeline --list-configs
        """,
    )

    parser.add_argument(
        "-t", "--type",
        type=str,
        choices=["fp", "sp", "auto"],
        default="auto",
        help="Pipeline type: 'fp' (frequent pattern), 'sp' (sequential pattern), or 'auto' (detect from config)",
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
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
        "-f", "--min-confidence",
        type=float,
        default=None,
        help="Override minimum confidence threshold",
    )

    parser.add_argument(
        "-m", "--max-length",
        type=int,
        default=None,
        help="Override maximum pattern length (SP) or max itemset length (FP)",
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

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configuration files",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration without running pipeline",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # List configs mode
    if args.list_configs:
        config_dir = Path("config")
        if config_dir.exists():
            print("Available configuration files:")
            for config_file in sorted(config_dir.glob("*.yaml")):
                # Try to detect type
                try:
                    ptype = detect_pipeline_type(config_file)
                    print(f"  {config_file} [{ptype.upper()}]")
                except Exception:
                    print(f"  {config_file}")
        else:
            print("No config directory found")
        return 0

    # Require config file for running
    if not args.config:
        # Try default configs
        if args.type == "fp" and Path("config/fp_mining.yaml").exists():
            args.config = "config/fp_mining.yaml"
        elif args.type == "sp" and Path("config/sp_mining.yaml").exists():
            args.config = "config/sp_mining.yaml"
        else:
            parser.error("--config is required")

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1

    # Detect pipeline type if auto
    if args.type == "auto":
        args.type = detect_pipeline_type(config_path)
        logging.info(f"Auto-detected pipeline type: {args.type.upper()}")

    # Validation mode
    if args.validate:
        from common.config import ConfigLoader

        loader = ConfigLoader()
        try:
            config = loader.load(config_path)
            errors = ConfigLoader.validate_config(config)
            if errors:
                print("Configuration validation failed:")
                for error in errors:
                    print(f"  - {error}")
                return 1
            else:
                print(f"Configuration is valid: {config.name}")
                print(f"  Datasets: {config.dataset_names}")
                print(f"  Algorithms: {config.algorithm_names}")
                print(f"  Evaluators: {config.evaluator_names}")
                return 0
        except Exception as e:
            print(f"Configuration error: {e}")
            return 1

    # Build overrides from CLI arguments
    overrides: dict[str, Any] = {}

    algo_overrides: dict[str, Any] = {}
    if args.min_support is not None:
        algo_overrides["min_support"] = args.min_support
    if args.min_confidence is not None:
        algo_overrides["min_confidence"] = args.min_confidence
    if args.max_length is not None:
        if args.type == "sp":
            algo_overrides["max_pattern_length"] = args.max_length
        else:
            algo_overrides["max_len"] = args.max_length

    if algo_overrides:
        overrides["algorithm_defaults"] = algo_overrides

    if args.quiet:
        overrides["pipeline"] = {"verbose": False}

    # Check default config
    default_config = None
    if Path(args.default_config).exists():
        default_config = args.default_config

    # Run appropriate pipeline
    if args.type == "fp":
        return run_fp_pipeline(config_path, default_config, overrides)
    else:
        return run_sp_pipeline(config_path, default_config, overrides)


if __name__ == "__main__":
    sys.exit(main())

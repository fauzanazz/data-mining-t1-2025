"""Configuration loader and parser for mining pipelines.

This module provides YAML-based configuration loading and validation
for both FP and SP mining pipelines.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass
class LoaderConfig:
    """Configuration for a data loader."""

    type: str
    path: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformerConfig:
    """Configuration for a data transformer."""

    type: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    loader: LoaderConfig
    transformer: TransformerConfig


@dataclass
class AlgorithmConfig:
    """Configuration for an algorithm."""

    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatorConfig:
    """Configuration for an evaluator."""

    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """Configuration for output settings."""

    save_results: bool = False
    output_dir: str = "output"
    formats: list[str] = field(default_factory=lambda: ["json"])
    print_summary: bool = True
    top_rules: int = 5
    top_patterns: int = 5


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    name: str
    verbose: bool
    datasets: list[DatasetConfig]
    algorithms: list[AlgorithmConfig]
    evaluators: list[EvaluatorConfig]
    output: OutputConfig

    @property
    def dataset_names(self) -> list[str]:
        """Get list of dataset names."""
        return [d.name for d in self.datasets]

    @property
    def algorithm_names(self) -> list[str]:
        """Get list of algorithm names."""
        return [a.name for a in self.algorithms]

    @property
    def evaluator_names(self) -> list[str]:
        """Get list of evaluator names."""
        return [e.name for e in self.evaluators]


class ConfigLoader:
    """Load and parse YAML configuration files.

    Example:
        >>> loader = ConfigLoader()
        >>> config = loader.load("config/fp_mining.yaml")
        >>> print(config.name)
        "FP Mining Pipeline"
    """

    def __init__(self, default_config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize config loader.

        Args:
            default_config_path: Path to default configuration file.
        """
        self._default_config: dict[str, Any] = {}
        if default_config_path:
            self._default_config = self._load_yaml(Path(default_config_path))

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Load YAML file and return as dictionary."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _merge_configs(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _parse_loader_config(self, data: dict[str, Any]) -> LoaderConfig:
        """Parse loader configuration."""
        return LoaderConfig(
            type=data.get("type", "csv"),
            path=data.get("path", ""),
            options=data.get("options", {}),
        )

    def _parse_transformer_config(self, data: dict[str, Any]) -> TransformerConfig:
        """Parse transformer configuration."""
        return TransformerConfig(
            type=data.get("type", "retail"),
            options=data.get("options", {}),
        )

    def _parse_dataset_config(self, data: dict[str, Any]) -> DatasetConfig:
        """Parse dataset configuration."""
        return DatasetConfig(
            name=data.get("name", "unnamed"),
            loader=self._parse_loader_config(data.get("loader", {})),
            transformer=self._parse_transformer_config(data.get("transformer", {})),
        )

    def _parse_algorithm_config(
        self,
        data: dict[str, Any],
        defaults: dict[str, Any],
    ) -> AlgorithmConfig:
        """Parse algorithm configuration with defaults."""
        params = defaults.copy()
        params.update(data.get("params", {}))

        return AlgorithmConfig(
            name=data.get("name", data.get("type", "unnamed")),
            type=data.get("type", ""),
            params=params,
        )

    def _parse_evaluator_config(
        self,
        data: dict[str, Any],
        defaults: dict[str, Any],
    ) -> EvaluatorConfig:
        """Parse evaluator configuration with defaults."""
        params = defaults.copy()
        params.update(data.get("params", {}))

        return EvaluatorConfig(
            name=data.get("name", data.get("type", "unnamed")),
            type=data.get("type", ""),
            params=params,
        )

    def _parse_output_config(self, data: dict[str, Any]) -> OutputConfig:
        """Parse output configuration."""
        return OutputConfig(
            save_results=data.get("save_results", False),
            output_dir=data.get("output_dir", "output"),
            formats=data.get("formats", ["json"]),
            print_summary=data.get("print_summary", True),
            top_rules=data.get("top_rules", 5),
            top_patterns=data.get("top_patterns", 5),
        )

    def load(self, config_path: Union[str, Path]) -> PipelineConfig:
        """Load and parse a configuration file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Parsed PipelineConfig object.
        """
        path = Path(config_path)
        raw_config = self._load_yaml(path)

        # Merge with defaults
        if self._default_config:
            raw_config = self._merge_configs(self._default_config, raw_config)

        # Extract sections
        pipeline_section = raw_config.get("pipeline", {})
        datasets_section = raw_config.get("datasets", [])
        algorithms_section = raw_config.get("algorithms", [])
        evaluators_section = raw_config.get("evaluators", [])
        output_section = raw_config.get("output", {})

        # Get defaults for algorithms and evaluators
        algo_defaults = raw_config.get("algorithm_defaults", {})
        eval_defaults = raw_config.get("evaluator_defaults", {})

        # Parse each section
        datasets = [
            self._parse_dataset_config(d) for d in datasets_section
        ]

        algorithms = [
            self._parse_algorithm_config(a, algo_defaults)
            for a in algorithms_section
        ]

        evaluators = [
            self._parse_evaluator_config(e, eval_defaults)
            for e in evaluators_section
        ]

        output = self._parse_output_config(output_section)

        return PipelineConfig(
            name=pipeline_section.get("name", "Unnamed Pipeline"),
            verbose=pipeline_section.get("verbose", True),
            datasets=datasets,
            algorithms=algorithms,
            evaluators=evaluators,
            output=output,
        )

    def load_with_overrides(
        self,
        config_path: Union[str, Path],
        overrides: Optional[dict[str, Any]] = None,
    ) -> PipelineConfig:
        """Load configuration with command-line overrides.

        Args:
            config_path: Path to the YAML configuration file.
            overrides: Dictionary of override values.

        Returns:
            Parsed PipelineConfig with overrides applied.
        """
        path = Path(config_path)
        raw_config = self._load_yaml(path)

        # Merge with defaults
        if self._default_config:
            raw_config = self._merge_configs(self._default_config, raw_config)

        # Apply overrides
        if overrides:
            raw_config = self._merge_configs(raw_config, overrides)

        # Re-parse with overrides applied
        return self.load_from_dict(raw_config)

    def load_from_dict(self, raw_config: dict[str, Any]) -> PipelineConfig:
        """Load configuration from a dictionary.

        Args:
            raw_config: Configuration dictionary.

        Returns:
            Parsed PipelineConfig object.
        """
        # Extract sections
        pipeline_section = raw_config.get("pipeline", {})
        datasets_section = raw_config.get("datasets", [])
        algorithms_section = raw_config.get("algorithms", [])
        evaluators_section = raw_config.get("evaluators", [])
        output_section = raw_config.get("output", {})

        # Get defaults
        algo_defaults = raw_config.get("algorithm_defaults", {})
        eval_defaults = raw_config.get("evaluator_defaults", {})

        # Parse each section
        datasets = [
            self._parse_dataset_config(d) for d in datasets_section
        ]

        algorithms = [
            self._parse_algorithm_config(a, algo_defaults)
            for a in algorithms_section
        ]

        evaluators = [
            self._parse_evaluator_config(e, eval_defaults)
            for e in evaluators_section
        ]

        output = self._parse_output_config(output_section)

        return PipelineConfig(
            name=pipeline_section.get("name", "Unnamed Pipeline"),
            verbose=pipeline_section.get("verbose", True),
            datasets=datasets,
            algorithms=algorithms,
            evaluators=evaluators,
            output=output,
        )

    @staticmethod
    def validate_config(config: PipelineConfig) -> list[str]:
        """Validate configuration and return list of errors.

        Args:
            config: Configuration to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        if not config.datasets:
            errors.append("No datasets configured")

        if not config.algorithms:
            errors.append("No algorithms configured")

        for dataset in config.datasets:
            if not dataset.loader.path:
                errors.append(f"Dataset '{dataset.name}': loader path is required")

            path = Path(dataset.loader.path)
            if not path.exists():
                errors.append(
                    f"Dataset '{dataset.name}': file not found: {dataset.loader.path}"
                )

        return errors

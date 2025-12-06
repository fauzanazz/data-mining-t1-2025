"""Dependency Injection Container for the FP Mining Pipeline.

This module provides a simple yet flexible DI container that manages
the registration and resolution of components like algorithms,
data loaders, transformers, and evaluators.
"""

from typing import Any, Callable, TypeVar, Generic

from fp_mining.core.interfaces import (
    Algorithm,
    DataLoader,
    DataTransformer,
    Evaluator,
)


T = TypeVar("T")


class ServiceDescriptor(Generic[T]):
    """Describes a registered service with its factory and lifecycle."""

    def __init__(
        self,
        factory: Callable[..., T],
        singleton: bool = False,
    ) -> None:
        self.factory = factory
        self.singleton = singleton
        self._instance: T | None = None

    def resolve(self, container: "Container") -> T:
        """Resolve the service instance."""
        if self.singleton:
            if self._instance is None:
                self._instance = self.factory(container)
            return self._instance
        return self.factory(container)


class Container:
    """Dependency Injection Container.

    Manages registration and resolution of services. Supports both
    singleton and transient lifecycles.

    Example:
        >>> container = Container()
        >>> container.register_algorithm("apriori", lambda c: AprioriAlgorithm())
        >>> container.register_algorithm("fpgrowth", lambda c: FPGrowthAlgorithm())
        >>> apriori = container.resolve_algorithm("apriori")
    """

    def __init__(self) -> None:
        self._algorithms: dict[str, ServiceDescriptor[Algorithm]] = {}
        self._loaders: dict[str, ServiceDescriptor[DataLoader]] = {}
        self._transformers: dict[str, ServiceDescriptor[DataTransformer]] = {}
        self._evaluators: dict[str, ServiceDescriptor[Evaluator]] = {}
        self._services: dict[str, ServiceDescriptor[Any]] = {}

    # Algorithm registration and resolution
    def register_algorithm(
        self,
        name: str,
        factory: Callable[["Container"], Algorithm],
        singleton: bool = False,
    ) -> "Container":
        """Register an algorithm factory.

        Args:
            name: Unique identifier for the algorithm.
            factory: Callable that creates the algorithm instance.
            singleton: If True, reuse the same instance.

        Returns:
            Self for method chaining.
        """
        self._algorithms[name] = ServiceDescriptor(factory, singleton)
        return self

    def resolve_algorithm(self, name: str) -> Algorithm:
        """Resolve an algorithm by name.

        Args:
            name: The registered algorithm name.

        Returns:
            Algorithm instance.

        Raises:
            KeyError: If algorithm is not registered.
        """
        if name not in self._algorithms:
            raise KeyError(f"Algorithm '{name}' is not registered. Available: {list(self._algorithms.keys())}")
        return self._algorithms[name].resolve(self)

    def get_all_algorithms(self) -> dict[str, Algorithm]:
        """Get all registered algorithms."""
        return {name: desc.resolve(self) for name, desc in self._algorithms.items()}

    # DataLoader registration and resolution
    def register_loader(
        self,
        name: str,
        factory: Callable[["Container"], DataLoader],
        singleton: bool = True,
    ) -> "Container":
        """Register a data loader factory.

        Args:
            name: Unique identifier for the loader.
            factory: Callable that creates the loader instance.
            singleton: If True, reuse the same instance (default True for loaders).

        Returns:
            Self for method chaining.
        """
        self._loaders[name] = ServiceDescriptor(factory, singleton)
        return self

    def resolve_loader(self, name: str) -> DataLoader:
        """Resolve a data loader by name."""
        if name not in self._loaders:
            raise KeyError(f"Loader '{name}' is not registered. Available: {list(self._loaders.keys())}")
        return self._loaders[name].resolve(self)

    def get_all_loaders(self) -> dict[str, DataLoader]:
        """Get all registered data loaders."""
        return {name: desc.resolve(self) for name, desc in self._loaders.items()}

    # DataTransformer registration and resolution
    def register_transformer(
        self,
        name: str,
        factory: Callable[["Container"], DataTransformer],
        singleton: bool = True,
    ) -> "Container":
        """Register a data transformer factory.

        Args:
            name: Unique identifier for the transformer.
            factory: Callable that creates the transformer instance.
            singleton: If True, reuse the same instance.

        Returns:
            Self for method chaining.
        """
        self._transformers[name] = ServiceDescriptor(factory, singleton)
        return self

    def resolve_transformer(self, name: str) -> DataTransformer:
        """Resolve a data transformer by name."""
        if name not in self._transformers:
            raise KeyError(f"Transformer '{name}' is not registered. Available: {list(self._transformers.keys())}")
        return self._transformers[name].resolve(self)

    def get_all_transformers(self) -> dict[str, DataTransformer]:
        """Get all registered transformers."""
        return {name: desc.resolve(self) for name, desc in self._transformers.items()}

    # Evaluator registration and resolution
    def register_evaluator(
        self,
        name: str,
        factory: Callable[["Container"], Evaluator],
        singleton: bool = True,
    ) -> "Container":
        """Register an evaluator factory.

        Args:
            name: Unique identifier for the evaluator.
            factory: Callable that creates the evaluator instance.
            singleton: If True, reuse the same instance.

        Returns:
            Self for method chaining.
        """
        self._evaluators[name] = ServiceDescriptor(factory, singleton)
        return self

    def resolve_evaluator(self, name: str) -> Evaluator:
        """Resolve an evaluator by name."""
        if name not in self._evaluators:
            raise KeyError(f"Evaluator '{name}' is not registered. Available: {list(self._evaluators.keys())}")
        return self._evaluators[name].resolve(self)

    def get_all_evaluators(self) -> dict[str, Evaluator]:
        """Get all registered evaluators."""
        return {name: desc.resolve(self) for name, desc in self._evaluators.items()}

    # Generic service registration
    def register(
        self,
        name: str,
        factory: Callable[["Container"], T],
        singleton: bool = False,
    ) -> "Container":
        """Register a generic service.

        Args:
            name: Unique identifier for the service.
            factory: Callable that creates the service instance.
            singleton: If True, reuse the same instance.

        Returns:
            Self for method chaining.
        """
        self._services[name] = ServiceDescriptor(factory, singleton)
        return self

    def resolve(self, name: str) -> Any:
        """Resolve a generic service by name."""
        if name not in self._services:
            raise KeyError(f"Service '{name}' is not registered.")
        return self._services[name].resolve(self)

    def clear(self) -> None:
        """Clear all registrations."""
        self._algorithms.clear()
        self._loaders.clear()
        self._transformers.clear()
        self._evaluators.clear()
        self._services.clear()


# Global container instance (optional usage)
_default_container: Container | None = None


def get_container() -> Container:
    """Get the default container instance, creating if needed."""
    global _default_container
    if _default_container is None:
        _default_container = Container()
    return _default_container


def set_container(container: Container) -> None:
    """Set the default container instance."""
    global _default_container
    _default_container = container

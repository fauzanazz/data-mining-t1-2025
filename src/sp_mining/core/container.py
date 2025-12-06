"""Dependency Injection Container for the SP Mining Pipeline.

This module provides a DI container that manages the registration
and resolution of SP mining components.
"""

from typing import Any, Callable, TypeVar, Generic

from sp_mining.core.interfaces import (
    SPAlgorithm,
    SequenceLoader,
    SequenceTransformer,
    SPEvaluator,
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

    def resolve(self, container: "SPContainer") -> T:
        """Resolve the service instance."""
        if self.singleton:
            if self._instance is None:
                self._instance = self.factory(container)
            return self._instance
        return self.factory(container)


class SPContainer:
    """Dependency Injection Container for Sequential Pattern Mining.

    Manages registration and resolution of services. Supports both
    singleton and transient lifecycles.

    Example:
        >>> container = SPContainer()
        >>> container.register_algorithm("prefixspan", lambda c: PrefixSpanAlgorithm())
        >>> container.register_algorithm("gsp", lambda c: GSPAlgorithm())
        >>> prefixspan = container.resolve_algorithm("prefixspan")
    """

    def __init__(self) -> None:
        self._algorithms: dict[str, ServiceDescriptor[SPAlgorithm]] = {}
        self._loaders: dict[str, ServiceDescriptor[SequenceLoader]] = {}
        self._transformers: dict[str, ServiceDescriptor[SequenceTransformer]] = {}
        self._evaluators: dict[str, ServiceDescriptor[SPEvaluator]] = {}
        self._services: dict[str, ServiceDescriptor[Any]] = {}

    # Algorithm registration and resolution
    def register_algorithm(
        self,
        name: str,
        factory: Callable[["SPContainer"], SPAlgorithm],
        singleton: bool = False,
    ) -> "SPContainer":
        """Register an SP algorithm factory.

        Args:
            name: Unique identifier for the algorithm.
            factory: Callable that creates the algorithm instance.
            singleton: If True, reuse the same instance.

        Returns:
            Self for method chaining.
        """
        self._algorithms[name] = ServiceDescriptor(factory, singleton)
        return self

    def resolve_algorithm(self, name: str) -> SPAlgorithm:
        """Resolve an algorithm by name."""
        if name not in self._algorithms:
            raise KeyError(
                f"Algorithm '{name}' is not registered. "
                f"Available: {list(self._algorithms.keys())}"
            )
        return self._algorithms[name].resolve(self)

    def get_all_algorithms(self) -> dict[str, SPAlgorithm]:
        """Get all registered algorithms."""
        return {name: desc.resolve(self) for name, desc in self._algorithms.items()}

    # SequenceLoader registration and resolution
    def register_loader(
        self,
        name: str,
        factory: Callable[["SPContainer"], SequenceLoader],
        singleton: bool = True,
    ) -> "SPContainer":
        """Register a sequence loader factory."""
        self._loaders[name] = ServiceDescriptor(factory, singleton)
        return self

    def resolve_loader(self, name: str) -> SequenceLoader:
        """Resolve a sequence loader by name."""
        if name not in self._loaders:
            raise KeyError(
                f"Loader '{name}' is not registered. "
                f"Available: {list(self._loaders.keys())}"
            )
        return self._loaders[name].resolve(self)

    def get_all_loaders(self) -> dict[str, SequenceLoader]:
        """Get all registered loaders."""
        return {name: desc.resolve(self) for name, desc in self._loaders.items()}

    # SequenceTransformer registration and resolution
    def register_transformer(
        self,
        name: str,
        factory: Callable[["SPContainer"], SequenceTransformer],
        singleton: bool = True,
    ) -> "SPContainer":
        """Register a sequence transformer factory."""
        self._transformers[name] = ServiceDescriptor(factory, singleton)
        return self

    def resolve_transformer(self, name: str) -> SequenceTransformer:
        """Resolve a sequence transformer by name."""
        if name not in self._transformers:
            raise KeyError(
                f"Transformer '{name}' is not registered. "
                f"Available: {list(self._transformers.keys())}"
            )
        return self._transformers[name].resolve(self)

    def get_all_transformers(self) -> dict[str, SequenceTransformer]:
        """Get all registered transformers."""
        return {name: desc.resolve(self) for name, desc in self._transformers.items()}

    # Evaluator registration and resolution
    def register_evaluator(
        self,
        name: str,
        factory: Callable[["SPContainer"], SPEvaluator],
        singleton: bool = True,
    ) -> "SPContainer":
        """Register an evaluator factory."""
        self._evaluators[name] = ServiceDescriptor(factory, singleton)
        return self

    def resolve_evaluator(self, name: str) -> SPEvaluator:
        """Resolve an evaluator by name."""
        if name not in self._evaluators:
            raise KeyError(
                f"Evaluator '{name}' is not registered. "
                f"Available: {list(self._evaluators.keys())}"
            )
        return self._evaluators[name].resolve(self)

    def get_all_evaluators(self) -> dict[str, SPEvaluator]:
        """Get all registered evaluators."""
        return {name: desc.resolve(self) for name, desc in self._evaluators.items()}

    # Generic service registration
    def register(
        self,
        name: str,
        factory: Callable[["SPContainer"], T],
        singleton: bool = False,
    ) -> "SPContainer":
        """Register a generic service."""
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
_default_container: SPContainer | None = None


def get_sp_container() -> SPContainer:
    """Get the default SP container instance, creating if needed."""
    global _default_container
    if _default_container is None:
        _default_container = SPContainer()
    return _default_container


def set_sp_container(container: SPContainer) -> None:
    """Set the default SP container instance."""
    global _default_container
    _default_container = container

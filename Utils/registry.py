from __future__ import annotations

from typing import Any, Callable


MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}
TASK_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(builder: Callable[..., Any]) -> Callable[..., Any]:
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = builder
        return builder

    return decorator


def register_task(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(builder: Callable[..., Any]) -> Callable[..., Any]:
        if name in TASK_REGISTRY:
            raise KeyError(f"Task '{name}' is already registered.")
        TASK_REGISTRY[name] = builder
        return builder

    return decorator


def build_model(name: str, *args: Any, **kwargs: Any) -> Any:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered.")
    return MODEL_REGISTRY[name](*args, **kwargs)


def build_task(name: str, *args: Any, **kwargs: Any) -> Any:
    if name not in TASK_REGISTRY:
        raise KeyError(f"Task '{name}' is not registered.")
    return TASK_REGISTRY[name](*args, **kwargs)


def list_registered_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())

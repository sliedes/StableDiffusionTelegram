from typing import TYPE_CHECKING, Any, Callable, TypeVar

T = TypeVar("T")


def mypy_dummy_decorator(func: T) -> T:
    assert False, "mypy_dummy_decorator somehow got called"
    return func


def nop_decorator_factory() -> Callable[[T], T]:
    return mypy_dummy_decorator


def typed_decorator(deco: Callable[[T], T]) -> Callable[[T], T]:
    """
    This function takes in a decorator that does not change function signature.
    It returns the decorator. However, it will be typed.
    """
    return deco


def typed_decorator_factory(*args: Any, **kwargs: Any) -> Callable[[T], T]:
    """
    This function takes in a decorator factory that does not change function signature.
    It returns the decorator. However, it will be typed.
    """
    return mypy_dummy_decorator


__all__ = ["typed_decorator", "typed_decorator_factory"]

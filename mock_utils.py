from typing import Any
from unittest.mock import ANY, Mock, _Call

from my_logging import logger


def fill_default_args(mock: Any) -> None:
    """
    Fill in the default values of a call. After this, all arguments will be keyword arguments.
    """
    if mock._spec_signature is None or mock.call_args is None:
        return

    logger.info("sig={}", mock._spec_signature)
    logger.info("args={}", mock.call_args.args)
    logger.info("kwargs={}", mock.call_args.kwargs)
    bound = mock._spec_signature.bind(*mock.call_args.args, **mock.call_args.kwargs)
    bound.apply_defaults()
    logger.info("bound={}", bound)
    logger.info("bound.kwargs={}", bound.arguments)

    mock.call_args = _Call(((), bound.arguments), two=True)


def assert_called_with_partial(
    mock: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Assert that a mock was called with the given arguments, ignoring any extra
    arguments passed to the mock. Default values of the mocked object are filled in.
    """
    if not mock.call_args:
        raise AssertionError(f"Expected {mock!r} to have been called.")

    # If the mock has no spec, there's not much we can do besides checking that the args we need are there.
    if mock._spec_signature is None:
        mock.assert_called_with(*args, **kwargs)  # FIXME: This is not a partial check.
        return

    called_bound = mock._spec_signature.bind(*mock.call_args.args, **mock.call_args.kwargs)
    called_bound.apply_defaults()

    expected_bound = mock._spec_signature.bind_partial(*args, **kwargs)

    for name, value in expected_bound.arguments.items():
        if value is not called_bound.arguments[name] and value != called_bound.arguments[name]:
            raise AssertionError(
                f"Expected {mock!r} to have been called with {name}={value!r}, got {called_bound.arguments[name]!r}."
            )


def assert_called_once_with_partial(
    mock: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Assert that a mock was called once with the given arguments, ignoring any extra
    arguments passed to the mock. Default values of the mocked object are filled in.
    """
    if mock.call_count != 1:
        raise AssertionError(f"Expected {mock!r} to have been called once, got {mock.call_count}.")

    assert_called_with_partial(mock, *args, **kwargs)

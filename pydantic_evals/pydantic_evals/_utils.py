from __future__ import annotations as _annotations

import inspect
from functools import partial
from typing import Any, Callable


class Unset:
    """A singleton to represent an unset value.

    Copied from pydantic_ai/_utils.py.
    """

    pass


UNSET = Unset()


def get_unwrapped_function_name(func: Callable[..., Any]) -> str:
    def _unwrap(f: Callable[..., Any]) -> Callable[..., Any]:
        # Unwraps f, also unwrapping partials, for the sake of getting f's name
        if isinstance(f, partial):
            return _unwrap(f.func)
        return inspect.unwrap(f)

    return _unwrap(func).__qualname__

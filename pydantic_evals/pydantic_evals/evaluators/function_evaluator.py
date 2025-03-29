import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from inspect import iscoroutinefunction
from typing import Any, Callable, Concatenate

from typing_extensions import ParamSpec, TypeVar

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

InputsT = TypeVar('InputsT', default=Any, contravariant=True)
"""Type variable for the inputs type of the task being evaluated."""

OutputT = TypeVar('OutputT', default=Any, contravariant=True)
"""Type variable for the output type of the task being evaluated."""

MetadataT = TypeVar('MetadataT', default=Any, contravariant=True)
"""Type variable for the metadata type of the task being evaluated."""

P = ParamSpec('P', default=...)


def function_evaluator(
    func: Callable[
        Concatenate[EvaluatorContext[InputsT, OutputT, MetadataT], P], EvaluatorOutput | Awaitable[EvaluatorOutput]
    ],
) -> Callable[P, Evaluator[InputsT, OutputT, MetadataT]]:
    """Decorator to create an Evaluator type from a function."""
    params = inspect.signature(func).parameters
    params = dict(list(params.items())[1:])  # remove first parameter

    if iscoroutinefunction(func):

        class MyEvaluator(Evaluator):  # type: ignore
            async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
                return await func(ctx, **self.__dict__)  # type: ignore
    else:

        class MyEvaluator(Evaluator):
            def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
                return func(ctx, **self.__dict__)  # type: ignore

    annotations = {}
    for name, p in params.items():
        annotations[name] = p.annotation
        if p.default != inspect.Parameter.empty:
            setattr(MyEvaluator, name, p.default)

    MyEvaluator.__annotations__ = {name: p.annotation for name, p in params.items()}
    MyEvaluator.__name__ = func.__name__
    MyEvaluator.__qualname__ = f'evaluator({func.__qualname__})'
    MyEvaluator = dataclass(MyEvaluator)
    return MyEvaluator  # type: ignore

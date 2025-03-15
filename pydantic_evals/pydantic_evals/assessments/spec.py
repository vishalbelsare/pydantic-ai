from __future__ import annotations

import inspect
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from functools import partial
from inspect import iscoroutinefunction
from typing import Any, Callable, Concatenate, Generic, NotRequired, cast

import anyio.to_thread
from pydantic import RootModel, TypeAdapter, ValidationError, field_validator
from pydantic._internal import _typing_extra
from typing_extensions import TypedDict, TypeVar

from .scoring import ScoringContext

InputsT = TypeVar('InputsT', default=dict[str, Any])
OutputT = TypeVar('OutputT', default=dict[str, Any])
MetadataT = TypeVar('MetadataT', default=dict[str, Any])


class AssessmentSpec(RootModel[str | dict[str, Any]]):
    """The specification of an assessment to be run.

    Corresponds to the serialized form of an assessment call in a yaml file.
    """

    @field_validator('root')
    @classmethod
    def enforce_one_key(cls, value: str | dict[str, Any]) -> Any:
        if isinstance(value, str):
            return value
        if len(value) != 1:
            raise ValueError(f'Expected a single key containing the call name, found keys {list(value.keys())}')
        return value

    @property
    def call(self) -> str:
        if isinstance(self.root, str):
            return self.root
        return next(iter(self.root.keys()))

    @property
    def arguments(self) -> Any:
        # dicts are treated as kwargs, non-dicts are passed as the first argument
        if isinstance(self.root, str):
            return {}  # a plain str means a function call without any arguments
        return next(iter(self.root.values()))


AssessmentValue = bool | int | float | str


@dataclass
class AssessmentResult:
    """The result of running an assessment."""

    value: AssessmentValue
    name: str | None = None  # If none, the name of the assessment function will be used
    reason: str | None = None


AssessmentFunctionResult = AssessmentValue | list[AssessmentValue] | AssessmentResult | list[AssessmentResult]

# TODO: Add bound=AssessmentValue to the following typevar after we upgrade to pydantic 2.11
AssessmentValueT = TypeVar('AssessmentValueT', default=AssessmentValue, covariant=True)


@dataclass
class AssessmentDetail(Generic[AssessmentValueT]):
    """The details of an assessment that has been run."""

    name: str
    value: AssessmentValueT
    reason: str | None
    source: AssessmentSpec


SyncAssessmentFunction = Callable[
    Concatenate[ScoringContext[InputsT, OutputT, MetadataT], ...], AssessmentFunctionResult
]
AsyncAssessmentFunction = Callable[
    Concatenate[ScoringContext[InputsT, OutputT, MetadataT], ...], Awaitable[AssessmentFunctionResult]
]
AssessmentFunction = (
    SyncAssessmentFunction[InputsT, OutputT, MetadataT] | AsyncAssessmentFunction[InputsT, OutputT, MetadataT]
)
AssessmentRegistry = Mapping[str, AssessmentFunction[InputsT, OutputT, MetadataT]]

SyncBoundAssessmentFunction = Callable[[ScoringContext[InputsT, OutputT, MetadataT]], AssessmentFunctionResult]
AsyncBoundAssessmentFunction = Callable[
    [ScoringContext[InputsT, OutputT, MetadataT]], Awaitable[AssessmentFunctionResult]
]
BoundAssessmentFunction = (
    SyncBoundAssessmentFunction[InputsT, OutputT, MetadataT] | AsyncBoundAssessmentFunction[InputsT, OutputT, MetadataT]
)


@dataclass
class Assessment(Generic[InputsT, OutputT, MetadataT]):
    """An assertion to be run on a scoring context."""

    spec: AssessmentSpec
    function: BoundAssessmentFunction[InputsT, OutputT, MetadataT]

    @staticmethod
    def from_function(
        function: BoundAssessmentFunction[InputsT, OutputT, MetadataT],
    ) -> Assessment[InputsT, OutputT, MetadataT]:
        spec = AssessmentSpec(function.__name__)
        return Assessment[InputsT, OutputT, MetadataT](spec=spec, function=function)

    @staticmethod
    def from_registry(
        registry: AssessmentRegistry[InputsT, OutputT, MetadataT], case_name: str, spec: AssessmentSpec
    ) -> Assessment[InputsT, OutputT, MetadataT]:
        Assessment._validate_against_registry(registry, case_name, spec)
        try:
            function = registry[spec.call]
        except KeyError:
            raise ValueError(
                f'Assessment call {spec.call!r} is not registered. Registered choices: {list(registry.keys())}'
            )

        arguments = spec.arguments
        if not isinstance(arguments, dict):
            # the first argument should always be the scoring context, so in this case we need to bind the _second_ argument
            second_argument_name = list(inspect.signature(function).parameters.keys())[1]
            arguments = {second_argument_name: arguments}
        bound_function = partial(function, **arguments)

        if iscoroutinefunction(function):
            bound_async_function = cast(AsyncBoundAssessmentFunction[InputsT, OutputT, MetadataT], bound_function)
            return Assessment[InputsT, OutputT, MetadataT](spec=spec, function=bound_async_function)
        else:
            bound_sync_function = cast(SyncBoundAssessmentFunction[InputsT, OutputT, MetadataT], bound_function)
            return Assessment[InputsT, OutputT, MetadataT](spec=spec, function=bound_sync_function)

    async def execute(self, ctx: ScoringContext[InputsT, OutputT, MetadataT]) -> list[AssessmentDetail]:
        if inspect.iscoroutinefunction(self.function):
            async_function = cast(AsyncBoundAssessmentFunction[InputsT, OutputT, MetadataT], self.function)
            results = await async_function(ctx)
        else:
            sync_function = cast(SyncBoundAssessmentFunction[InputsT, OutputT, MetadataT], self.function)
            results = await anyio.to_thread.run_sync(sync_function, ctx)

        if not isinstance(results, list):
            results = [results]

        details: list[AssessmentDetail] = []
        for result in results:
            if not isinstance(result, AssessmentResult):
                result = AssessmentResult(value=result)
            details.append(
                AssessmentDetail(
                    name=result.name or self.spec.call, value=result.value, reason=result.reason, source=self.spec
                )
            )

        return details

    @staticmethod
    def _validate_against_registry(
        registry: AssessmentRegistry[Any, Any, Any], case_name: str, spec: AssessmentSpec
    ) -> None:
        function = registry.get(spec.call)
        if function is None:
            raise ValueError(
                f'Assertion call {spec.call!r} is not registered. Registered choices: {list(registry.keys())}'
            )

        signature = inspect.signature(function)
        first_param = list(signature.parameters.values())[0]
        type_hints = _typing_extra.get_function_type_hints(function)
        type_hints.pop(first_param.name)
        type_hints.pop('return', None)
        for p in signature.parameters.values():
            if p.default is not p.empty:
                type_hints[p.name] = NotRequired[type_hints[p.name]]

        td = TypedDict(function.__name__, type_hints)  # type: ignore
        td.__pydantic_config__ = {'extra': 'forbid'}  # type: ignore
        adapter = TypeAdapter(td)
        arguments = spec.arguments
        if not isinstance(arguments, dict):
            first_kwarg = '__arg'
            for kwarg in type_hints:
                first_kwarg = kwarg
                break
            arguments = {first_kwarg: arguments}

        try:
            adapter.validate_python(arguments)
        except ValidationError as e:
            error_message = str(e)
            error_message = error_message.replace(' for typed-dict', ':')
            raise ValueError(f'Assessment {spec.call!r} in case {case_name!r} has {error_message}')


_DEFAULT_REGISTRY: dict[str, AssessmentFunction[Any, Any, Any]] = {}

F = TypeVar('F', bound=AssessmentFunction[Any, Any, Any])


def assessment(f: F) -> F:
    """Decorator that registers an assessment function in the default registry."""
    _DEFAULT_REGISTRY[f.__name__] = f
    return f


def get_default_registry() -> AssessmentRegistry:
    """Get the default assessment registry."""
    return _DEFAULT_REGISTRY

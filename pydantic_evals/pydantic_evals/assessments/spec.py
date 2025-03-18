from __future__ import annotations

import inspect
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Concatenate, Generic, NotRequired, cast

import anyio.to_thread
from pydantic import (
    BaseModel,
    Field,
    ModelWrapValidatorHandler,
    RootModel,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic._internal import _typing_extra
from pydantic_core.core_schema import SerializerFunctionWrapHandler
from typing_extensions import TypedDict, TypeGuard, TypeVar

from .._utils import get_unwrapped_function_name
from ..otel.span_tree import SpanTree

InputsT = TypeVar('InputsT', default=dict[str, Any])
OutputT = TypeVar('OutputT', default=dict[str, Any])
MetadataT = TypeVar('MetadataT', default=dict[str, Any])


@dataclass
class ScoringContext(Generic[InputsT, OutputT, MetadataT]):
    """Context for scoring an evaluation case."""

    name: str
    inputs: InputsT
    metadata: MetadataT
    expected_output: OutputT | None

    output: OutputT
    duration: float
    span_tree: SpanTree

    attributes: dict[str, Any]
    metrics: dict[str, int | float]


class AssessmentSpec(BaseModel):
    """The specification of an assessment to be run."""

    call: str
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='wrap')
    @classmethod
    def deserialize(cls, value: Any, handler: ModelWrapValidatorHandler[AssessmentSpec]) -> AssessmentSpec:
        try:
            result = handler(value)
            return result
        except ValidationError as exc:
            try:
                deserialized = _SerializedAssessmentSpec.model_validate(value)
            except ValidationError:
                raise exc  # raise the original error; TODO: We should somehow combine the errors
            return deserialized.to_assessment_spec()

    @model_serializer(mode='wrap')
    def serialize(self, handler: SerializerFunctionWrapHandler) -> Any:
        # In this case, use the standard "long-form" serialization
        if len(self.args) > 1:
            return handler(self)

        # Note: The rest of this logic needs to be kept in sync with the definition of _SerializedAssessmentSpec
        # In this case, use the shortest compatible form of serialization:
        if not self.args and not self.kwargs:
            return self.call
        elif len(self.args) == 1:
            return {self.call: self.args[0]}
        else:
            return {self.call: self.kwargs}


class _SerializedAssessmentSpec(RootModel[str | dict[str, Any]]):
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
    def _call(self) -> str:
        if isinstance(self.root, str):
            return self.root
        return next(iter(self.root.keys()))

    @property
    def _args_kwargs(self) -> tuple[list[Any], dict[str, Any]]:
        # dicts are treated as kwargs, non-dicts are passed as the first argument
        if isinstance(self.root, str):
            return [], {}  # a plain str means a function call without any arguments
        value = next(iter(self.root.values()))

        if isinstance(value, dict):
            keys: list[Any] = list(value.keys())  # pyright: ignore[reportUnknownArgumentType]
            if all(isinstance(k, str) for k in keys):
                return [], cast(dict[str, Any], value)

        return [value], {}

    def to_assessment_spec(self) -> AssessmentSpec:
        call = self._call
        args, kwargs = self._args_kwargs
        return AssessmentSpec(call=call, args=args, kwargs=kwargs)


AssessmentValue = bool | int | float | str


@dataclass
class AssessmentResult:
    """The result of running an assessment."""

    value: AssessmentValue
    reason: str | None = None


AssessmentFunctionResult = AssessmentValue | AssessmentResult | Mapping[str, AssessmentValue | AssessmentResult]


def _convert_to_mapping(
    result: AssessmentFunctionResult, *, scalar_name: str
) -> Mapping[str, AssessmentValue | AssessmentResult]:
    if isinstance(result, (bool, int, float, str, AssessmentResult)):
        return {scalar_name: result}
    return result


# TODO(DavidM): Add bound=AssessmentValue to the following typevar after we upgrade to pydantic 2.11
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
        spec = AssessmentSpec(call=get_unwrapped_function_name(function))
        return Assessment[InputsT, OutputT, MetadataT](spec=spec, function=function)

    @staticmethod
    def from_registry(
        registry: AssessmentRegistry[InputsT, OutputT, MetadataT], case_name: str, spec: AssessmentSpec
    ) -> Assessment[InputsT, OutputT, MetadataT]:
        function = Assessment[InputsT, OutputT, MetadataT]._validate_against_registry(registry, case_name, spec)
        bound_function = _bind_assessment_function(function, spec.args, spec.kwargs)
        return Assessment[InputsT, OutputT, MetadataT](spec=spec, function=bound_function)

    async def execute(self, ctx: ScoringContext[InputsT, OutputT, MetadataT]) -> list[AssessmentDetail]:
        if inspect.iscoroutinefunction(self.function):
            async_function = cast(AsyncBoundAssessmentFunction[InputsT, OutputT, MetadataT], self.function)
            results = await async_function(ctx)
        else:
            sync_function = cast(SyncBoundAssessmentFunction[InputsT, OutputT, MetadataT], self.function)
            results = await anyio.to_thread.run_sync(sync_function, ctx)

        results = _convert_to_mapping(results, scalar_name=self.spec.call)

        details: list[AssessmentDetail] = []
        for name, result in results.items():
            if not isinstance(result, AssessmentResult):
                result = AssessmentResult(value=result)
            details.append(AssessmentDetail(name=name, value=result.value, reason=result.reason, source=self.spec))

        return details

    @staticmethod
    def _validate_against_registry(
        registry: AssessmentRegistry[InputsT, OutputT, MetadataT], case_name: str, spec: AssessmentSpec
    ) -> AssessmentFunction[InputsT, OutputT, MetadataT]:
        function = registry.get(spec.call)
        if function is None:
            raise ValueError(
                f'Assessment call {spec.call!r} is not registered. Registered choices: {list(registry.keys())}'
            )

        signature = inspect.signature(function)

        scoring_context_param = list(signature.parameters.values())[0]
        type_hints = _typing_extra.get_function_type_hints(function)
        type_hints.pop(scoring_context_param.name)
        type_hints.pop('return', None)
        for p in signature.parameters.values():
            if p.default is not p.empty:
                type_hints[p.name] = NotRequired[type_hints[p.name]]

        td = TypedDict(function.__name__, type_hints)  # type: ignore
        td.__pydantic_config__ = {'extra': 'forbid'}  # type: ignore
        adapter = TypeAdapter(td)

        try:
            # Include `...` as the first argument to account for the scoring_context parameter
            bound_arguments = signature.bind(..., *spec.args, **spec.kwargs).arguments
            bound_arguments.pop(scoring_context_param.name)
        except TypeError as e:
            raise ValueError(f'Assessment {spec.call!r} in case {case_name!r} failed to bind arguments: {e}')

        try:
            adapter.validate_python(bound_arguments)
        except ValidationError as e:
            error_message = str(e)
            error_message = error_message.replace(' for typed-dict', ':')
            raise ValueError(f'Assessment {spec.call!r} in case {case_name!r} has {error_message}')

        return function


_DEFAULT_REGISTRY: dict[str, AssessmentFunction[Any, Any, Any]] = {}

F = TypeVar('F', bound=AssessmentFunction[Any, Any, Any])


def assessment(f: F) -> F:
    """Decorator that registers an assessment function in the default registry."""
    _DEFAULT_REGISTRY[f.__name__] = f
    return f


def get_default_registry() -> AssessmentRegistry:
    """Get the default assessment registry."""
    return _DEFAULT_REGISTRY


def _bind_assessment_function(
    function: AssessmentFunction[InputsT, OutputT, MetadataT], args: list[Any], kwargs: dict[str, Any]
) -> BoundAssessmentFunction[InputsT, OutputT, MetadataT]:
    """Bind spec.args and spec.kwargs to `function` without using functools.partial.

    Returns a function (sync or async) that, when called,
    invokes `function` with spec.args/kwargs already included.
    """
    # If there are no extra arguments to bind, just return the original function
    if not args and not kwargs:
        return function

    # Validate that these arguments can bind to the function.
    # This will raise a TypeError if they don't fit the signature.
    inspect.signature(function).bind(..., *args, **kwargs)

    # Decide which wrapper to return based on whether `function` is async.
    if _is_async(function):

        @wraps(function)
        async def bound_async_function(ctx: ScoringContext[InputsT, OutputT, MetadataT]) -> AssessmentFunctionResult:
            return await function(ctx, *args, **kwargs)

        return bound_async_function
    else:
        assert _is_sync(function)

        @wraps(function)
        def bound_function(ctx: ScoringContext[InputsT, OutputT, MetadataT]) -> AssessmentFunctionResult:
            return function(ctx, *args, **kwargs)

        return bound_function


def _is_async(
    function: AssessmentFunction[InputsT, OutputT, MetadataT],
) -> TypeGuard[AsyncAssessmentFunction[InputsT, OutputT, MetadataT]]:
    return inspect.iscoroutinefunction(function)


def _is_sync(
    function: AssessmentFunction[InputsT, OutputT, MetadataT],
) -> TypeGuard[SyncAssessmentFunction[InputsT, OutputT, MetadataT]]:
    return not inspect.iscoroutinefunction(function)

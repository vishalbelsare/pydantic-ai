import inspect
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Concatenate, Generic, NotRequired

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic._internal import _typing_extra
from typing_extensions import TypedDict, TypeVar

from pydantic_evals.scoring import ScoringContext

InputsT = TypeVar('InputsT', default=dict[str, Any])
OutputT = TypeVar('OutputT', default=dict[str, Any])
MetadataT = TypeVar('MetadataT', default=dict[str, Any])
AssertionFunction = Callable[Concatenate[ScoringContext[InputsT, OutputT, MetadataT], ...], Awaitable[bool]]
AssertionRegistry = Mapping[str, AssertionFunction[InputsT, OutputT, MetadataT]]


@dataclass
class Assertion(Generic[InputsT, OutputT, MetadataT]):
    """An assertion to be run on a scoring context."""

    function: Callable[[ScoringContext[InputsT, OutputT, MetadataT]], Awaitable[bool]]

    async def check(self, ctx: ScoringContext[InputsT, OutputT, MetadataT]) -> bool:
        return await self.function(ctx)


class SerializedAssertion(BaseModel, extra='allow'):
    """A serialized assertion call. Requires a registry to be deserialized."""

    call: str
    """The name of the assertion function to call.

    The registered function should accept a ScoringContext[InputsT, OutputT, MetadataT] as its first argument.
    Any additional required arguments must have values provided in `kwargs`.
    The function should return a boolean indicating whether the assertion passed.

    The call must be registered on the `Evaluation` prior to the call to `run`.
    """

    def __init__(self, call: str, **kwargs: Any):
        super().__init__(call=call, **kwargs)

    def to_assertion(
        self, registry: AssertionRegistry[InputsT, OutputT, MetadataT]
    ) -> Assertion[InputsT, OutputT, MetadataT]:
        # Note: validate_kwargs should have been called prior to this
        function = registry[self.call]
        return Assertion[InputsT, OutputT, MetadataT](function=partial(function, **(self.__pydantic_extra__ or {})))

    def validate_against_registry(self, registry: AssertionRegistry[Any, Any, Any], case_name: str) -> None:
        function = registry.get(self.call)
        if function is None:
            raise ValueError(
                f'Assertion call {self.call!r} is not registered. Registered choices: {list(registry.keys())}'
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
        try:
            adapter.validate_python(self.__pydantic_extra__ or {})
        except ValidationError as e:
            error_message = str(e)
            error_message = error_message.replace(' for typed-dict', ':')
            raise ValueError(f'Assertion {self.call!r} in case {case_name!r} has {error_message}')

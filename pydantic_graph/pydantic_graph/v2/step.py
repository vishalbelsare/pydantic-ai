from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol, Any, TYPE_CHECKING, Generic

from typing_extensions import TypeVar

from pydantic_graph.v2.id_types import NodeId

StateT = TypeVar('StateT', infer_variance=True)
InputT = TypeVar('InputT', infer_variance=True)


class StepContext(Generic[StateT, InputT]):
    """The main reason this is not a dataclass is that we need it to be covariant in its type parameters."""

    if TYPE_CHECKING:

        def __init__(self, state: StateT, inputs: InputT):
            self._state = state
            self._inputs = inputs

        @property
        def state(self) -> StateT:
            return self._state

        @property
        def inputs(self) -> InputT:
            return self._inputs
    else:
        state: StateT
        inputs: InputT

    def __repr__(self):
        return f'{self.__class__.__name__}(inputs={self.inputs})'


if not TYPE_CHECKING:
    StepContext = dataclass(StepContext)


class StepFunction[StateT, InputT, OutputT](Protocol):
    """The purpose of this is to make it possible to deserialize step calls similar to how Evaluators work."""

    def __call__(self, ctx: StepContext[StateT, InputT]) -> Awaitable[OutputT]:
        raise NotImplementedError


AnyStepFunction = StepFunction[Any, Any, Any]


class Step[StateT, InputT, OutputT]:
    """The main reason this is not a dataclass is that we need appropriate variance in the type parameters."""

    def __init__(
        self,
        id: NodeId,
        call: StepFunction[StateT, InputT, OutputT],
        user_label: str | None = None,
        activity: bool = False,
    ):
        self.id = id
        self._call = call
        self.user_label = user_label
        self.activity = activity

    @property
    def call(self) -> StepFunction[StateT, InputT, OutputT]:
        # The use of a property here is necessary to ensure that Step is covariant/contravariant as appropriate.
        return self._call

    @property
    def label(self) -> str | None:
        return self.user_label

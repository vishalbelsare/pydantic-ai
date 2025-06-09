from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pydantic_graph.v2.id_types import GraphRunId


class StateManagerFactory(ABC):
    @abstractmethod
    def get_instance[StateT](
        self, state_type: type[StateT], run_id: GraphRunId, initial_value: StateT
    ) -> StateManager[StateT]:
        raise NotImplementedError


class StateManager[StateT](ABC):
    @abstractmethod
    async def get_immutable_state(self) -> StateT:
        raise NotImplementedError

    @abstractmethod
    @asynccontextmanager
    async def get_mutable_state(self) -> AsyncIterator[StateT]:
        raise NotImplementedError
        yield

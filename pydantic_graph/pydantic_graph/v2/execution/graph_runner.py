from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from pydantic_graph.v2.execution.graph_task import GraphTask
from pydantic_graph.v2.graph import Graph
from pydantic_graph.v2.id_types import JoinId, NodeRunId, TaskId
from pydantic_graph.v2.join import Reducer
from pydantic_graph.v2.util import Maybe


class GraphRunAPI[StateT, DepsT](ABC):
    @abstractmethod
    async def start_task_soon(self, task: GraphTask) -> None:
        """Request a task to be run. This is typically called by the graph runner when it needs to execute a step."""
        raise NotImplementedError

    @abstractmethod
    async def mark_task_completed(self, task_id: TaskId) -> None:
        """Mark a task as completed. This is necessary so that we can track whether a join is ready to proceed."""
        raise NotImplementedError

    @abstractmethod
    async def any_tasks_remain(self) -> bool:
        """Check if there are any tasks remaining to be processed.

        If not, either the result is returned for the graph run, or an error should be raised if there is no result.

        Typically, if you reach the end of a graph run and there are no tasks remaining, it means there was a bug in
        the graph or the graph runner. Either way, this is probably a bug in or shortcoming of the library, but if we
        don't check and eagerly raise an error here it will probably lead to a hang and be difficult/painful to debug.
        """
        raise NotImplementedError

    # State management stuff
    @abstractmethod
    async def initialize_state(self, state: StateT) -> None:
        """Initialize the state for the graph run. This is typically called at the start of a graph run."""
        raise NotImplementedError

    @abstractmethod
    async def get_immutable_state(self) -> StateT:
        raise NotImplementedError

    @abstractmethod
    @asynccontextmanager
    async def get_mutable_state(self) -> AsyncIterator[StateT]:
        raise NotImplementedError
        yield

    @abstractmethod
    @asynccontextmanager
    async def get_reducer(
        self, join_id: JoinId, fork_run_id: NodeRunId
    ) -> AsyncIterator[Reducer[StateT, DepsT, Any, Any] | None]:
        """Get the active reducer state for the given join ID and fork run ID.

        Should lock on fork_run_id to prevent concurrent modifications leading to race conditions.
        """
        raise NotImplementedError
        yield

    @abstractmethod
    async def set_reducer(
        self, join_id: JoinId, fork_run_id: NodeRunId, fork_thread_index: int, reducer: Reducer[StateT, DepsT, Any, Any]
    ) -> None:
        """Update the stored reducer state for the given join ID and fork run ID.

        The fork_thread_index is provided to help ensure idempotency of the reduction.
        """
        raise NotImplementedError

    @abstractmethod
    async def mark_reducer_completed(self, join_id: JoinId, fork_run_id: NodeRunId) -> None:
        """Mark a reducer as completed for the given join ID and fork run ID.

        When this is the case, we can compute the reducer output and drop the reducer state.
        """
        raise NotImplementedError

    @abstractmethod
    async def is_fork_run_completed(self, fork_run_id: NodeRunId) -> bool:
        """Check if there are any tasks remaining that are descendants of the given fork run ID.

        This is used as part of the join logic to determine if its parent fork has completed its execution.
        If so, the corresponding join can proceed to the next step.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_active_reducers_with_fork_run_id(
        self, fork_run_ids: Sequence[NodeRunId]
    ) -> list[tuple[tuple[JoinId, NodeRunId], Reducer[StateT, DepsT, Any, Any]]]:
        """Get all active reducers that are associated with the given fork run IDs.

        This is used during clean-up of a task to check if there are any reducers that may be ready to be completed.
        """
        raise NotImplementedError

    # Graph run result stuff
    @abstractmethod
    async def set_run_result(self, result: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    async def mark_run_finished(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def wait(self) -> Maybe[Any]:
        raise NotImplementedError


type GraphRunResult[StateT, OutputT] = tuple[StateT, OutputT]


class GraphRunner[StateT, DepsT, InputT, OutputT](ABC):
    graph: Graph[StateT, DepsT, InputT, OutputT]

    @abstractmethod
    async def run(
        self,
        state: StateT,
        deps: DepsT,
        inputs: InputT,
    ) -> GraphRunResult[StateT, OutputT]:
        raise NotImplementedError

    # @abstractmethod
    # async def run_soon(
    #     self,
    #     state: StateT,
    #     deps: DepsT,
    #     inputs: InputT,
    # ) -> GraphRunId:
    #     raise NotImplementedError
    #
    # @abstractmethod
    # async def get_result(self, run_id: GraphRunId, clean_up: bool = False) -> Maybe[GraphRunResult[StateT, OutputT]]:
    #     raise NotImplementedError
    #
    # @abstractmethod
    # async def pause(self, run_id: GraphRunId) -> None:
    #     raise NotImplementedError
    #
    # @abstractmethod
    # async def terminate(self, run_id: GraphRunId) -> None:
    #     raise NotImplementedError

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic_graph.v2.id_types import ForkId, NodeId, NodeRunId, TaskId


@dataclass
class GraphTask:
    # With our current BaseNode thing, next_node_id and next_node_inputs are merged into `next_node` itself
    node_id: NodeId
    inputs: Any
    fork_stack: tuple[tuple[ForkId, NodeRunId], ...]
    """
    Stack of forks that have been entered; used so that the GraphRunner can decide when to proceed through joins
    """

    task_id: TaskId = field(default_factory=lambda: TaskId(uuid.uuid4()))

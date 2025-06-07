from __future__ import annotations

import uuid
from typing import NewType

NodeId = NewType('NodeId', str)
NodeRunId = NewType('NodeRunId', str)

JoinId = NewType('JoinId', NodeId)
ForkId = NewType('ForkId', NodeId)

GraphRunId = NewType('GraphRunId', str)
TaskId = NewType('TaskId', str)
ThreadIndex = NewType('ThreadIndex', int)

ForkStack = tuple[tuple[ForkId, NodeRunId, ThreadIndex], ...]

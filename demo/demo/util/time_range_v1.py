from __future__ import annotations as _annotations

import datetime

from pydantic import BaseModel
from typing_extensions import TypedDict


class TimeRangeBuilderSuccess(BaseModel):
    min_timestamp_with_offset: datetime.datetime
    max_timestamp_with_offset: datetime.datetime
    explanation: str | None


class TimeRangeBuilderError(BaseModel):
    error_message: str


TimeRangeResponse = TimeRangeBuilderSuccess | TimeRangeBuilderError


class TimeRangeInputs(TypedDict):
    prompt: str
    now: datetime.datetime

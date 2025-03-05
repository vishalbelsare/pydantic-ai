from __future__ import annotations as _annotations

import datetime

from fastapi import FastAPI
from pydantic import BaseModel


class TimeRangeBuilderSuccess(BaseModel):
    min_timestamp: datetime.datetime
    max_timestamp: datetime.datetime
    explanation: str | None


class TimeRangeBuilderError(BaseModel):
    error_message: str


TimeRangeResponse = TimeRangeBuilderSuccess | TimeRangeBuilderError
app = FastAPI()


@app.get("/infer-time-range")
async def infer_time_range(prompt: str) -> TimeRangeResponse:
    return TimeRangeBuilderSuccess(
        min_timestamp=datetime.datetime(2025, 3, 6),
        max_timestamp=datetime.datetime(2025, 3, 7),
        explanation="Today is a good day",
    )


if __name__ == "__main__":
    import uvicorn

    print("Swagger UI Link: http://localhost:8099/docs")
    uvicorn.run(app, port=8099)

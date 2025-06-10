from __future__ import annotations as _annotations

from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel


class TimeRangeBuilderSuccess(BaseModel):
    min_timestamp_with_offset: datetime
    max_timestamp_with_offset: datetime
    explanation: str | None


class TimeRangeBuilderError(BaseModel):
    error_message: str


TimeRangeResponse = TimeRangeBuilderSuccess | TimeRangeBuilderError
app = FastAPI()


@app.get("/infer-time-range")
async def infer_time_range(prompt: str) -> TimeRangeResponse:
    return TimeRangeBuilderSuccess(
        min_timestamp_with_offset=datetime(2025, 3, 6),
        max_timestamp_with_offset=datetime(2025, 3, 7),
        explanation="Hello world",
    )


if __name__ == "__main__":
    import uvicorn

    print("Swagger UI Link: http://localhost:8099/docs")
    uvicorn.run(app, port=8099)

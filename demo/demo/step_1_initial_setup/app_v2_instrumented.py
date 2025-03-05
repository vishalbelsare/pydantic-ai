from __future__ import annotations as _annotations

import datetime

import logfire
from fastapi import FastAPI
from pydantic import BaseModel

logfire.configure(send_to_logfire="if-token-present")


class TimeRangeBuilderSuccess(BaseModel, use_attribute_docstrings=True):
    min_timestamp: datetime.datetime
    max_timestamp: datetime.datetime
    explanation: str | None


class TimeRangeBuilderError(BaseModel):
    error_message: str


TimeRangeResponse = TimeRangeBuilderSuccess | TimeRangeBuilderError
app = FastAPI()


@app.get("/infer-time-range")
async def infer_time_range(prompt: str) -> TimeRangeResponse:
    # logfire.info(f'Handling {prompt=}')
    return TimeRangeBuilderSuccess(
        min_timestamp=datetime.datetime(2025, 3, 6),
        max_timestamp=datetime.datetime(2025, 3, 7),
        explanation="Today is a good day",
    )


if __name__ == "__main__":
    import uvicorn

    print("Swagger UI Link: http://localhost:8099/docs")
    logfire.instrument_fastapi(app, capture_headers=True, record_send_receive=True)
    uvicorn.run(app, port=8099)

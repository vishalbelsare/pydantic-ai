from __future__ import annotations as _annotations

import datetime

import logfire
from fastapi import FastAPI

from demo.models.time_range_v1 import TimeRangeBuilderSuccess, TimeRangeResponse

logfire.configure(
    send_to_logfire="if-token-present",
    environment="prod",
    service_name="app",
    service_version="v3",
)

app = FastAPI()


@app.get("/infer-time-range")
async def infer_time_range(prompt: str) -> TimeRangeResponse:
    logfire.info(f"Handling {prompt=}")
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

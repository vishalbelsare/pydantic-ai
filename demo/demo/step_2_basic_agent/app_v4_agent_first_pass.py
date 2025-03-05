from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from datetime import datetime

import logfire
from fastapi import FastAPI
from pydantic_ai import Agent, RunContext

from demo.models.time_range_v1 import TimeRangeResponse

logfire.configure(
    send_to_logfire="if-token-present",
    environment="prod",
    service_name="app",
    service_version="v4",
)


@dataclass
class TimeRangeDeps:
    now: datetime = field(default_factory=lambda: datetime.now().astimezone())


def time_range_system_prompt(ctx: RunContext[TimeRangeDeps]):
    now_str = ctx.deps.now.strftime(
        "%A, %B %d, %Y %H:%M:%S %Z"
    )  # Format like: Friday, November 22, 2024 11:15:14 PST
    return f"Convert the user's request into a structured time range. The user's current time is {now_str}."


time_range_agent = Agent[TimeRangeDeps, TimeRangeResponse](
    "gpt-4o",
    result_type=TimeRangeResponse,  # type: ignore  # we can't yet annotate something as receiving a TypeForm
    deps_type=TimeRangeDeps,
    retries=1,
    instrument=True,
)
time_range_agent.system_prompt(time_range_system_prompt)

app = FastAPI()


@app.get("/infer-time-range")
async def infer_time_range(prompt: str) -> TimeRangeResponse:
    logfire.info(f"Handling {prompt=}")
    agent_run_result = await time_range_agent.run(prompt, deps=TimeRangeDeps())
    return agent_run_result.data


if __name__ == "__main__":
    logfire.instrument_fastapi(app, capture_headers=True, record_send_receive=True)

    import uvicorn

    print("Swagger UI Link: http://localhost:8099/docs")
    uvicorn.run(app, port=8099)

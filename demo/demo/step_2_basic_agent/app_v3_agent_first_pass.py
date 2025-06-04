from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from datetime import datetime

import logfire
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from demo.util.tokens import get_app_write_token

token = get_app_write_token()
logfire.configure(
    token=token,
    environment="prod",
    service_name="app",
    service_version="v3",
    advanced=logfire.AdvancedOptions(base_url="http://localhost:8000"),
)


class TimeRangeBuilderSuccess(BaseModel):
    min_timestamp: datetime
    max_timestamp: datetime
    explanation: str | None


class TimeRangeBuilderError(BaseModel):
    error_message: str


TimeRangeResponse = TimeRangeBuilderSuccess | TimeRangeBuilderError


@dataclass
class TimeRangeDeps:
    now: datetime = field(default_factory=lambda: datetime.now().astimezone())


time_range_agent = Agent[TimeRangeDeps, TimeRangeResponse](
    "gpt-4o",
    result_type=TimeRangeResponse,  # type: ignore  # we can't yet annotate something as receiving a TypeForm
    deps_type=TimeRangeDeps,
    system_prompt="Convert the user's request into a structured time range.",
    retries=1,
    instrument=True,
)


@time_range_agent.tool
def get_current_time(ctx: RunContext[TimeRangeDeps]) -> str:
    now_str = ctx.deps.now.strftime(
        "%A, %B %d, %Y %H:%M:%S %Z"
    )  # Format like: Friday, November 22, 2024 11:15:14 PST
    return f"The user's current time is {now_str}."


app = FastAPI()


@app.get("/infer-time-range")
async def infer_time_range(prompt: str) -> TimeRangeResponse:
    logfire.info(f"Handling {prompt=}")
    agent_run_result = await time_range_agent.run(prompt, deps=TimeRangeDeps())
    return agent_run_result.output


if __name__ == "__main__":
    import uvicorn

    print("Swagger UI Link: http://localhost:8099/docs")
    logfire.instrument_fastapi(app, capture_headers=True, record_send_receive=True)
    uvicorn.run(app, port=8099)

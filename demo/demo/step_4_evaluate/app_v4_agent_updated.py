from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from datetime import datetime
from textwrap import dedent

import logfire
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from typing_extensions import TypedDict

from demo.util.tokens import get_app_write_token

token = get_app_write_token()
logfire.configure(
    token=token,
    environment="prod",
    service_name="app",
    service_version="v4",
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
    retries=1,
    instrument=True,
)


@time_range_agent.system_prompt
def time_range_system_prompt(ctx: RunContext[TimeRangeDeps]):
    now_str = ctx.deps.now.strftime(
        "%A, %B %d, %Y %H:%M:%S %Z"
    )  # Format like: Friday, November 22, 2024 11:15:14 PST
    return dedent(
        f"""
        Convert the user's request into a structured time range. Both the min and max must have a timezone offset specified.
        If the user does not request a specific timezone, use the offset from their local time (provided below).
        If ambiguous, prefer to select a time range in the (recent) past.

        If the user's request is too ambiguous or cannot be converted into a time range, return an error message.
        If the user mentions a specific point in time, select a 10 minute window around that time.

        In the explanation field, include a brief message addressed to the user in passive voice indicating how the time range has been updated.
        If you had to interpret anything possibly-ambiguous in the user's request, please address that in your response.
        In the explanation, **DO NOT** repeat anything about the user's request that was unambiguous.

        Examples:
        - If the user says "yesterday", you should say "Selected midnight to midnight yesterday."
        - If the user says "the last 24 hours", you should say "The time range has been updated."
        - If the user says "next week", you might say "Selected a 7-day window starting now."
        - If the user says "around noon Tokyo time" you might say "Selected a 10-minute window around 12PM JST."

        The user's local time is {now_str}.
        """
    )


app = FastAPI()


@app.get("/infer-time-range")
async def infer_time_range(prompt: str) -> TimeRangeResponse:
    logfire.info(f"Handling {prompt=}")
    return await run_infer_time_range(
        {"prompt": prompt, "now": datetime.now().astimezone()}
    )


class TimeRangeInputs(TypedDict):
    prompt: str
    now: datetime


async def run_infer_time_range(inputs: TimeRangeInputs) -> TimeRangeResponse:
    """Infer a time range from a user prompt."""
    deps = TimeRangeDeps(now=inputs["now"])
    return (await time_range_agent.run(inputs["prompt"], deps=deps)).data


if __name__ == "__main__":
    logfire.instrument_fastapi(app, capture_headers=True, record_send_receive=True)

    import uvicorn

    print("Swagger UI Link: http://localhost:8099/docs")
    uvicorn.run(app, port=8099)

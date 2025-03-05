from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from datetime import datetime
from textwrap import dedent

import logfire
from fastapi import FastAPI
from pydantic_ai import Agent, RunContext
from pydantic_evals.demo.time_range.models import TimeRangeResponse

logfire.configure(
    send_to_logfire="if-token-present",
    environment="prod",
    service_name="app",
    service_version="v6",
)


@dataclass
class TimeRangeDeps:
    """Dependencies for the time range inference agent."""

    now: datetime = field(default_factory=lambda: datetime.now().astimezone())


def time_range_system_prompt(ctx: RunContext[TimeRangeDeps]):
    """Build the system prompt for the time range inference agent."""
    # Format like: Friday, November 22, 2024 11:15:14 PST
    now_str = ctx.deps.now.strftime("%A, %B %d, %Y %H:%M:%S %Z")
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

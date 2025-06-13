# /// script
# dependencies = [
#   "temporalio",
#   "logfire",
# ]
# ///
import asyncio
import random
from datetime import timedelta

from temporalio import workflow
from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.runtime import OpenTelemetryConfig, Runtime, TelemetryConfig
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from pydantic_ai import Agent
    from pydantic_ai.models.temporal import (
        TemporalModel,
        TemporalSettings,
        initialize_temporal,
    )

    initialize_temporal()

    # import logfire
    # logfire.configure(send_to_logfire=False)
    # logfire.instrument_pydantic_ai()
    model = TemporalModel(
        'openai:gpt-4o',
        options=TemporalSettings(start_to_close_timeout=timedelta(seconds=60)),
    )
    my_agent = Agent(model=model, instructions='be helpful')


# Basic workflow that logs and invokes an activity
@workflow.defn
class MyAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await my_agent.run(prompt)).output


def init_runtime_with_telemetry() -> Runtime:
    import logfire

    logfire.configure(send_to_logfire=True, service_version='0.0.1')
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)

    # Setup SDK metrics to OTel endpoint
    return Runtime(
        telemetry=TelemetryConfig(
            metrics=OpenTelemetryConfig(url='http://localhost:4318')
        )
    )


async def main():
    client = await Client.connect(
        'localhost:7233',
        interceptors=[TracingInterceptor()],
        data_converter=pydantic_data_converter,
        runtime=init_runtime_with_telemetry(),
    )

    async with Worker(
        client,
        task_queue='my-agent-task-queue',
        workflows=[MyAgentWorkflow],
        activities=[model.request_activity],
    ):
        result = await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
            MyAgentWorkflow.run,
            'what is 2+3?',
            id=f'my-agent-workflow-id-{random.random()}',
            task_queue='my-agent-task-queue',
        )
        print(f'Result: {result!r}')


if __name__ == '__main__':
    asyncio.run(main())

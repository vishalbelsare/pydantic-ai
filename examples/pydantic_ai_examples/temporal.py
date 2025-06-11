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
from temporalio.contrib.pydantic import pydantic_data_converter
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
    agent = Agent(model=model, instructions='be helpful')


# Basic workflow that logs and invokes an activity
@workflow.defn
class MyAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await agent.run(prompt)).output


async def main():
    client = await Client.connect(
        'localhost:7233', data_converter=pydantic_data_converter
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

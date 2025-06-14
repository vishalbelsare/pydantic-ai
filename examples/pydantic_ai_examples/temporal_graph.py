import os

os.environ['PYDANTIC_DISABLE_PLUGINS'] = 'true'
import asyncio
import random
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from types import NoneType
from typing import Any, Literal

from temporalio import workflow, activity
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker


with workflow.unsafe.imports_passed_through():
    # from temporalio.contrib.pydantic import pydantic_data_converter

    from pydantic_graph.v2.execution.graph_walker import GraphRunner
    from pydantic_graph.v2.graph import GraphBuilder
    from pydantic_graph.v2.join import NullReducer
    from pydantic_graph.v2.step import StepContext
    from pydantic_graph.v2.util import TypeExpression


@dataclass
class MyContainer[T]:
    field_1: T | None
    field_2: T | None
    field_3: list[T] | None


@dataclass
class MyState:
    type_name: str | None
    container: MyContainer[Any] | None

    def with_workflow(self, workflow: 'MyWorkflow'):
        return MyStateWithWorkflow(
            type_name=self.type_name, container=self.container, workflow=workflow
        )


# If you want access to the workflow inside the nodes, add it as a field onto the state type for the graph
@dataclass
class MyStateWithWorkflow(MyState):
    workflow: 'MyWorkflow'

    def without_workflow(self) -> MyState:
        return MyState(type_name=self.type_name, container=self.container)


g = GraphBuilder(
    state_type=MyStateWithWorkflow, input_type=NoneType, output_type=NoneType
)


@activity.defn
async def get_random_number() -> float:
    return random.random()


@g.step
async def choose_type(
    ctx: StepContext[MyStateWithWorkflow, object],
) -> Literal['int', 'str']:
    random_number = await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
        get_random_number, start_to_close_timeout=timedelta(seconds=1)
    )
    chosen_type = int if random_number < 0.5 else str
    ctx.state.type_name = chosen_type.__name__
    ctx.state.container = MyContainer(field_1=None, field_2=None, field_3=None)
    return 'int' if chosen_type is int else 'str'


@g.step
async def handle_int(ctx: StepContext[object, object]) -> None:
    pass


@g.step
async def handle_str(ctx: StepContext[object, object]) -> None:
    pass


@g.step
async def handle_int_1(ctx: StepContext[MyStateWithWorkflow, object]) -> None:
    print('start int 1')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    ctx.state.container.field_1 = 1
    print('end int 1')


@g.step
async def handle_int_2(ctx: StepContext[MyStateWithWorkflow, object]) -> None:
    print('start int 2')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    ctx.state.container.field_2 = 1
    print('end int 2')


@g.step
async def handle_int_3(
    ctx: StepContext[MyStateWithWorkflow, object],
) -> list[int]:
    print('start int 3')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    output = ctx.state.container.field_3 = [1, 2, 3]
    print('end int 3')
    return output


@g.step
async def handle_str_1(ctx: StepContext[MyStateWithWorkflow, object]) -> None:
    print('start str 1')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    ctx.state.container.field_1 = 1
    print('end str 1')


@g.step
async def handle_str_2(ctx: StepContext[MyStateWithWorkflow, object]) -> None:
    print('start str 2')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    ctx.state.container.field_2 = 1
    print('end str 2')


@g.step
async def handle_str_3(
    ctx: StepContext[MyStateWithWorkflow, object],
) -> Iterable[str]:
    print('start str 3')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    output = ctx.state.container.field_3 = ['a', 'b', 'c']
    print('end str 3')
    return output


@g.step(node_id='handle_field_3_item')
async def handle_field_3_item(ctx: StepContext[MyStateWithWorkflow, int | str]) -> None:
    inputs = ctx.inputs
    print(f'handle_field_3_item: {inputs}')
    await asyncio.sleep(0.25)
    assert ctx.state.container is not None
    assert ctx.state.container.field_3 is not None
    ctx.state.container.field_3.append(inputs * 2)
    await asyncio.sleep(0.25)


handle_join = g.join(NullReducer, node_id='handle_join')

g.add(
    g.edge_from(g.start_node).label('begin').to(choose_type),
    g.edge_from(choose_type).to(
        g.decision()
        .branch(g.match(TypeExpression[Literal['str']]).to(handle_str))
        .branch(g.match(TypeExpression[Literal['int']]).to(handle_int))
    ),
    g.edge_from(handle_int).to(handle_int_1, handle_int_2, handle_int_3),
    g.edge_from(handle_str).to(
        lambda e: [
            e.label('abc').to(handle_str_1),
            e.label('def').to(handle_str_2),
            e.to(handle_str_3),
        ]
    ),
    g.edge_from(handle_int_3).spread().to(handle_field_3_item),
    g.edge_from(handle_str_3).spread().to(handle_field_3_item),
    g.edge_from(
        handle_int_1, handle_int_2, handle_str_1, handle_str_2, handle_field_3_item
    ).to(handle_join),
    g.edge_from(handle_join).to(g.end_node),
)

graph = g.build()
print(graph)
print('----------')


@workflow.defn
class MyWorkflow:
    @workflow.run
    async def run(self, state: MyState) -> MyState:
        runner = GraphRunner(graph)
        final_state, _ = await runner.run(
            state=state.with_workflow(self),
            inputs=None,
        )
        return final_state.without_workflow()


async def main_temporal():
    client = await Client.connect(
        'localhost:7233',
        data_converter=pydantic_data_converter,
    )

    async with Worker(
        client,
        task_queue='my-task-queue',
        workflows=[MyWorkflow],
        activities=[get_random_number],
    ):
        result = await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
            MyWorkflow.run,
            MyState(type_name=None, container=None),
            id=f'my-workflow-id-{random.random()}',
            task_queue='my-task-queue',
        )
        print(f'Result: {result!r}')


if __name__ == '__main__':
    # asyncio.run(main())
    asyncio.run(main_temporal())

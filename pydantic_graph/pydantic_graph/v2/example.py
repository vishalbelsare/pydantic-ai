import asyncio
import random
from collections.abc import Iterable
from dataclasses import dataclass
from types import NoneType
from typing import Any, Literal

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
    type_name: str | None = None
    container: MyContainer[Any] | None = None


g = GraphBuilder(state_type=MyState, input_type=NoneType, output_type=NoneType)


@g.step
async def choose_type(ctx: StepContext[MyState, object]) -> Literal['int', 'str']:
    chosen_type = int if random.random() < 0.5 else str
    state = ctx.state
    state.type_name = chosen_type.__name__
    state.container = MyContainer(field_1=None, field_2=None, field_3=None)
    return 'int' if chosen_type is int else 'str'


@g.step
async def handle_int(ctx: StepContext[object, object]) -> None:
    pass


@g.step
async def handle_str(ctx: StepContext[object, object]) -> None:
    pass


@g.step
async def handle_int_1(ctx: StepContext[MyState, object]) -> None:
    print('start int 1')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    ctx.state.container.field_1 = 1
    print('end int 1')


@g.step
async def handle_int_2(ctx: StepContext[MyState, object]) -> None:
    print('start int 2')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    ctx.state.container.field_2 = 1
    print('end int 2')


@g.step
async def handle_int_3(
    ctx: StepContext[MyState, object],
) -> list[int]:
    print('start int 3')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    output = ctx.state.container.field_3 = [1, 2, 3]
    print('end int 3')
    return output


@g.step
async def handle_str_1(ctx: StepContext[MyState, object]) -> None:
    print('start str 1')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    ctx.state.container.field_1 = 1
    print('end str 1')


@g.step
async def handle_str_2(ctx: StepContext[MyState, object]) -> None:
    print('start str 2')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    ctx.state.container.field_2 = 1
    print('end str 2')


@g.step
async def handle_str_3(
    ctx: StepContext[MyState, object],
) -> Iterable[str]:
    print('start str 3')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    output = ctx.state.container.field_3 = ['a', 'b', 'c']
    print('end str 3')
    return output


@g.step(node_id='handle_field_3_item')
async def handle_field_3_item(ctx: StepContext[MyState, int | str]) -> None:
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
        lambda e: [e.label('abc').to(handle_str_1), e.label('def').to(handle_str_2), e.to(handle_str_3)]
    ),
    g.edge_from(handle_int_3).spread().to(handle_field_3_item),
    g.edge_from(handle_str_3).spread().to(handle_field_3_item),
    g.edge_from(handle_int_1, handle_int_2, handle_str_1, handle_str_2, handle_field_3_item).to(handle_join),
    g.edge_from(handle_join).to(g.end_node),
)

graph = g.build()
print(graph)
print('----------')


async def main():
    runner = GraphRunner(graph)
    state, result = await runner.run(state=MyState(), inputs=None)

    print('-')
    print(f'{result=}')
    print(f'{state=}')


if __name__ == '__main__':
    asyncio.run(main())

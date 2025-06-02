import random
from dataclasses import dataclass
from types import NoneType
from typing import Any, Literal

from pydantic_graph.v2.execution.in_memory import InMemoryGraphRunner
from pydantic_graph.v2.graph import GraphBuilder, TypeExpression
from pydantic_graph.v2.join import reduce_to_none
from pydantic_graph.v2.step import StepContext


@dataclass
class MyContainer[T]:
    field_1: T | None
    field_2: T | None
    field_3: list[T] | None


@dataclass
class MyState:
    type_name: str | None = None
    container: MyContainer[Any] | None = None


gb = GraphBuilder(state_type=MyState, deps_type=NoneType, input_type=NoneType, output_type=NoneType)


@gb.step
async def choose_type(ctx: StepContext[MyState, object, object]) -> Literal['int', 'str']:
    chosen_type = int if random.random() < 0.5 else str
    async with ctx.get_mutable_state() as state:
        state.type_name = chosen_type.__name__
        state.container = MyContainer(field_1=None, field_2=None, field_3=None)
    return 'int' if chosen_type is int else 'str'


@gb.step
async def handle_int(ctx: StepContext[object, object, object]) -> None:
    pass


@gb.step
async def handle_str(ctx: StepContext[object, object, object]) -> None:
    pass


@gb.step
async def handle_int_1(ctx: StepContext[MyState, object, object]) -> None:
    print('start int 1')
    await asyncio.sleep(1)
    async with ctx.get_mutable_state() as state:
        assert state.container is not None
        state.container.field_1 = 1
    print('end int 1')


@gb.step
async def handle_int_2(ctx: StepContext[MyState, object, object]) -> None:
    print('start int 2')
    await asyncio.sleep(1)
    async with ctx.get_mutable_state() as state:
        assert state.container is not None
        state.container.field_2 = 1
    print('end int 2')


@gb.step
async def handle_int_3(ctx: StepContext[MyState, object, object]) -> list[int]:
    print('start int 3')
    await asyncio.sleep(1)
    async with ctx.get_mutable_state() as state:
        assert state.container is not None
        output = state.container.field_3 = [1, 2, 3]
    print('end int 3')
    return output


@gb.step
async def handle_str_1(ctx: StepContext[MyState, object, object]) -> None:
    print('start str 1')
    await asyncio.sleep(1)
    async with ctx.get_mutable_state() as state:
        assert state.container is not None
        state.container.field_1 = 1
    print('end str 1')


@gb.step
async def handle_str_2(ctx: StepContext[MyState, object, object]) -> None:
    print('start str 2')
    await asyncio.sleep(1)
    async with ctx.get_mutable_state() as state:
        assert state.container is not None
        state.container.field_2 = 1
    print('end str 2')


@gb.step
async def handle_str_3(ctx: StepContext[MyState, object, object]) -> list[str]:
    print('start str 3')
    await asyncio.sleep(1)
    async with ctx.get_mutable_state() as state:
        assert state.container is not None
        output = state.container.field_3 = ['a', 'b', 'c']
    print('end str 3')
    return output


async def handle_field_3_item(ctx: StepContext[MyState, object, int | str]) -> None:
    inputs = ctx.inputs
    print(f'handle_field_3_item: {inputs}')
    await asyncio.sleep(0.25)
    async with ctx.get_mutable_state() as state:
        assert state.container is not None
        assert state.container.field_3 is not None
        state.container.field_3.append(inputs * 2)
    await asyncio.sleep(0.25)


handle_field_3_item_int = gb.step(handle_field_3_item, node_id='handle_field_3_item_int')
handle_field_3_item_str = gb.step(handle_field_3_item, node_id='handle_field_3_item_str')

handle_int_join = gb.join(reduce_to_none, node_id='handle_int_join')
handle_str_join = gb.join(reduce_to_none, node_id='handle_str_join')

gb.add_edges(gb.from_(gb.start_node).label('begin').to(choose_type))
gb.add_decision(
    choose_type,
    gb.branch(TypeExpression[Literal['int']]).to(handle_int).branch(TypeExpression[Literal['str']]).to(handle_str),
)
gb.add_edges(
    # ints
    gb.from_(handle_int).to(handle_int_1, handle_int_2, handle_int_3),
    # gb.from_(handle_int).fork(
    #     lambda e: [e.to(handle_int_1), e.to(handle_int_2), e.to(handle_int_3)]
    # ),
    gb.from_(handle_int_3).spread().to(handle_field_3_item_int),
    gb.from_(handle_int_1, handle_int_2, handle_field_3_item_int).to(handle_int_join),
    gb.from_(handle_int_join).to(gb.end_node),
    # strs
    gb.from_(handle_str).fork(lambda e: [e.to(handle_str_1), e.to(handle_str_2), e.to(handle_str_3)]),
    gb.from_(handle_str_3).spread().to(handle_field_3_item_str),
    gb.from_(handle_str_1, handle_str_2, handle_field_3_item_str).to(handle_str_join),
    gb.from_(handle_str_join).to(gb.end_node),
)

g = gb.build()
print(g)
print('-')


async def main():
    executor = InMemoryGraphRunner(g)
    state, result = await executor.run(state=MyState(), deps=None, inputs=None)

    print('-')
    print(f'{result=}')
    print(f'{state=}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())

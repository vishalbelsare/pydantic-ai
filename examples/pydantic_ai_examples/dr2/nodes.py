from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, cast, get_args, get_origin, overload

from pydantic import TypeAdapter
from pydantic_core import to_json
from pydantic_graph.v2.id_types import NodeId
from pydantic_graph.v2.step import StepContext
from pydantic_graph.v2.util import TypeExpression

from pydantic_ai import Agent, models


@dataclass(init=False)
class Prompt[InputT, OutputT]:
    input_type: type[InputT]
    output_type: type[Any]
    output_selector: Callable[[InputT, Any], OutputT] | None
    prompt: str
    model: models.Model | models.KnownModelName | str = 'openai:gpt-4o'

    @overload
    def __init__(
        self,
        *,
        input_type: type[InputT],
        output_type: type[TypeExpression[OutputT]] | type[OutputT],
        prompt: str,
        model: models.Model | models.KnownModelName | str = 'openai:gpt-4o',
    ) -> None: ...
    @overload
    def __init__[IntermediateT](
        self,
        *,
        input_type: type[InputT],
        output_type: type[TypeExpression[IntermediateT]] | type[IntermediateT],
        output_transform: Callable[[InputT, IntermediateT], OutputT],
        prompt: str,
        model: models.Model | models.KnownModelName | str = 'openai:gpt-4o',
    ) -> None: ...
    def __init__(
        self,
        *,
        input_type: type[InputT],
        output_type: type[Any],
        output_transform: Callable[[InputT, Any], OutputT] | None = None,
        prompt: str,
        model: models.Model | models.KnownModelName | str = 'openai:gpt-4o',
    ):
        self.input_type = input_type
        self.output_type = output_type
        self.output_transform = output_transform
        self.prompt = prompt
        self.model = model

    @cached_property
    def agent(self) -> Agent[None, OutputT]:
        input_json_schema = to_json(
            TypeAdapter(self.input_type).json_schema(), indent=2
        ).decode()
        instructions = '\n'.join(
            [
                'You will receive messages matching the following JSON schema:',
                input_json_schema,
                '',
                'Generate output based on the following instructions:',
                self.prompt,
            ]
        )
        output_type = self.output_type
        if get_origin(output_type) is TypeExpression:
            output_type = get_args(self.output_type)[0]
        return Agent(
            model=self.model,
            output_type=cast(type[OutputT], output_type),
            instructions=instructions,
        )

    async def __call__(self, ctx: StepContext[Any, Any, InputT]) -> OutputT:
        result = self.agent.run_sync(to_json(ctx.inputs, indent=2).decode())
        output = result.output
        if self.output_transform:
            output = self.output_transform(ctx.inputs, output)
        return output


@dataclass
class Interruption[StopT, ResumeT]:
    value: StopT
    next_node: (
        NodeId  # This is the node this walk should resume from after the interruption
    )
    graph_state: Any = None  # TODO: Need a way to pass the graph state ...?

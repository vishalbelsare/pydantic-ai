"""PlanOutline subgraph.

state PlanOutline {
    [*]
    ClarifyRequest: Clarify user request & scope
    HumanFeedback: Human provides clarifications
    GenerateOutline: Draft initial outline
    ReviewOutline: Supervisor reviews outline

    [*] --> ClarifyRequest
    ClarifyRequest --> HumanFeedback: need more info
    HumanFeedback --> ClarifyRequest
    ClarifyRequest --> GenerateOutline: ready
    GenerateOutline --> ReviewOutline
    ReviewOutline --> GenerateOutline: revise
    ReviewOutline --> [*]: approve
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from .graph import Graph, Interruption, TransformContext, graph_node
from .nodes import Prompt, TypeUnion
from .shared_types import MessageHistory, Outline

# from .graph import Routing, GraphBuilder


# Types
## State
@dataclass
class State:
    chat: MessageHistory
    outline: Outline | None


## handle_user_message
class Clarify(BaseModel):
    """Ask some questions to clarify the user request."""

    choice: Literal['clarify']
    message: str


class Refuse(BaseModel):
    """Use this if you should not do research.

    This is the right choice if the user didn't ask for research, or if the user did but there was a safety concern.
    """

    choice: Literal['refuse']
    message: str  # message to show user


class Proceed(BaseModel):
    """There is enough information to proceed with handling the user's request."""

    choice: Literal['proceed']


## generate_outline
class ExistingOutlineFeedback(BaseModel):
    outline: Outline
    feedback: str


class GenerateOutlineInputs(BaseModel):
    chat: MessageHistory
    feedback: ExistingOutlineFeedback | None


## review_outline
class ReviewOutlineInputs(BaseModel):
    chat: MessageHistory
    outline: Outline


class ReviseOutline(BaseModel):
    choice: Literal['revise']
    details: str


class ApproveOutline(BaseModel):
    choice: Literal['approve']
    message: str  # message to user describing the research you are going to do


class OutlineStageOutput(BaseModel):
    """Use this if you have enough information to proceed."""

    outline: Outline  # outline of the research
    message: str  # message to show user before beginning research


# Node types
@dataclass
class YieldToHuman:
    message: str




def transform_proceed(ctx: TransformContext[object, MessageHistory, object]):
    return GenerateOutlineInputs(chat=ctx.inputs, feedback=None)


def transform_clarify(ctx: TransformContext[object, object, Clarify]):
    return Interruption(YieldToHuman(ctx.output.message), handle_user_message)


def transform_outline(ctx: TransformContext[State, object, Outline]):
    return ReviewOutlineInputs(chat=ctx.state.chat, outline=ctx.output)


def transform_revise_outline(
    ctx: TransformContext[State, ReviewOutlineInputs, ReviseOutline],
):
    return GenerateOutlineInputs(
        chat=ctx.state.chat,
        feedback=ExistingOutlineFeedback(
            outline=ctx.inputs.outline, feedback=ctx.output.details
        ),
    )


def transform_approve_outline(
    ctx: TransformContext[object, ReviewOutlineInputs, ApproveOutline],
):
    return OutlineStageOutput(outline=ctx.inputs.outline, message=ctx.output.message)


# Graph
g = Graph.builder(
    state_type=State,
    input_type=MessageHistory,
    output_type=TypeUnion[
        Refuse | OutlineStageOutput | Interruption[YieldToHuman, MessageHistory]
    ],
    # start_at=handle_user_message,
)

# TODO: If we drop inputs from the transform functions, we'll need a way to decide what to pass to the prompt node to only include some of the inputs...
#   Is there any typesafe way to generate a template in JSON schema?


# Graph nodes
# TODO: Need to make it easier to change the model; should be a parameter to the graph
@graph_node
async def handle_user_message(inputs: MessageHistory) -> Refuse | Clarify | Proceed:
    result = await Prompt(
        input_type=MessageHistory,
        output_type=Refuse | Clarify | Proceed,  # pyright: ignore['abc']
        prompt='Decide how to proceed from user message',
    ).run(None, inputs)
    return result

@graph_node
async def generate_outline(inputs: GenerateOutlineInputs) -> Outline:
    result = await Prompt(
        input_type=GenerateOutlineInputs,
        output_type=Outline,
        prompt='Generate the outline',
    ).run(None, inputs)
    return result

@graph_node
async def review_outline(inputs: ReviewOutlineInputs) -> ReviseOutline | ApproveOutline:
    result = await Prompt(
        input_type=ReviewOutlineInputs,
        output_type=ReviseOutline | ApproveOutline,
        prompt='Review the outline',
    ).run(None, inputs)
    return result


g.add_edges(
    handle := g.start_edge(handle_user_message),
    g.edges()
    .branch(handle(Refuse).end())
    .branch(handle(Proceed).transform(transform_proceed).route_to(generate_outline))
    .branch(handle(Clarify).transform(transform_clarify).end()),
)


# g.edges(
#     handle_user_message,
#     lambda h: Routing[
#         h(Refuse).end()
#         | h(Proceed).transform(transform_proceed).route_to(generate_outline)
#         | h(Clarify).transform(transform_clarify).end()
#     ],
# )
# g.edges(
#     generate_outline,
#     lambda h: Routing[h(Outline).transform(transform_outline).route_to(review_outline)],
# )
# g.edges(
#     review_outline,
#     lambda h: Routing[
#         h(ReviseOutline).transform(transform_revise_outline).route_to(generate_outline)
#         | h(ApproveOutline).transform(transform_approve_outline).end()
#     ],
# )


# class Route[SourceT, EndT]:
#     _force_source_invariant: Callable[[SourceT], SourceT]
#     _force_end_covariant: Callable[[], EndT]
#
#     def case[S, E, S2, E2](
#         self: Route[S, E], route: Route[S2, E2]
#     ) -> Route[S | S2, E | E2]:
#         raise NotImplementedError
#
#
# class Case[SourceT, OutT]:
#     def _execute(self, source: SourceT) -> OutT:
#         raise NotImplementedError
#
#     def transform[T](
#         self, transform_fn: Callable[[TransformContext[Any, Any, OutT]], T]
#     ) -> Case[SourceT, T]:
#         raise NotImplementedError
#
#     def route_to(self, node: Node[Any, OutT, Any]) -> Route[SourceT, Never]:
#         raise NotImplementedError
#
#     def end(self: Case[SourceT, OutT]) -> Route[SourceT, OutT]:
#         raise NotImplementedError
#
#
# def handle[SourceT](source: type[SourceT]) -> Case[SourceT, SourceT]:
#     raise NotImplementedError
#
#
# def cases() -> Route[Never, Never]:
#     raise NotImplementedError
#
#
# def add_edges[GraphOutputT, NodeOutputT](
#     g: GraphBuilder[Any, Any, GraphOutputT],
#     n: Node[Any, Any, NodeOutputT],
#     c: Route[NodeOutputT, GraphOutputT],
# ):
#     raise NotImplementedError
#
#
# # reveal_type(approve_pipe)
# # edges = cases(
# #     revise_pipe,
# #     approve_pipe
# # )
# # add_edges(g, review_outline, edges)
# # cases_ = cases().case(approve_pipe)#.case(revise_pipe)
# # add_edges(g, review_outline, cases_)
#
# # Things that need to emit type errors:
# # * Routing an incompatible output into a transform
# # * Routing an incompatible output into a node
# # * Not covering all outputs of a node
# # * Ending a graph run with an incompatible output
#
# add_edges(
#     g,
#     review_outline,
#     cases()
#     .case(
#         handle(ReviseOutline)
#         .transform(transform_revise_outline)
#         .route_to(generate_outline)
#     )
#     .case(handle(ApproveOutline).transform(transform_approve_outline).end()),
# )

# reveal_type(g)
# reveal_type(edges)

# reveal_type(review_outline)
# reveal_type(edges)

# add_edges(reveal_type(review_outline), reveal_type(edges))

# g.edge(
#     source=generate_outline,
#     transform=transform_outline,
#     destination=review_outline,
# )
# g.edges(  # or g.edge?
#     generate_outline,
#     review_outline,
# )
# g.edges(
#     generate_outline,
#     lambda h: Routing[h(Outline).route_to(review_outline)],
# )

# graph = g.build()

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Never, cast, get_args, get_origin, overload

from pydantic_graph.v2.decision import Decision, DecisionBranchBuilder
from pydantic_graph.v2.id_types import ForkId, JoinId, NodeId
from pydantic_graph.v2.join import Join, ReducerFactory
from pydantic_graph.v2.mermaid import StateDiagramDirection, generate_code
from pydantic_graph.v2.node import (
    END,
    START,
    EndNode,
    Spread,
    StartNode,
)
from pydantic_graph.v2.node_types import (
    AnyDestinationNode,
    AnyNode,
    AnySourceNode,
    is_destination,
    is_source,
)
from pydantic_graph.v2.parent_forks import ParentFork, ParentForkFinder
from pydantic_graph.v2.step import Step, StepCallProtocol
from pydantic_graph.v2.transform import AnyTransformFunction, TransformFunction
from pydantic_graph.v2.util import TypeExpression, get_callable_name, TypeOrTypeExpression


# Node building:
@overload
def step[StateT, DepsT, InputT, OutputT](
    *,
    node_id: str | None = None,
    label: str | None = None,
) -> Callable[[StepCallProtocol[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]: ...


@overload
def step[StateT, DepsT, InputT, OutputT](
    call: StepCallProtocol[StateT, DepsT, InputT, OutputT],
    *,
    node_id: str | None = None,
    label: str | None = None,
) -> Step[StateT, DepsT, InputT, OutputT]: ...


def step[StateT, DepsT, InputT, OutputT](
    call: StepCallProtocol[StateT, DepsT, InputT, OutputT] | None = None,
    *,
    node_id: str | None = None,
    label: str | None = None,
) -> (
    Step[StateT, DepsT, InputT, OutputT]
    | Callable[[StepCallProtocol[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]
):
    if call is None:

        def decorator(
            func: StepCallProtocol[StateT, DepsT, InputT, OutputT],
        ) -> Step[StateT, DepsT, InputT, OutputT]:
            return step(call=func, node_id=node_id, label=label)

        return decorator

    if node_id is None:
        node_id = get_callable_name(call)

    return Step[StateT, DepsT, InputT, OutputT](id=NodeId(node_id), call=call, user_label=label)


@overload
def join[StateT, DepsT, InputT, OutputT](
    *,
    node_id: str | None = None,
) -> Callable[[ReducerFactory[StateT, DepsT, InputT, OutputT]], Join[StateT, DepsT, InputT, OutputT]]: ...


@overload
def join[StateT, DepsT, InputT, OutputT](
    reducer_factory: ReducerFactory[StateT, DepsT, InputT, OutputT],
    *,
    node_id: str | None = None,
) -> Join[StateT, DepsT, InputT, OutputT]: ...


def join[StateT, DepsT](
    reducer_factory: ReducerFactory[StateT, DepsT, Any, Any] | None = None,
    *,
    node_id: str | None = None,
) -> Join[StateT, DepsT, Any, Any] | Callable[[ReducerFactory[StateT, DepsT, Any, Any]], Join[StateT, DepsT, Any, Any]]:
    if reducer_factory is None:

        def decorator(
            reducer_factory: ReducerFactory[StateT, DepsT, Any, Any],
        ) -> Join[StateT, DepsT, Any, Any]:
            return join(reducer_factory=reducer_factory, node_id=node_id)

        return decorator

    if node_id is None:
        # TODO: Ideally we'd be able to infer this from the parent frame variable assignment or similar
        node_id = get_callable_name(reducer_factory)

    return Join[StateT, DepsT, Any, Any](
        id=JoinId(NodeId(node_id)),
        reducer_factory=reducer_factory,
    )


def decision(*, node_id: str, note: str | None = None) -> Decision[Any, Any, Never, Never]:
    return Decision[Any, Any, Never, Never](id=NodeId(node_id), branches=[], note=note)


@dataclass
class Edge:
    source_id: NodeId
    transform: AnyTransformFunction | None
    destination_id: NodeId
    user_label: str | None

    def source(self, nodes: dict[NodeId, AnyNode]) -> AnySourceNode:
        node = nodes.get(self.source_id)
        if node is None:
            raise ValueError(f'Node {self.source_id} not found in graph')
        if not is_source(node):
            raise ValueError(f'Node {self.source_id} is not a source node: {node}')
        return node

    def destination(self, nodes: dict[NodeId, AnyNode]) -> AnyDestinationNode:
        node = nodes.get(self.destination_id)
        if node is None:
            raise ValueError(f'Node {self.destination_id} not found in graph')
        if not is_destination(node):
            raise ValueError(f'Node {self.destination_id} is not a source node: {node}')
        return node

    @property
    def label(self) -> str | None:
        # TODO: Add some default behavior?
        return self.user_label


@dataclass
class GraphBuilder[StateT, DepsT, GraphInputT, GraphOutputT]:
    state_type: TypeOrTypeExpression[StateT]
    deps_type: TypeOrTypeExpression[DepsT]
    input_type: TypeOrTypeExpression[GraphInputT]
    output_type: TypeOrTypeExpression[GraphOutputT]

    parallel: bool = True  # if False, allow direct state modification and don't copy state sent to steps, but disallow parallel node execution

    _nodes: dict[NodeId, AnyNode] = field(init=False, default_factory=dict)
    _edges_by_source: dict[NodeId, list[Edge]] = field(init=False, default_factory=lambda: defaultdict(list))
    _decision_index: int = field(init=False, default=1)

    type Source[OutputT] = Step[StateT, DepsT, Any, OutputT] | Join[StateT, DepsT, Any, OutputT] | Spread[Any, OutputT]
    type Destination[InputT] = (
        Step[StateT, DepsT, InputT, Any]
        | Join[StateT, DepsT, InputT, Any]
        | Decision[StateT, DepsT, InputT, GraphOutputT]
        | Spread[InputT, Any]
    )

    def __post_init__(self):
        self._nodes[START.id] = START
        self._nodes[END.id] = END

    # Node building:
    @overload
    def step[InputT, OutputT](
        self,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Callable[[StepCallProtocol[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]: ...

    @overload
    def step[InputT, OutputT](
        self,
        call: StepCallProtocol[StateT, DepsT, InputT, OutputT],
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Step[StateT, DepsT, InputT, OutputT]: ...

    def step[InputT, OutputT](
        self,
        call: StepCallProtocol[StateT, DepsT, InputT, OutputT] | None = None,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> (
        Step[StateT, DepsT, InputT, OutputT]
        | Callable[[StepCallProtocol[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]
    ):
        if call is None:
            return step(node_id=node_id, label=label)
        else:
            return step(call=call, node_id=node_id, label=label)

    def spread[ItemT](
        self,
        item_type: TypeOrTypeExpression[ItemT],
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Spread[Sequence[ItemT], ItemT]:
        # TODO: Need to get a unique spread ID
        return Spread(id=ForkId(NodeId(node_id or 'spread')),)

    @overload
    def join[InputT, OutputT](
        self,
        *,
        node_id: str | None = None,
    ) -> Callable[[ReducerFactory[StateT, DepsT, InputT, OutputT]], Join[StateT, DepsT, InputT, OutputT]]: ...

    @overload
    def join[InputT, OutputT](
        self,
        reducer_factory: ReducerFactory[StateT, DepsT, InputT, OutputT],
        *,
        node_id: str | None = None,
    ) -> Join[StateT, DepsT, InputT, OutputT]: ...

    def join(
        self,
        reducer_factory: ReducerFactory[StateT, DepsT, Any, Any] | None = None,
        *,
        node_id: str | None = None,
    ) -> (
        Join[StateT, DepsT, Any, Any]
        | Callable[[ReducerFactory[StateT, DepsT, Any, Any]], Join[StateT, DepsT, Any, Any]]
    ):
        if reducer_factory is None:
            return join(node_id=node_id)
        else:
            return join(reducer_factory=reducer_factory, node_id=node_id)

    def decision(self, *, node_id: str | None = None, note: str | None = None) -> Decision[StateT, DepsT, Never, Never]:
        if node_id is None:
            node_id = self._get_new_decision_id()

        return decision(node_id=node_id, note=note)

    def _get_new_decision_id(self) -> str:
        node_id = f'decision_{self._decision_index}'
        self._decision_index += 1
        while node_id in self._nodes:
            node_id = f'decision_{self._decision_index}'
            self._decision_index += 1
        return node_id

    def _get_new_spread_id(self, from_: str, to: str) -> str:
        prefix = f'spread_from_{from_}_to_{to}'

        node_id = prefix
        index = 2
        while node_id in self._nodes:
            node_id = f'{prefix}_{index}'
            index += 1
        return node_id

    # def handle[SourceT](
    #     self,
    #     case: type[SourceT] | type[TypeExpression[SourceT]],
    #     *,
    #     matches: Callable[[Any], bool] | None = None,
    #     label: str | None = None,
    # ) -> DecisionBranchBuilder[StateT, DepsT, SourceT, Any]:
    #     extracted_case = cast(type[SourceT], get_args(case)[0] if get_origin(case) is TypeExpression else case)
    #     return DecisionBranchBuilder(extracted_case, matches, transforms=(), user_label=label)

    # Edge building
    # Node "types" to be connected into edges: 'start', 'end', Step, Decision, Join, Fork.
    # You typically don't manually create forks — they are inferred from multiple edges coming out of a single node.
    @overload
    def start_with(
        self,
        destination: Destination[GraphInputT],
        *,
        label: str | None = None,
    ) -> None: ...

    @overload
    def start_with[DestinationInputT](
        self,
        destination: Destination[DestinationInputT],
        *,
        transform: TransformFunction[StateT, DepsT, GraphInputT, DestinationInputT],
        label: str | None = None,
    ) -> None: ...

    def start_with(
        self,
        destination: Destination[Any],
        *,
        transform: TransformFunction[StateT, DepsT, Any, Any] | None = None,
        label: str | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=START,
            transform=transform,
            destination=destination,
            label=label,
        )

    @overload
    def start_with_spread[GraphInputItemT](
        self: GraphBuilder[StateT, DepsT, Sequence[GraphInputItemT], GraphOutputT],
        node: Destination[GraphInputItemT],
        *,
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def start_with_spread[DestinationInputT](
        self,
        node: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[StateT, DepsT, GraphInputT, Sequence[DestinationInputT]],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def start_with_spread[GraphInputItemT, DestinationInputT](
        self: GraphBuilder[StateT, DepsT, Sequence[GraphInputItemT], GraphOutputT],
        node: Destination[DestinationInputT],
        *,
        post_spread_transform: TransformFunction[StateT, DepsT, GraphInputItemT, DestinationInputT],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def start_with_spread[IntermediateT, DestinationInputT](
        self: GraphBuilder[StateT, DepsT, GraphInputT, GraphOutputT],
        node: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[StateT, DepsT, GraphInputT, Sequence[IntermediateT]],
        post_spread_transform: TransformFunction[StateT, DepsT, IntermediateT, DestinationInputT],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    def start_with_spread(
        self,
        node: Destination[Any],
        *,
        spread_id: str | None = None,
        pre_spread_transform: TransformFunction[StateT, DepsT, Any, Sequence[Any]] | None = None,
        post_spread_transform: TransformFunction[StateT, DepsT, Any, Any] | None = None,
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None:
        # TODO: Need to allow specifying the id manually to prevent conflicts if there are multiple spreads between the same two nodes
        if spread_id is None:
            spread_id = self._get_new_spread_id(from_='start', to=node.id)
        if spread_id in self._nodes:
            raise ValueError(f'Spread ID {spread_id!r} is already used in the graph. Please specify a unique ID.')

        spread = Spread(id=ForkId(NodeId(spread_id)))
        self._add_edge_from_nodes(
            source=START,
            transform=pre_spread_transform,
            destination=spread,
            label=pre_spread_label,
        )
        self._add_edge_from_nodes(
            source=spread,
            transform=post_spread_transform,
            destination=node,
            label=post_spread_label,
        )

    @overload
    def add_edge[SourceOutputT](
        self,
        source: Source[SourceOutputT],
        destination: Destination[SourceOutputT],
        *,
        label: str | None = None,
    ) -> None: ...

    @overload
    def add_edge[SourceOutputT, DestinationInputT](
        self,
        source: Source[SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        transform: TransformFunction[StateT, DepsT, SourceOutputT, DestinationInputT],
        label: str | None = None,
    ) -> None: ...

    def add_edge[SourceOutputT](
        self,
        source: Source[SourceOutputT],
        destination: Destination[Any],
        *,
        transform: TransformFunction[StateT, DepsT, SourceOutputT, Any] | None = None,
        label: str | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=source,
            transform=transform,
            destination=destination,
            label=label,
        )

    @overload
    def add_spreading_edge[DestinationInputT](
        self,
        source: Source[Sequence[DestinationInputT]],
        destination: Destination[DestinationInputT],
        *,
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def add_spreading_edge[SourceOutputT, DestinationInputT](
        self,
        source: Source[SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[StateT, DepsT, SourceOutputT, Sequence[DestinationInputT]],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def add_spreading_edge[SourceOutputItemT, DestinationInputT](
        self,
        source: Source[Sequence[SourceOutputItemT]],
        destination: Destination[DestinationInputT],
        *,
        post_spread_transform: TransformFunction[
            StateT,
            DepsT,
            SourceOutputItemT,
            DestinationInputT,
        ],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def add_spreading_edge[SourceOutputT, IntermediateT, DestinationInputT](
        self,
        source: Source[SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[StateT, DepsT, SourceOutputT, Sequence[IntermediateT]],
        post_spread_transform: TransformFunction[StateT, DepsT, IntermediateT, DestinationInputT],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    def add_spreading_edge(
        self,
        source: Source[Any],
        destination: Destination[Any],
        *,
        spread_id: str | None = None,
        pre_spread_transform: TransformFunction[StateT, DepsT, Any, Sequence[Any]] | None = None,
        post_spread_transform: TransformFunction[StateT, DepsT, Any, Any] | None = None,
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None:
        if spread_id is None:
            spread_id = self._get_new_spread_id(from_=source.id, to=destination.id)
        if spread_id in self._nodes:
            raise ValueError(f'Spread ID {spread_id!r} is already used in the graph. Please specify a unique ID.')

        spread = Spread(id=ForkId(NodeId(spread_id)))
        self._add_edge_from_nodes(
            source=source,
            transform=pre_spread_transform,
            destination=spread,
            label=pre_spread_label,
        )
        self._add_edge_from_nodes(
            source=spread,
            transform=post_spread_transform,
            destination=destination,
            label=post_spread_label,
        )

    @overload
    def end_from(
        self,
        source: Source[GraphOutputT],
        *,
        label: str | None = None,
    ) -> None: ...

    @overload
    def end_from[SourceOutputT](
        self,
        source: Source[SourceOutputT],
        *,
        label: str | None = None,
        transform: TransformFunction[StateT, DepsT, SourceOutputT, GraphOutputT],
    ) -> None: ...

    def end_from(
        self,
        source: Source[Any],
        *,
        label: str | None = None,
        transform: TransformFunction[StateT, DepsT, Any, GraphOutputT] | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=source,
            transform=transform,
            destination=END,
            label=label,
        )

    def _add_edge_from_nodes(
        self,
        *,
        source: AnySourceNode,
        transform: AnyTransformFunction | None,
        destination: AnyDestinationNode,
        label: str | None = None,
    ) -> None:
        self._insert_node(source)
        self._insert_node(destination)

        edge = Edge(source_id=source.id, transform=transform, destination_id=destination.id, user_label=label)
        self._insert_edge(edge)

    def _insert_node(self, node: AnyNode) -> None:
        existing = self._nodes.get(node.id)
        if existing is None:
            self._nodes[node.id] = node
        elif isinstance(existing, (StartNode, EndNode)):
            pass  # it's not a problem to have non-unique instances of StartNode and EndNode
        elif existing is not node:
            raise ValueError(f'All nodes must have unique node IDs. {node.id!r} was the ID for {existing} and {node}')

    def _insert_edge(self, edge: Edge) -> None:
        assert edge.source_id in self._nodes, f'Edge source {edge.source_id} not found in graph'
        assert edge.destination_id in self._nodes, f'Edge destination {edge.destination_id} not found in graph'
        self._edges_by_source[edge.source_id].append(edge)

    def build(self, parallel: bool = True) -> Graph[StateT, DepsT, GraphInputT, GraphOutputT]:
        # TODO: Warn/error if there is no start node / edges, or end node / edges
        # TODO: Warn/error if the graph is not connected
        # TODO: Warn/error if any non-End node is a dead end
        # TODO: Error if the graph does not meet the every-join-has-a-parent-fork requirement (otherwise can't know when to proceed past joins)
        # TODO: Allow the user to specify the parent forks; only infer them if _not_ specified
        # TODO: Verify that any user-specified parent forks are _actually_ valid parent forks, and if not, generate a helpful error message
        # TODO: Consider doing a deepcopy here to prevent modifications to the underlying nodes and edges
        # TODO: Error if `parallel` is False but the graph has forks / spreads
        nodes = self._nodes
        edges = self._edges_by_source
        nodes, edges = _convert_decision_spreads(nodes, edges)
        parent_forks = _collect_dominating_forks(nodes, edges)

        output_type = cast(type[GraphOutputT], self.output_type)
        if get_origin(output_type) is TypeExpression:
            output_type = get_args(output_type)[0]

        return Graph[StateT, DepsT, GraphInputT, GraphOutputT](
            state_type=self.state_type,
            deps_type=self.deps_type,
            input_type=self.input_type,
            output_type=output_type,
            parallel=parallel,
            nodes=self._nodes,
            edges_by_source=self._edges_by_source,
            parent_forks=parent_forks,
        )


def _convert_decision_spreads(
    graph_nodes: dict[NodeId, AnyNode], graph_edges_by_source: dict[NodeId, list[Edge]]
) -> tuple[dict[NodeId, AnyNode], dict[NodeId, list[Edge]]]:
    def _get_next_spread_id(to: str) -> NodeId:
        prefix = f'spread_to_{to}'
        node_id = prefix
        index = 2
        while node_id in graph_nodes:
            node_id = f'{prefix}_{index}'
            index += 1
        return NodeId(node_id)

    nodes = graph_nodes
    edges = graph_edges_by_source

    for node in list(nodes.values()):
        if isinstance(node, Decision):
            for branch in node.branches:
                if branch.spread:
                    spread = Spread(id=ForkId(_get_next_spread_id(to=branch.route_to.id)))
                    old_route_to = branch.route_to
                    nodes[spread.id] = spread
                    edges[spread.id].append(
                        Edge(
                            source_id=spread.id,
                            transform=branch.post_spread_transform,
                            destination_id=old_route_to.id,
                            user_label=branch.post_spread_user_label,
                        )
                    )
                    branch.route_to = spread
                    branch.spread = False
                    branch.post_spread_transform = None
    return nodes, edges


def _collect_dominating_forks(
    graph_nodes: dict[NodeId, AnyNode], graph_edges_by_source: dict[NodeId, list[Edge]]
) -> dict[JoinId, ParentFork[NodeId]]:
    nodes = set(graph_nodes)
    start_ids = {StartNode.start.id}
    edges = {source_id: [e.destination_id for e in graph_edges_by_source[source_id]] for source_id in nodes}
    for node_id, node in graph_nodes.items():
        if isinstance(node, Decision):
            # For decisions, we need to add edges for the branches
            for branch in node.branches:
                # If any branches have a spread, it's a bug in graph building
                assert not branch.spread, 'Decision branches should not be spreads at this point'
                edges[node_id].append(branch.route_to.id)

    fork_ids = {
        node_id for node_id, node in graph_nodes.items() if isinstance(node, Spread) or len(edges.get(node_id, [])) > 1
    }

    finder = ParentForkFinder(
        nodes=nodes,
        start_ids=start_ids,
        fork_ids=fork_ids,
        edges=edges,
    )

    join_ids = {node.id for node in graph_nodes.values() if isinstance(node, Join)}
    dominating_forks: dict[JoinId, ParentFork[NodeId]] = {}
    for join_id in join_ids:
        dominating_fork = finder.find_parent_fork(join_id)
        if dominating_fork is None:
            # TODO: Print out the mermaid graph and explain the problem
            raise ValueError(f'Join node {join_id} has no dominating fork')
        dominating_forks[join_id] = dominating_fork

    return dominating_forks


@dataclass(repr=False)
class Graph[StateT, DepsT, InputT, OutputT]:
    state_type: type[StateT]
    deps_type: type[DepsT]
    input_type: type[InputT]
    output_type: type[OutputT]

    nodes: dict[NodeId, AnyNode]
    edges_by_source: dict[NodeId, list[Edge]]
    parent_forks: dict[JoinId, ParentFork[NodeId]]

    parallel: bool  # if False, allow direct state modification and don't copy state sent to steps, but disallow parallel node execution

    def __post_init__(self):
        assert StartNode.start.id in self.nodes, 'Graph must have a start node'

    @property
    def start_edges(self) -> list[Edge]:
        return self.edges_by_source.get(START.id, [])

    def get_parent_fork(self, join_id: JoinId) -> ParentFork[NodeId]:
        result = self.parent_forks.get(join_id)
        if result is None:
            raise RuntimeError(f'Node {join_id} is not a join node or did not have a dominating fork (this is a bug)')
        return result

    def render(self, *, title: str | None = None, direction: StateDiagramDirection | None = None) -> str:
        return generate_code(self, title=title, direction=direction)

    # TODO: Should we re-add a `run` method that just uses a default global GraphRunner?

    def __repr__(self):
        return self.render()

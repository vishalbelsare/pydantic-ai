import asyncio

import logfire
import pytest
from logfire.testing import CaptureLogfire

from pydantic_evals.otel.context_in_memory_span_exporter import (
    context_subtree,
)
from pydantic_evals.otel.span_tree import SpanTree

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def use_logfire(capfire: CaptureLogfire):
    assert capfire


async def test_context_subtree_concurrent():
    """Test that context_subtree correctly records spans in independent async contexts."""

    # Create independent async tasks
    async def task1():
        with context_subtree() as tree:
            with logfire.span('task1'):
                with logfire.span('task1_child1'):
                    await asyncio.sleep(0.01)
                with logfire.span('task1_child2'):
                    await asyncio.sleep(0.01)
        return tree

    async def task2():
        with context_subtree() as tree:
            with logfire.span('task2'):
                with logfire.span('task2_child1'):
                    await asyncio.sleep(0.01)
                    with logfire.span('task2_grandchild'):
                        await asyncio.sleep(0.01)
        return tree

    # Execute tasks concurrently
    tree1, tree2 = await asyncio.gather(task1(), task2())

    # Verify that tree1 only contains spans from task1
    assert len(tree1.roots) == 1, 'tree1 should have exactly one root span'
    assert tree1.roots[0].name == 'task1', 'tree1 root should be task1'
    assert not tree1.any(lambda node: node.name == 'task2'), 'tree1 should not contain task2 spans'
    assert not tree1.any(lambda node: node.name == 'task2_child1'), 'tree1 should not contain task2_child1 spans'
    assert not tree1.any(lambda node: node.name == 'task2_grandchild'), (
        'tree1 should not contain task2_grandchild spans'
    )

    # Verify task1 children
    task1_root = tree1.roots[0]
    assert len(task1_root.children) == 2, 'task1 should have exactly two children'
    task1_child_names = {child.name for child in task1_root.children}
    assert task1_child_names == {'task1_child1', 'task1_child2'}, (
        "task1's children should be task1_child1 and task1_child2"
    )

    # Verify that tree2 only contains spans from task2
    assert len(tree2.roots) == 1, 'tree2 should have exactly one root span'
    assert tree2.roots[0].name == 'task2', 'tree2 root should be task2'
    assert not tree2.any(lambda node: node.name == 'task1'), 'tree2 should not contain task1 spans'
    assert not tree2.any(lambda node: node.name == 'task1_child1'), 'tree2 should not contain task1_child1 spans'
    assert not tree2.any(lambda node: node.name == 'task1_child2'), 'tree2 should not contain task1_child2 spans'

    # Verify task2 structure
    task2_root = tree2.roots[0]
    assert len(task2_root.children) == 1, 'task2 should have exactly one child'
    assert task2_root.children[0].name == 'task2_child1', "task2's child should be task2_child1"

    # Verify grandchild
    task2_child = task2_root.children[0]
    assert len(task2_child.children) == 1, 'task2_child1 should have exactly one child'
    assert task2_child.children[0].name == 'task2_grandchild', "task2_child1's child should be task2_grandchild"


@pytest.fixture
async def span_tree() -> SpanTree:
    """Fixture that creates a span tree with a predefined structure and attributes."""
    # Create spans with a tree structure and attributes
    with context_subtree() as tree:
        with logfire.span('root', level='0'):
            with logfire.span('child1', level='1', type='important'):
                with logfire.span('grandchild1', level='2', type='important'):
                    pass
                with logfire.span('grandchild2', level='2', type='normal'):
                    pass
            with logfire.span('child2', level='1', type='normal'):
                with logfire.span('grandchild3', level='2', type='normal'):
                    pass
    return tree


async def test_span_tree_flattened(span_tree: SpanTree):
    """Test the flattened() method of SpanTree."""
    flattened_nodes = span_tree.flattened()
    assert len(flattened_nodes) == 6, 'Should have 6 spans in total'

    # Check that all expected nodes are in the flattened list
    node_names = {node.name for node in flattened_nodes}
    expected_names = {'root', 'child1', 'child2', 'grandchild1', 'grandchild2', 'grandchild3'}
    assert node_names == expected_names


async def test_span_tree_find_all(span_tree: SpanTree):
    """Test the find_all method of SpanTree."""
    # Find nodes with important type
    important_nodes = span_tree.find_all(lambda node: node.attributes.get('type') == 'important')
    assert len(important_nodes) == 2
    important_names = {node.name for node in important_nodes}
    assert important_names == {'child1', 'grandchild1'}

    # Find nodes with level 2
    level2_nodes = span_tree.find_all(lambda node: node.attributes.get('level') == '2')
    assert len(level2_nodes) == 3
    level2_names = {node.name for node in level2_nodes}
    assert level2_names == {'grandchild1', 'grandchild2', 'grandchild3'}


async def test_span_tree_any(span_tree: SpanTree):
    """Test the any() method of SpanTree."""
    # Test existence of a node by name
    assert span_tree.any(lambda node: node.name == 'grandchild2')

    # Test non-existence
    assert not span_tree.any(lambda node: node.name == 'non_existent')

    # Test existence by attribute
    assert span_tree.any(lambda node: node.attributes.get('type') == 'important')


async def test_span_node_find_children(span_tree: SpanTree):
    """Test the find_children method of SpanNode."""
    root_node = span_tree.roots[0]
    assert root_node.name == 'root'

    # Find all children with a level attribute
    child_nodes = root_node.find_children(lambda node: 'level' in node.attributes)
    assert len(child_nodes) == 2

    # Check that the children have the expected names
    child_names = {node.name for node in child_nodes}
    assert child_names == {'child1', 'child2'}


async def test_span_node_first_child(span_tree: SpanTree):
    """Test the first_child method of SpanNode."""
    root_node = span_tree.roots[0]

    # Find first child with important type
    first_important_child = root_node.first_child(lambda node: node.attributes.get('type') == 'important')
    assert first_important_child is not None
    assert first_important_child.name == 'child1'

    # Test for non-existent attribute
    non_existent = root_node.first_child(lambda node: node.attributes.get('non_existent') == 'value')
    assert non_existent is None


async def test_span_node_any_child(span_tree: SpanTree):
    """Test the any_child method of SpanNode."""
    root_node = span_tree.roots[0]

    # Test existence of child with normal type
    assert root_node.any_child(lambda node: node.attributes.get('type') == 'normal')

    # Test non-existence
    assert not root_node.any_child(lambda node: node.name == 'non_existent')


async def test_span_node_find_descendants(span_tree: SpanTree):
    """Test the find_descendants method of SpanNode."""
    root_node = span_tree.roots[0]

    # Find all descendants with level 2
    level2_nodes = root_node.find_descendants(lambda node: node.attributes.get('level') == '2')
    assert len(level2_nodes) == 3

    # Check that they have the expected names
    level2_names = {node.name for node in level2_nodes}
    assert level2_names == {'grandchild1', 'grandchild2', 'grandchild3'}


async def test_span_node_matches(span_tree: SpanTree):
    """Test the matches method of SpanNode."""
    root_node = span_tree.roots[0]
    child1_node = root_node.first_child(lambda node: node.name == 'child1')
    assert child1_node is not None

    # Test matches by name
    assert child1_node.matches(name='child1')
    assert not child1_node.matches(name='child2')

    # Test matches by attributes
    assert child1_node.matches(attributes={'level': '1', 'type': 'important'})
    assert not child1_node.matches(attributes={'level': '2', 'type': 'important'})

    # Test matches by both name and attributes
    assert child1_node.matches(name='child1', attributes={'type': 'important'})
    assert not child1_node.matches(name='child1', attributes={'type': 'normal'})


async def test_span_tree_ancestors_methods():
    """Test the ancestor traversal methods in SpanNode."""
    # Configure logfire
    logfire.configure()

    # Create spans with a deep structure for testing ancestor methods
    with context_subtree() as tree:
        with logfire.span('root', depth=0):
            with logfire.span('level1', depth=1):
                with logfire.span('level2', depth=2):
                    with logfire.span('level3', depth=3):
                        with logfire.span('leaf', depth=4):
                            # Add a log message to test nested logs
                            logfire.info('This is a leaf node log message')

    # Get the leaf node
    leaf_node = tree.find_first(lambda node: node.name == 'leaf')
    assert leaf_node is not None

    # Test find_ancestors
    ancestors = leaf_node.find_ancestors(lambda node: True)
    assert len(ancestors) == 4
    ancestor_names = [node.name for node in ancestors]
    assert ancestor_names == ['level3', 'level2', 'level1', 'root']

    # Test first_ancestor by name instead of depth comparison to avoid type issues
    level2_ancestor = leaf_node.first_ancestor(lambda node: node.name == 'level2')
    assert level2_ancestor is not None
    assert level2_ancestor.name == 'level2'

    # Test any_ancestor
    assert leaf_node.any_ancestor(lambda node: node.name == 'root')
    assert not leaf_node.any_ancestor(lambda node: node.name == 'non_existent')


async def test_log_levels_and_exceptions():
    """Test recording different log levels and exceptions in spans."""
    # Configure logfire
    logfire.configure()

    with context_subtree() as tree:
        # Test different log levels
        with logfire.span('parent_span'):
            logfire.debug('Debug message')
            logfire.info('Info message')
            logfire.warn('Warning message')

            # Create child span with error
            with logfire.span('error_child') as error_span:
                logfire.error('Error occurred')
                # Record exception
                try:
                    raise ValueError('Test exception')
                except ValueError as e:
                    error_span.record_exception(e)

    # Verify log levels are preserved
    parent_span = tree.find_first(lambda node: node.name == 'parent_span')
    assert parent_span is not None

    # Find the error child span
    error_child = parent_span.first_child(lambda node: node.name == 'error_child')
    assert error_child is not None

    # Verify attributes reflect log levels and exceptions
    log_nodes = parent_span.find_descendants(
        lambda node: 'Debug message' in str(node.attributes)
        or 'Info message' in str(node.attributes)
        or 'Warning message' in str(node.attributes)
        or 'Error occurred' in str(node.attributes)
    )
    assert len(log_nodes) > 0, 'Should have log messages as spans'

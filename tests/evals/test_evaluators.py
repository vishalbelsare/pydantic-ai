from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_evals.evaluators.common import is_instance, llm_judge
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.evaluators.spec import Evaluator, EvaluatorResult, EvaluatorSpec
from pydantic_evals.otel.span_tree import SpanTree

pytestmark = pytest.mark.anyio


class TaskInput(BaseModel):
    query: str


class TaskOutput(BaseModel):
    answer: str


class TaskMetadata(BaseModel):
    difficulty: str = 'easy'


@pytest.fixture
def test_context() -> EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]:
    return EvaluatorContext[TaskInput, TaskOutput, TaskMetadata](
        name='test_case',
        inputs=TaskInput(query='What is 2+2?'),
        output=TaskOutput(answer='4'),
        expected_output=TaskOutput(answer='4'),
        metadata=TaskMetadata(difficulty='easy'),
        duration=0.1,
        span_tree=SpanTree(),
        attributes={},
        metrics={},
    )


async def test_evaluator_spec_initialization():
    """Test initializing EvaluatorSpec."""
    # Simple form with just a name
    spec1 = EvaluatorSpec(call='my_evaluator')
    assert spec1.call == 'my_evaluator'
    assert spec1.args == []
    assert spec1.kwargs == {}

    # Form with args
    spec2 = EvaluatorSpec(call='my_evaluator', args=['arg1', 'arg2'])
    assert spec2.call == 'my_evaluator'
    assert spec2.args == ['arg1', 'arg2']
    assert spec2.kwargs == {}

    # Form with kwargs
    spec3 = EvaluatorSpec(call='my_evaluator', kwargs={'key1': 'value1', 'key2': 'value2'})
    assert spec3.call == 'my_evaluator'
    assert spec3.args == []
    assert spec3.kwargs == {'key1': 'value1', 'key2': 'value2'}

    # Full form
    spec4 = EvaluatorSpec(call='my_evaluator', args=['arg1'], kwargs={'key1': 'value1'})
    assert spec4.call == 'my_evaluator'
    assert spec4.args == ['arg1']
    assert spec4.kwargs == {'key1': 'value1'}


async def test_evaluator_spec_serialization():
    """Test serializing EvaluatorSpec."""
    # Create a spec
    spec = EvaluatorSpec(call='my_evaluator', args=['arg1'], kwargs={'key1': 'value1'})

    # Serialize it
    serialized = spec.model_dump()

    # Check the serialized form
    assert serialized['call'] == 'my_evaluator'
    assert serialized['args'] == ['arg1']
    assert serialized['kwargs'] == {'key1': 'value1'}

    # Deserialize it
    deserialized = EvaluatorSpec.model_validate(serialized)

    # Check the deserialized form
    assert deserialized.call == 'my_evaluator'
    assert deserialized.args == ['arg1']
    assert deserialized.kwargs == {'key1': 'value1'}


async def test_evaluator_from_function():
    """Test creating an Evaluator from a function."""

    async def test_evaluator_func(ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> dict[str, Any]:
        return {'result': 'test'}

    evaluator = Evaluator[TaskInput, TaskOutput, TaskMetadata].from_function(test_evaluator_func)

    assert evaluator.function is not None
    assert evaluator.spec.call == 'test_evaluator_func'


async def test_evaluator_call(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test calling an Evaluator."""

    async def test_evaluator_func(ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> dict[str, Any]:
        assert ctx.inputs.query == 'What is 2+2?'
        assert ctx.output.answer == '4'
        assert ctx.expected_output and ctx.expected_output.answer == '4'
        assert ctx.metadata and ctx.metadata.difficulty == 'easy'
        return {'result': 'passed'}

    evaluator = Evaluator[TaskInput, TaskOutput, TaskMetadata].from_function(test_evaluator_func)

    results = await evaluator.execute(test_context)
    # results is a list of SourcedEvaluatorOutput, so we need to get the value from it
    assert len(results) == 1
    assert results[0].value == 'passed'


async def test_is_instance_evaluator():
    """Test the is_instance evaluator."""
    # Create a context with the correct object typing for is_instance
    object_context = EvaluatorContext[object, object, object](
        name='test_case',
        inputs=TaskInput(query='What is 2+2?'),
        output=TaskOutput(answer='4'),
        expected_output=None,
        metadata=None,
        duration=0.1,
        span_tree=SpanTree(),
        attributes={},
        metrics={},
    )

    # Test with matching types
    result = await is_instance(object_context, type_name='TaskOutput')
    assert isinstance(result, EvaluatorResult)
    assert result.value is True

    # Test with non-matching types
    class DifferentOutput(BaseModel):
        different_field: str

    # Create a context with DifferentOutput
    diff_context = EvaluatorContext[object, object, object](
        name='mismatch_case',
        inputs=TaskInput(query='What is 2+2?'),
        output=DifferentOutput(different_field='not an answer'),
        expected_output=None,
        metadata=None,
        duration=0.1,
        span_tree=SpanTree(),
        attributes={},
        metrics={},
    )

    result = await is_instance(diff_context, type_name='TaskInput')
    assert isinstance(result, EvaluatorResult)
    assert result.value is False


async def test_llm_judge_evaluator():
    """Test the llm_judge evaluator."""
    # We can't easily test this without mocking the LLM, so we'll just check that it's importable
    assert callable(llm_judge)


async def test_custom_evaluator(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test a custom evaluator."""

    async def custom_evaluator(ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> dict[str, Any]:
        # Check if the answer is correct based on expected output
        is_correct = ctx.output.answer == ctx.expected_output.answer if ctx.expected_output else False

        # Use metadata if available
        difficulty = ctx.metadata.difficulty if ctx.metadata else 'unknown'

        return {
            'is_correct': is_correct,
            'difficulty': difficulty,
        }

    result = await custom_evaluator(test_context)
    assert result['is_correct'] is True
    assert result['difficulty'] == 'easy'


async def test_evaluator_error_handling(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test error handling in evaluators."""

    async def failing_evaluator(ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> dict[str, Any]:
        raise ValueError('Simulated error')

    evaluator = Evaluator[TaskInput, TaskOutput, TaskMetadata].from_function(failing_evaluator)

    # When called directly, it should raise an error
    with pytest.raises(ValueError, match='Simulated error'):
        await evaluator.execute(test_context)


async def test_evaluator_with_null_values():
    """Test evaluator with null expected_output and metadata."""
    context = EvaluatorContext[TaskInput, TaskOutput, TaskMetadata](
        name=None,
        inputs=TaskInput(query='What is 2+2?'),
        output=TaskOutput(answer='4'),
        expected_output=None,
        metadata=None,
        duration=0.1,
        span_tree=SpanTree(),
        attributes={},
        metrics={},
    )

    async def test_evaluator(ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> dict[str, Any]:
        return {
            'has_expected_output': ctx.expected_output is not None,
            'has_metadata': ctx.metadata is not None,
        }

    result = await test_evaluator(context)
    assert result['has_expected_output'] is False
    assert result['has_metadata'] is False

from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_evals.evaluators.spec import EvaluatorSpec, SourcedEvaluatorOutput
from pydantic_evals.reporting import (
    EvaluationReport,
    RenderNumberConfig,
    RenderValueConfig,
    ReportCase,
    ReportCaseAggregate,
)

pytestmark = pytest.mark.anyio


class TaskInput(BaseModel):
    query: str


class TaskOutput(BaseModel):
    answer: str


class TaskMetadata(BaseModel):
    difficulty: str


@pytest.fixture
def sample_evaluator_output() -> dict[str, Any]:
    return {'correct': True, 'confidence': 0.95}


@pytest.fixture
def mock_evaluator_spec() -> EvaluatorSpec:
    return EvaluatorSpec(
        call='test_call',
        args=[],
        kwargs={},
    )


@pytest.fixture
def sample_sourced_output(
    sample_evaluator_output: dict[str, Any], mock_evaluator_spec: EvaluatorSpec
) -> SourcedEvaluatorOutput:
    return SourcedEvaluatorOutput(
        name='test_evaluator',
        value=True,
        reason=None,
        source=mock_evaluator_spec,
    )


@pytest.fixture
def sample_report_case(sample_sourced_output: SourcedEvaluatorOutput) -> ReportCase:
    return ReportCase(
        name='test_case',
        inputs={'query': 'What is 2+2?'},
        output={'answer': '4'},
        expected_output={'answer': '4'},
        metadata={'difficulty': 'easy'},
        metrics={},
        attributes={},
        evaluator_outputs=[sample_sourced_output],
        task_duration=0.1,
        total_duration=0.2,
        trace_id='test-trace-id',
        span_id='test-span-id',
    )


@pytest.fixture
def sample_report(sample_report_case: ReportCase) -> EvaluationReport:
    return EvaluationReport(
        cases=[sample_report_case],
        name='test_report',
    )


async def test_report_case_init(sample_sourced_output: SourcedEvaluatorOutput, mock_evaluator_spec: EvaluatorSpec):
    """Test ReportCase initialization."""
    case = ReportCase(
        name='test_case',
        inputs={'query': 'What is 2+2?'},
        output={'answer': '4'},
        expected_output={'answer': '4'},
        metadata={'difficulty': 'easy'},
        metrics={},
        attributes={},
        evaluator_outputs=[sample_sourced_output],
        task_duration=0.1,
        total_duration=0.2,
        trace_id='test-trace-id',
        span_id='test-span-id',
    )

    assert case.name == 'test_case'
    assert case.inputs['query'] == 'What is 2+2?'
    assert case.output['answer'] == '4'
    assert case.expected_output['answer'] == '4'
    assert case.metadata['difficulty'] == 'easy'
    assert len(case.evaluator_outputs) == 1
    assert case.task_duration == 0.1
    assert case.total_duration == 0.2


async def test_report_init(sample_report_case: ReportCase):
    """Test EvaluationReport initialization."""
    report = EvaluationReport(
        cases=[sample_report_case],
        name='test_report',
    )

    assert report.name == 'test_report'
    assert len(report.cases) == 1


async def test_report_add_case(
    sample_report: EvaluationReport, sample_report_case: ReportCase, mock_evaluator_spec: EvaluatorSpec
):
    """Test adding cases to a report."""
    initial_case_count = len(sample_report.cases)

    # Create a new case
    new_case = ReportCase(
        name='new_case',
        inputs={'query': 'What is 3+3?'},
        output={'answer': '6'},
        expected_output={'answer': '6'},
        metadata={'difficulty': 'medium'},
        metrics={},
        attributes={},
        evaluator_outputs=[],
        task_duration=0.1,
        total_duration=0.15,
        trace_id='test-trace-id-2',
        span_id='test-span-id-2',
    )

    # Add the case
    sample_report.cases.append(new_case)

    # Check that the case was added
    assert len(sample_report.cases) == initial_case_count + 1
    assert sample_report.cases[-1].name == 'new_case'


async def test_report_case_aggregate():
    """Test ReportCaseAggregate functionality."""
    # Create a case aggregate
    aggregate = ReportCaseAggregate(
        name='test_aggregate',
        scores={'test_evaluator': 0.75},
        labels={'test_label': {'value': 0.75}},
        metrics={'accuracy': 0.75},
        assertions=0.75,
        task_duration=0.1,
        total_duration=0.2,
    )

    assert aggregate.name == 'test_aggregate'
    assert aggregate.scores['test_evaluator'] == 0.75
    assert aggregate.labels['test_label']['value'] == 0.75
    assert aggregate.metrics['accuracy'] == 0.75
    assert aggregate.assertions == 0.75
    assert aggregate.task_duration == 0.1
    assert aggregate.total_duration == 0.2


async def test_report_serialization(sample_report: EvaluationReport):
    """Test serializing a report to dict."""
    # Serialize the report
    serialized = sample_report.model_dump()

    # Check the serialized structure
    assert 'cases' in serialized
    assert 'name' in serialized

    # Check the values
    assert serialized['name'] == 'test_report'
    assert len(serialized['cases']) == 1


async def test_report_with_error(mock_evaluator_spec: EvaluatorSpec):
    """Test a report with error in one of the cases."""
    # Create an evaluator output
    error_output = SourcedEvaluatorOutput(
        name='error_evaluator',
        value=False,  # No result
        reason='Test error message',
        source=mock_evaluator_spec,
    )

    # Create a case
    error_case = ReportCase(
        name='error_case',
        inputs={'query': 'What is 1/0?'},
        output=None,
        expected_output={'answer': 'Error'},
        metadata={'difficulty': 'hard'},
        metrics={},
        attributes={'error': 'Division by zero'},
        evaluator_outputs=[error_output],
        task_duration=0.05,
        total_duration=0.1,
        trace_id='test-error-trace-id',
        span_id='test-error-span-id',
    )

    # Create a report with the error case
    report = EvaluationReport(
        cases=[error_case],
        name='error_report',
    )

    assert report.cases[0].attributes['error'] == 'Division by zero'
    assert report.cases[0].evaluator_outputs[0].reason == 'Test error message'


async def test_render_config():
    """Test render configuration objects."""
    # Test RenderNumberConfig
    number_config: RenderNumberConfig = {
        'value_formatter': '{:.0%}',
        'diff_formatter': '{:+.0%}',
        'diff_atol': 0.01,
        'diff_rtol': 0.05,
        'diff_increase_style': 'green',
        'diff_decrease_style': 'red',
    }

    # Assert the dictionary has the expected keys
    assert 'value_formatter' in number_config
    assert 'diff_formatter' in number_config
    assert 'diff_atol' in number_config
    assert 'diff_rtol' in number_config
    assert 'diff_increase_style' in number_config
    assert 'diff_decrease_style' in number_config

    # Test RenderValueConfig
    value_config: RenderValueConfig = {
        'value_formatter': '{value}',
        'diff_checker': lambda x, y: x != y,
        'diff_formatter': None,
        'diff_style': 'magenta',
    }

    # Assert the dictionary has the expected keys
    assert 'value_formatter' in value_config
    assert 'diff_checker' in value_config
    assert 'diff_formatter' in value_config
    assert 'diff_style' in value_config

from pathlib import Path
from types import NoneType

import logfire
from pydantic_evals import Dataset

from pydantic_ai_examples.evals.time_range.agent import infer_time_range
from pydantic_ai_examples.evals.time_range.custom_evaluators import (
    CUSTOM_EVALUATOR_TYPES,
)
from pydantic_ai_examples.evals.time_range.models import (
    TimeRangeInputs,
    TimeRangeResponse,
)

logfire.configure(
    # send_to_logfire='if-token-present',
    environment='development',
    service_name='evals',
    advanced=logfire.AdvancedOptions(base_url='http://localhost:8000')
)
logfire.instrument_pydantic_ai()


def evaluate_dataset():
    dataset_path = Path(__file__).parent / 'datasets' / 'time_range_v2.yaml'
    dataset = Dataset[TimeRangeInputs, TimeRangeResponse, NoneType].from_file(
        dataset_path, custom_evaluator_types=CUSTOM_EVALUATOR_TYPES
    )
    report = dataset.evaluate_sync(infer_time_range)
    print(report)

    assertion_pass_rate = report.averages().assertions
    assert assertion_pass_rate is not None, 'There should be at least one assertion'
    assert assertion_pass_rate > 0.9, (
        f'The assertion pass rate was {assertion_pass_rate:.1%}; it should be above 90%.'
    )


if __name__ == '__main__':
    evaluate_dataset()

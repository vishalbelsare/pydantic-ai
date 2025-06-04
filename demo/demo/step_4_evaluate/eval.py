from pathlib import Path
from typing import Any

import logfire
from pydantic_evals.dataset import Dataset
from pydantic_evals.evaluators import Evaluator
from pydantic_evals.evaluators.common import IsInstance, LlmJudge
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.evaluators.llm_as_a_judge import GradingOutput, judge_input_output

from demo.step_4_evaluate.app_v4_agent_updated import (
    TimeRangeInputs,
    TimeRangeResponse,
    run_infer_time_range,
)
from demo.util.tokens import get_app_write_token

token = get_app_write_token()
logfire.configure(
    token=token,
    environment="dev",
    service_name="eval",
    advanced=logfire.AdvancedOptions(base_url="http://localhost:8000"),
)


async def judge_time_range_case(
    inputs: TimeRangeInputs, output: TimeRangeResponse
) -> GradingOutput:
    """Judge the output of a time range inference agent based on a rubric."""
    rubric = (
        "The output should be a reasonable time range to select for the given inputs, or an error "
        "message if no good time range could be selected. Pick a score between 0 and 1 to represent how confident "
        "you are that the selected time range was what the user intended, or that an error message was "
        "an appropriate response."
    )
    return await judge_input_output(inputs, output, rubric)


async def main():
    dataset = Dataset[TimeRangeInputs, TimeRangeResponse, dict[str, Any]].from_file(
        Path(__file__).parent / "retrieved_test_cases.yaml",
        custom_evaluator_types=[LlmJudge, IsInstance],
    )

    class MyEvaluator(Evaluator[TimeRangeInputs, TimeRangeResponse, Any]):
        async def evaluate(self, ctx: EvaluatorContext[TimeRangeInputs, TimeRangeResponse]):
            result = await judge_time_range_case(inputs=ctx.inputs, output=ctx.output)
            return {
                "is_reasonable": "yes" if result.pass_ else "no",
                "accuracy": result.score,
            }

    dataset.add_evaluator(MyEvaluator())
    report = await dataset.evaluate(run_infer_time_range, max_concurrency=10)

    report.print(include_input=True, include_output=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

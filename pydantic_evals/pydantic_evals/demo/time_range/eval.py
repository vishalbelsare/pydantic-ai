from pydantic_evals import evaluation
from pydantic_evals.demo.time_range import TimeRangeAgentResponse, infer_time_range
from pydantic_evals.demo.time_range.models import TimeRangeDataset, TimeRangeInputs
from pydantic_evals.evals import EvalCase
from pydantic_evals.llm_as_a_judge import GradingOutput, judge_input_output


async def judge_time_range_case(inputs: TimeRangeInputs, output: TimeRangeAgentResponse) -> GradingOutput:
    """Judge the output of a time range inference agent based on a rubric."""
    rubric = 'The output should be a reasonable time range to select for the given inputs.'
    return await judge_input_output(inputs, output, rubric)


async def main():
    """TODO: Remove this file before merging."""
    import logfire

    logfire.configure(send_to_logfire=False, console=logfire.ConsoleOptions(verbose=True))

    async def handle_case(eval_case: EvalCase[TimeRangeInputs, TimeRangeAgentResponse]):
        result = await judge_time_range_case(inputs=eval_case.inputs, output=eval_case.output)
        eval_case.record_label('reasonable', 'yes' if result else 'no')

    cases = TimeRangeDataset.from_yaml().rows
    async with evaluation(infer_time_range, handler=handle_case) as my_eval:
        for case_data in cases:
            my_eval.case(name=case_data.name, inputs=case_data.inputs, handler=handle_case)

    my_eval.print_report(include_input=True, include_output=True)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())

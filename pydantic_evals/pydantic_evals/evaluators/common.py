from __future__ import annotations as _annotations

from pydantic_ai import models

from .context import EvaluatorContext
from .spec import EvaluatorResult


async def llm_judge(
    ctx: EvaluatorContext[object, object, object],
    rubric: str,
    model: models.KnownModelName = 'gpt-4o',
    include_input: bool = False,
) -> EvaluatorResult:
    """Judge whether the output of a language model meets the criteria of a provided rubric."""
    if include_input:
        from .llm_as_a_judge import judge_input_output

        grading_output = await judge_input_output(ctx.inputs, ctx.output, rubric, model)
    else:
        from .llm_as_a_judge import judge_output

        grading_output = await judge_output(ctx.output, rubric, model)
    return EvaluatorResult(value=grading_output.pass_, reason=grading_output.reason)


async def is_instance(ctx: EvaluatorContext[object, object, object], type_name: str) -> EvaluatorResult:
    """Check if the output is an instance of a type with the given name."""
    output = ctx.output
    for cls in type(output).__mro__:
        if cls.__name__ == type_name or cls.__qualname__ == type_name:
            return EvaluatorResult(value=True)

    reason = f'output is of type {type(output).__name__}'
    if type(output).__qualname__ != type(output).__name__:
        reason += f' (qualname: {type(output).__qualname__})'
    return EvaluatorResult(value=False, reason=reason)


# TODO: Add a list of default evaluators or similar, and/or a decorator for registering them
# DEFAULT_EVALUATORS: list[EvaluatorFunction[object, object, object]] = [llm_judge, is_instance]

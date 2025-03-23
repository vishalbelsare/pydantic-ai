# Pydantic Evals

Pydantic Evals is a powerful evaluation framework designed to help you systematically test and evaluate the quality of your code, especially when working with LLM-powered applications.

!!! note "In Beta"
    Pydantic Evals support was [introduced](https://github.com/pydantic/pydantic-ai/pull/935) in v0.0.44 and is currently in beta. The API is subject to change. The documentation is incomplete.


## Installation

To install the Pydantic Evals package, you can use:

```bash
pip/uv-add pydantic-evals
```

## Core Concepts

### Datasets and Cases

The foundation of Pydantic Evals is the concept of datasets and test cases:

- [`Case`][pydantic_evals.Case]: A single test scenario consisting of inputs, expected outputs, metadata, and optional evaluators.
- [`Dataset`][pydantic_evals.Case]: A collection of test cases designed to evaluate a specific task or function.

```python {title="simple_eval_dataset.py"}
from pydantic_evals import Case, Dataset

case1 = Case(  # (1)!
    name='simple_case',
    inputs='What is the capital of France?',
    expected_output='Paris',
    metadata={'difficulty': 'easy'},
)

dataset = Dataset(cases=[case1])  # (2)!
```

1. Create a test case
2. Create a dataset from cases, in most real world cases you would have multiple cases `#!python Dataset(cases=[case1, case2, case3])`

### Evaluators

Evaluators are the components that analyze and score the results of your task when tested against a case.

Pydantic Evals includes several built-in evaluators and allows you to create custom evaluators:

```python {title="simple_eval_evaluator.py"}
from functools import partial

from simple_eval_dataset import dataset

from pydantic_evals.evaluators.common import is_instance  # (1)!
from pydantic_evals.evaluators.context import EvaluatorContext

dataset.add_evaluator(partial(is_instance, type_name='str'))  # (2)!


async def my_evaluator(ctx: EvaluatorContext[str, str]) -> float:  # (3)!
    if ctx.output == ctx.expected_output:
        return 1.0
    elif (
        isinstance(ctx.output, str)
        and ctx.expected_output.lower() in ctx.output.lower()
    ):
        return 0.8
    else:
        return 0.0


dataset.add_evaluator(my_evaluator)
```
1. Import built-in evaluators, here we import [`is_instance`][pydantic_evals.evaluators.is_instance].
2. Add built-in evaluators [`is_instance`][pydantic_evals.evaluators.is_instance] to the dataset.
3. Create a custom evaluator function that takes an [`EvaluatorContext`][pydantic_evals.evaluators.context.EvaluatorContext] and returns a simple score.

### Evaluation Process

The evaluation process involves running a task against all cases in a dataset:

Putting the above two examples together and using the more declarative `evaluators` kwarg to `Dataset`:

```python {title="simple_eval_complete.py"}
from functools import partial

import logfire

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators.common import is_instance
from pydantic_evals.evaluators.context import EvaluatorContext

logfire.configure()

case1 = Case(  # (1)!
    name='simple_case',
    inputs='What is the capital of France?',
    expected_output='Paris',
    metadata={'difficulty': 'easy'},
)


def my_evaluator(ctx: EvaluatorContext[str, str]) -> float:
    if ctx.output == ctx.expected_output:
        return 1.0
    elif (
        isinstance(ctx.output, str)
        and ctx.expected_output.lower() in ctx.output.lower()
    ):
        return 0.8
    else:
        return 0.0


dataset = Dataset(
    cases=[case1],
    evaluators=[partial(is_instance, type_name='str'), my_evaluator],
)


async def guess_city(question: str) -> str:
    return 'Paris'


report = dataset.evaluate_sync(guess_city)  # (2)!
report.print(include_input=True, include_output=True)  # (3)!
```

# TODO complete from here on:

## LLM as a Judge

Pydantic Evals integrates seamlessly with LLMs for both evaluation and dataset generation:

You can use LLMs to evaluate the quality of outputs:

```python
from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output


async def judge_case(inputs, output):
    """Judge the output based on a rubric."""
    rubric = 'The output should be accurate, complete, and relevant to the inputs.'
    return await judge_input_output(inputs, output, rubric)
```

### Generating Test Datasets

Pydantic Evals allows you to generate test datasets using LLMs:

```python
from pydantic_evals.examples import generate_dataset

dataset = await generate_dataset(
    path="my_test_cases.yaml",
    inputs_type=MyInputs,
    output_type=MyOutput,
    metadata_type=dict,
    n_examples=5
)
```

## Advanced Usage

### Saving and Loading Datasets

Datasets can be saved to and loaded from files (YAML or JSON format):

```python
# Save dataset to file
dataset.to_file("test_cases.yaml")

# Load dataset from file
loaded_dataset = Dataset.from_file("test_cases.yaml", custom_evaluators=[my_evaluator])
```

### Parallel Evaluation

You can control concurrency during evaluation:

```python
# Run evaluation with limited concurrency
report = await dataset.evaluate(my_task, max_concurrency=10)
```

### OpenTelemetry Integration

Pydantic Evals integrates with OpenTelemetry for tracing and metrics:

```python
# Evaluation is automatically traced with spans
# Metrics are collected and can be exported
```

## Example: Time Range Evaluation

Here's a complete example of using Pydantic Evals for evaluating a time range inference function:

```python
from pydantic import BaseModel

from pydantic_evals import Dataset
from pydantic_evals.evaluators.common import is_instance, llm_judge
from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output


# Define input and output models
class TimeRangeInputs(BaseModel):
    query: str
    context: str | None = None


class TimeRangeResponse(BaseModel):
    start_time: str | None = None
    end_time: str | None = None
    error: str | None = None


# Create judge function
async def judge_time_range_case(inputs, output):
    rubric = (
        'The output should be a reasonable time range to select for the given inputs.'
    )
    return await judge_input_output(inputs, output, rubric)


# Load dataset
dataset = Dataset.from_file(
    'test_cases.yaml', custom_evaluators=[is_instance, llm_judge]
)


# Add custom evaluator
async def time_range_evaluator(ctx):
    result = await judge_time_range_case(inputs=ctx.inputs, output=ctx.output)
    return {
        'is_reasonable': 'yes' if result.pass_ else 'no',
        'accuracy': result.score,
    }


dataset.add_evaluator(time_range_evaluator)


# Define function to test
async def infer_time_range(inputs: TimeRangeInputs) -> TimeRangeResponse:
    # Your implementation here
    pass


# Run evaluation
report = dataset.evaluate_sync(infer_time_range)
report.print(include_input=True, include_output=True)
```

## Best Practices

### Creating Effective Test Cases

- Include a diverse range of inputs covering both common and edge cases
- Provide clear expected outputs for deterministic evaluation
- Include metadata to help categorize and filter test cases
- Use descriptive names for test cases

### Designing Robust Evaluators

- Combine multiple evaluation metrics for a comprehensive assessment
- Consider both objective metrics and subjective quality when appropriate
- Customize evaluators for your specific domain
- Use LLM-based evaluators for complex semantic judgments

### Interpreting Results

- Look beyond aggregate scores to understand individual case failures
- Use the detailed reports to identify patterns in errors
- Compare results across multiple model versions or implementations
- Pay attention to both precision and recall in your evaluation metrics

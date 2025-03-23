# Pydantic Evals

Pydantic Evals is a powerful evaluation framework designed to help you systematically test and evaluate the quality of your code, especially when working with LLM-powered applications.

## Installation

To install the Pydantic Evals package, you can use:

```bash
pip install pydantic-evals
# or
uv add pydantic-evals
```

## Core Concepts

### Datasets and Cases

The foundation of Pydantic Evals is the concept of datasets and test cases:

- **Case**: A single test scenario consisting of inputs, expected outputs, metadata, and optional evaluators.
- **Dataset**: A collection of test cases designed to evaluate a specific task or function.

```python
from pydantic_evals import Case, Dataset

# Create individual test cases
case1 = Case(
    name="simple_case",
    inputs={"query": "What is the capital of France?"},
    expected_output={"answer": "Paris"},
    metadata={"difficulty": "easy"}
)

# Create a dataset from cases
dataset = Dataset(cases=[case1, case2, case3])
```

### Evaluators

Evaluators are the components that analyze and score the results of your task when tested against a case:

- **Evaluator**: A class that encapsulates the logic for evaluating the output of a function against expected results.
- **EvaluatorSpec**: A specification for how an evaluator should be configured and run.

Pydantic Evals includes several built-in evaluators and allows you to create custom evaluators:

```python
from pydantic_evals.evaluators.common import is_instance, llm_judge

# Add evaluators to the dataset
dataset.add_evaluator(is_instance)
dataset.add_evaluator(llm_judge)

# Create a custom evaluator function
async def my_evaluator(ctx):
    # Custom evaluation logic
    return {
        "accuracy": 0.95,
        "reasons": ["Matches expected format", "Contains all required information"]
    }

dataset.add_evaluator(my_evaluator)
```

### Evaluation Process

The evaluation process involves running a task against all cases in a dataset and collecting metrics:

```python
async def my_task(inputs):
    # Your implementation here
    return {"answer": "Paris"}

# Run evaluation and get a report
report = await dataset.evaluate(my_task)

# Print results
report.print(include_input=True, include_output=True)
```

## Working with LLMs

Pydantic Evals integrates seamlessly with LLMs for both evaluation and dataset generation:

### LLM as a Judge

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

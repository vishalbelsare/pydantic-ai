import asyncio
from pathlib import Path
from typing import Any

from logfire.experimental.query_client import AsyncLogfireQueryClient
from pydantic import TypeAdapter
from pydantic_evals.assessments.common import is_instance, llm_rubric
from pydantic_evals.dataset import Dataset, DatasetRow

from demo.step_4_evaluate.app_v4_agent_updated import TimeRangeInputs, TimeRangeResponse
from demo.util.tokens import get_app_read_token

read_token = get_app_read_token()

successes_query = """
SELECT
    start_timestamp as now,
    attributes->'all_messages_events'->1->>'content' AS prompt,
    attributes->'final_result' AS expected_output
FROM records
WHERE
    span_name = 'agent run'
    AND attributes->>'agent_name' = 'time_range_agent'
    AND attributes->'final_result'->'error_message' IS NULL
LIMIT 3
"""

errors_query = """
SELECT
    start_timestamp as now,
    attributes->'all_messages_events'->1->>'content' AS prompt,
    attributes->'final_result' AS expected_output
FROM records
WHERE
    span_name = 'agent run'
    AND attributes->>'agent_name' = 'time_range_agent'
    AND attributes->'final_result'->'error_message' IS NOT NULL
LIMIT 3
"""

response_adapter = TypeAdapter[TimeRangeResponse](TimeRangeResponse)


def get_dataset_rows(
    name_prefix: str, data: list[dict[str, Any]]
) -> list[DatasetRow[TimeRangeInputs, TimeRangeResponse, dict[str, Any]]]:
    dataset_rows: list[
        DatasetRow[TimeRangeInputs, TimeRangeResponse, dict[str, Any]]
    ] = []
    for i, row in enumerate(data, 1):
        dataset_row = DatasetRow[TimeRangeInputs, TimeRangeResponse, dict[str, Any]](
            name=f"{name_prefix}_{i}",
            inputs=TimeRangeInputs(prompt=row["prompt"], now=row["now"]),
            metadata={},
            expected_output=response_adapter.validate_python(row["expected_output"]),
        )
        dataset_rows.append(dataset_row)
    return dataset_rows


class TimeRangeDataset(Dataset[TimeRangeInputs, TimeRangeResponse, dict[str, Any]]):
    pass


async def main():
    client: AsyncLogfireQueryClient
    async with AsyncLogfireQueryClient(
        read_token=read_token, base_url="http://localhost:8000"
    ) as client:
        successes = await client.query_json_rows(successes_query)
        success_rows = get_dataset_rows("success", successes["rows"])

        errors = await client.query_json_rows(errors_query)
        error_rows = get_dataset_rows("error", errors["rows"])

    dataset = TimeRangeDataset(data=success_rows + error_rows, assessments=[])
    dataset_path = Path(__file__).parent / "retrieved_test_cases.yaml"
    dataset.generate_dataset_files(dataset_path, scorers=(llm_rubric, is_instance))
    dataset.save(dataset_path)


asyncio.run(main())

"""
  assertions:
    - call: llm_rubric
      rubric: output should use offset -07:00
"""

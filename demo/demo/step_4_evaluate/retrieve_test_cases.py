import asyncio
from pathlib import Path
from typing import Any

from logfire.experimental.query_client import AsyncLogfireQueryClient
from pydantic import TypeAdapter
from pydantic_evals.datasets import Dataset, SerializedDatasetRow

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
    AND attributes->'final_result'->'error_message' IS NOT NULL
LIMIT 3
"""

response_adapter = TypeAdapter[TimeRangeResponse](TimeRangeResponse)

def get_dataset_rows(name_prefix: str, data: list[dict[str, Any]]) -> list[SerializedDatasetRow[TimeRangeInputs, TimeRangeResponse, dict[str, Any]]]:
    dataset_rows: list[
        SerializedDatasetRow[TimeRangeInputs, TimeRangeResponse, dict[str, Any]]
    ] = []
    for i, row in enumerate(data, 1):
        dataset_row = SerializedDatasetRow[
            TimeRangeInputs, TimeRangeResponse, dict[str, Any]
        ](
            name=f"{name_prefix}_{i}",
            inputs=TimeRangeInputs(prompt=row["prompt"], now=row["now"]),
            metadata={},
            expected_output=response_adapter.validate_python(
                row["expected_output"]
            ),
        )
        dataset_rows.append(dataset_row)
    return dataset_rows

async def main():

    client: AsyncLogfireQueryClient
    async with AsyncLogfireQueryClient(read_token=read_token) as client:
        successes = await client.query_json_rows(successes_query)
        success_rows = get_dataset_rows("success", successes["rows"])
        
        errors = await client.query_json_rows(errors_query)
        error_rows = get_dataset_rows("error", errors["rows"])

    dataset = Dataset[TimeRangeInputs, TimeRangeResponse, dict[str, Any]](
        rows=success_rows + error_rows
    )
    dataset.save(Path(__file__).parent / "retrieved_test_cases.yaml")


asyncio.run(main())

"""
  assertions:
    - call: llm_rubric
      rubric: output should use offset -07:00
"""
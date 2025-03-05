import asyncio
import os
from pathlib import Path
from typing import Any

from logfire.experimental.query_client import AsyncLogfireQueryClient
from pydantic import TypeAdapter
from pydantic_evals.datasets import Dataset, DatasetRow

from demo.models.time_range_v1 import TimeRangeInputs, TimeRangeResponse

successes_query = """
SELECT
    start_timestamp as now,
    attributes->'all_messages_events'->1->>'content' AS prompt,
    attributes->'final_result' AS expected_output
FROM records
WHERE
    span_name = 'agent run'
    AND attributes->'final_result'->'error_message' IS NULL
LIMIT 5
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
LIMIT 5
"""

read_token = os.environ["LOGFIRE_READ_TOKEN"]


async def main():
    response_adapter = TypeAdapter[TimeRangeResponse](TimeRangeResponse)

    dataset_rows: list[
        DatasetRow[TimeRangeInputs, TimeRangeResponse, dict[str, Any]]
    ] = []
    client: AsyncLogfireQueryClient
    async with AsyncLogfireQueryClient(read_token=read_token) as client:
        successes = await client.query_json_rows(successes_query)
        errors = await client.query_json_rows(errors_query)

        for i, row in enumerate(successes["rows"] + errors["rows"], 1):
            dataset_row = DatasetRow[
                TimeRangeInputs, TimeRangeResponse, dict[str, Any]
            ](
                name=f"example_{i}",
                inputs=TimeRangeInputs(prompt=row["prompt"], now=row["now"]),
                metadata={},
                expected_output=response_adapter.validate_python(
                    row["expected_output"]
                ),
            )
            dataset_rows.append(dataset_row)

    dataset = Dataset[TimeRangeInputs, TimeRangeResponse, dict[str, Any]](
        rows=dataset_rows
    )
    dataset.save(Path(__file__).parent / "retrieved_test_cases.yaml")


asyncio.run(main())

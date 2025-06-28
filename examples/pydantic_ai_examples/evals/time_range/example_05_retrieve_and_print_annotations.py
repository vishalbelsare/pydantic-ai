import yaml
from logfire.experimental.query_client import LogfireQueryClient

client = LogfireQueryClient('pylf_v1_local_BZTQvBKwJjHwwGVJX3Jmn0CkTkKPg2nB169b2J7K3B31', base_url='http://localhost:8000')

query_results = client.query_json_rows("""
WITH all_annotations AS (
    SELECT
        *,
        attributes->>'logfire.feedback.id' AS feedback_id,
        attributes->>'logfire.feedback.deleted' AS is_deleted
    FROM records_all
    WHERE kind = 'annotation'
),
latest_annotations AS (
    SELECT DISTINCT ON (feedback_id)
        *
    FROM all_annotations
    ORDER BY feedback_id, created_at DESC
),
annotations AS (
    SELECT *
    FROM latest_annotations
    WHERE is_deleted IS NULL
)
SELECT
    a.attributes->'path' AS feedback_target,
    a.attributes->'UI Feedback' AS sentiment,
    a.attributes->'logfire.feedback.comment' AS comment,
    r.attributes->'all_messages_events' AS all_messages_events
FROM records r
JOIN annotations a ON r.trace_id = a.trace_id AND r.span_id = a.parent_span_id
""")['rows']

for item in query_results:
    # scrub the instructions to save tokens when looking at the output
    item['all_messages_events'][0]['content'] = '<agent instructions>'

print(yaml.safe_dump(query_results))

"""TODO: Remove this comment before merging..

# TODO: Add commit hash, timestamp, and other metadata to reports (like pytest-speed does), possibly in a dedicated struct
# TODO: Implement serialization of reports for later comparison, and add git hashes etc.
#   Note: I made pydantic_ai.evals.reports.EvalReport a BaseModel specifically to make this easier
# TODO: Implement a CLI with some pytest-like filtering API to make it easier to run only specific cases
"""

# TODO: Use span links to store scores, this provides a way to update them, add them later, etc.
# TODO: Add some kind of `eval_function` decorator, which ensures that calls to the function send eval-review-compatible data to logfire
from .evals import Evaluation, increment_eval_metric
from .reporting.reports import RenderNumberConfig, RenderValueConfig

__all__ = (
    'Evaluation',
    'increment_eval_metric',
    'RenderNumberConfig',
    'RenderValueConfig',
)

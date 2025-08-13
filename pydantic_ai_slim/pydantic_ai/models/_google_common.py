from __future__ import annotations

from typing import Annotated, Literal, cast

import pydantic
from typing_extensions import NotRequired, TypedDict

from pydantic_ai import usage


class _GeminiModalityTokenCount(TypedDict):
    """See <https://ai.google.dev/api/generate-content#modalitytokencount>."""

    modality: Annotated[
        Literal['MODALITY_UNSPECIFIED', 'TEXT', 'IMAGE', 'VIDEO', 'AUDIO', 'DOCUMENT'], pydantic.Field(alias='modality')
    ]
    token_count: Annotated[int, pydantic.Field(alias='tokenCount', default=0)]


class GeminiUsageMetaData(TypedDict, total=False):
    """See <https://ai.google.dev/api/generate-content#UsageMetadata>.

    The docs suggest all fields are required, but some are actually not required, so we assume they are all optional.
    """

    prompt_token_count: Annotated[int, pydantic.Field(alias='promptTokenCount')]
    candidates_token_count: NotRequired[Annotated[int, pydantic.Field(alias='candidatesTokenCount')]]
    total_token_count: Annotated[int, pydantic.Field(alias='totalTokenCount')]
    cached_content_token_count: NotRequired[Annotated[int, pydantic.Field(alias='cachedContentTokenCount')]]
    thoughts_token_count: NotRequired[Annotated[int, pydantic.Field(alias='thoughtsTokenCount')]]
    tool_use_prompt_token_count: NotRequired[Annotated[int, pydantic.Field(alias='toolUsePromptTokenCount')]]
    prompt_tokens_details: NotRequired[
        Annotated[list[_GeminiModalityTokenCount], pydantic.Field(alias='promptTokensDetails')]
    ]
    cache_tokens_details: NotRequired[
        Annotated[list[_GeminiModalityTokenCount], pydantic.Field(alias='cacheTokensDetails')]
    ]
    candidates_tokens_details: NotRequired[
        Annotated[list[_GeminiModalityTokenCount], pydantic.Field(alias='candidatesTokensDetails')]
    ]
    tool_use_prompt_tokens_details: NotRequired[
        Annotated[list[_GeminiModalityTokenCount], pydantic.Field(alias='toolUsePromptTokensDetails')]
    ]


def metadata_as_request_usage(metadata: GeminiUsageMetaData | None) -> usage.Usage:
    if metadata is None:
        return usage.Usage(requests=1)
    details: dict[str, int] = {}
    if cached_content_token_count := metadata.get('cached_content_token_count', 0):
        details['cached_content_tokens'] = cached_content_token_count

    if thoughts_token_count := metadata.get('thoughts_token_count', 0):
        details['thoughts_tokens'] = thoughts_token_count

    if tool_use_prompt_token_count := metadata.get('tool_use_prompt_token_count', 0):
        details['tool_use_prompt_tokens'] = tool_use_prompt_token_count

    input_audio_tokens = 0
    output_audio_tokens = 0
    cache_audio_read_tokens = 0
    for key, metadata_details in metadata.items():
        if key.endswith('_details') and metadata_details:
            metadata_details = cast(list[_GeminiModalityTokenCount], metadata_details)
            suffix = key.removesuffix('_details')
            for detail in metadata_details:
                modality = detail['modality']
                details[f'{modality.lower()}_{suffix}'] = value = detail.get('token_count', 0)
                if value and modality == 'AUDIO':
                    if key == 'prompt_tokens_details':
                        input_audio_tokens = value
                    elif key == 'candidates_tokens_details':
                        output_audio_tokens = value
                    elif key == 'cache_tokens_details':  # pragma: no branch
                        cache_audio_read_tokens = value

    return usage.Usage(
        requests=1,
        input_tokens=metadata.get('prompt_token_count', 0),
        output_tokens=metadata.get('candidates_token_count', 0),
        cache_read_tokens=cached_content_token_count,
        input_audio_tokens=input_audio_tokens,
        output_audio_tokens=output_audio_tokens,
        cache_audio_read_tokens=cache_audio_read_tokens,
        details=details,
    )

from __future__ import annotations

from collections.abc import AsyncIterator, Hashable
from contextlib import asynccontextmanager
from copy import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import TypeAdapter
from pydantic_core import to_json, to_jsonable_python

from . import (

    KnownModelName,
    Model,
    ModelRequestParameters,
    StreamedResponse,
)
from .wrapper import WrapperModel
from .._parts_manager import ModelResponsePartsManager
from ..messages import (
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
)
from ..settings import ModelSettings
from ..tools import ToolDefinition
from ..usage import Usage

CacheMode = Literal['read', 'write', 'readwrite', 'error', 'disabled']
"""
* In 'readwrite' mode (the default), requests will be made if no cached response is found, and their responses will be cached.
* In 'read' mode, if a request is cached its response will be used, but responses to any uncached requests will not be cached.
* In 'write' mode, any cached responses will be ignored and requests will always be made. Cached responses will be replaced if the same request is made again.
* In 'error' mode, the cache will raise an error if you make any request that doesn't have a cached response. This is useful for testing.
* In 'disabled' mode, the cache is disabled and requests will always be made.
"""



@dataclass
class PartManagerCall:
    call: str
    kwargs: dict[str, Any]


@dataclass
class RequestInputs:
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters

    def __post_init__(self):
        # need to copy the messages because they get mutated during request handling 🤮
        self.messages = copy(self.messages)

    @cached_property
    def _equality_key(self) -> str:
        # TODO: Maybe convert this to a str so we can use a hashmap for lookups
        #   Alternatively, convert lists to tuples and dicts to tuples of pairs so they can be hashed
        return to_json(_recursively_remove_datetimes(asdict(self)), indent=2).decode()

    def __hash__(self) -> int:
        return hash(self._equality_key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RequestInputs):
            return self._equality_key == other._equality_key
        return False


@dataclass
class CachedRequestOutputs:
    response: ModelResponse
    usage: Usage


@dataclass
class CachedStreamRequestOutputs:
    model_name: str
    timestamp: datetime
    events: list[PartManagerCall | Usage]


def _recursively_remove_datetimes(inputs: Any) -> Any:
    if isinstance(inputs, dict):
        return {k: _recursively_remove_datetimes(v) for k, v in inputs.items() if not isinstance(v, datetime)}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(inputs, list):
        return [_recursively_remove_datetimes(v) for v in inputs if not isinstance(v, datetime)]  # pyright: ignore[reportUnknownVariableType]
    return inputs


REQUEST_INPUTS_ADAPTER = TypeAdapter(RequestInputs)
CACHE_ADAPTER = TypeAdapter(list[tuple[RequestInputs, CachedRequestOutputs]])
CACHE_STREAM_ADAPTER = TypeAdapter(list[tuple[RequestInputs, CachedStreamRequestOutputs]])



@dataclass(init=False)
class ModelResponseCache:
    path: Path
    """The path to use for storage"""
    stream_path: Path
    """The path to use for stream storage"""
    cached_requests: dict[RequestInputs, CachedRequestOutputs] = field(repr=False)
    """The requests that have been cached"""
    cached_stream_requests: dict[RequestInputs, CachedStreamRequestOutputs] = field(repr=False)
    """The stream requests that have been cached"""
    mode: CacheMode = 'readwrite'
    """The "mode" the cache is operating in. See `CacheMode` for more information."""

    def __init__(self, path: Path, stream_path: Path, mode: CacheMode = 'readwrite') -> None:
        CACHE_ADAPTER.rebuild(_types_namespace={'ToolDefinition': ToolDefinition})
        CACHE_STREAM_ADAPTER.rebuild(_types_namespace={'ToolDefinition': ToolDefinition})

        self.path = path
        self.stream_path = stream_path
        self.mode = mode

        if path.exists():
            items = CACHE_ADAPTER.validate_json(path.read_text())
            self.cached_requests = {inputs: outputs for inputs, outputs in items}
        else:
            self.cached_requests = {}

        if stream_path.exists():
            stream_items = CACHE_STREAM_ADAPTER.validate_json(stream_path.read_text())
            self.cached_stream_requests = {inputs: outputs for inputs, outputs in stream_items}
        else:
            self.cached_stream_requests = {}

    def get(self, inputs: RequestInputs) -> CachedRequestOutputs | None:
        if self.mode in {'write', 'disabled'}:
            print(f'Skipping cache read ({self.mode=})...')
            return None
        result = self.cached_requests.get(inputs)
        if self.mode == 'error' and result is None:
            raise ValueError(f'No cached response was found for {inputs}')
        print(f'Cache hit? {result is not None} ({self.mode=})')
        return result

    def set(self, inputs: RequestInputs, outputs: CachedRequestOutputs) -> None:
        existing_outputs = self.cached_requests.get(inputs)
        if self.mode in {'read', 'disabled', 'error'}:
            print(f'Skipping cache write ({self.mode=})...')
            return

        if existing_outputs is None or existing_outputs != outputs:
            print('Writing to cache...')
            self.cached_requests[inputs] = outputs
            items = list(self.cached_requests.items())
            self.path.write_bytes(to_json(items))
        else:
            print('Not writing to cache because inputs matched outputs...')

    def get_stream(self, inputs: RequestInputs) -> CachedStreamRequestOutputs | None:
        if self.mode in {'write', 'disabled'}:
            print(f'stream Skipping cache read ({self.mode=})...')
            return None
        result = self.cached_stream_requests.get(inputs)
        if self.mode == 'error' and result is None:
            raise ValueError(f'No cached stream response was found for {inputs}')
        print(f'stream Cache hit? {result is not None} ({self.mode=})')
        return result

    def set_stream(self, inputs: RequestInputs, outputs: CachedStreamRequestOutputs) -> None:
        existing_outputs = self.cached_stream_requests.get(inputs)
        if self.mode in {'read', 'disabled', 'error'}:
            print(f'stream Skipping cache write ({self.mode=})...')
            return

        if existing_outputs is None or existing_outputs != outputs:
            print('stream Writing to cache...')
            self.cached_stream_requests[inputs] = outputs
            items = list(self.cached_stream_requests.items())
            self.stream_path.write_text(to_jsonable_python(items))
        else:
            print('stream Not writing to cache because inputs matched outputs...')


@dataclass
class CacheModelResponsePartsManager(ModelResponsePartsManager):
    on_call: Callable[[PartManagerCall], None] | None = None

    def get_parts(self) -> list[ModelResponsePart]:
        return super().get_parts()

    def handle_text_delta(
        self,
        *,
        vendor_part_id: Hashable | None,
        content: str,
    ) -> ModelResponseStreamEvent:
        if self.on_call is not None:
            self.on_call(
                PartManagerCall('handle_text_delta', kwargs=dict(vendor_part_id=vendor_part_id, content=content))
            )
        return super().handle_text_delta(vendor_part_id=vendor_part_id, content=content)

    def handle_tool_call_delta(
        self,
        *,
        vendor_part_id: Hashable | None,
        tool_name: str | None,
        args: str | dict[str, Any] | None,
        tool_call_id: str | None,
    ) -> ModelResponseStreamEvent | None:
        if self.on_call is not None:
            self.on_call(
                PartManagerCall(
                    'handle_tool_call_delta',
                    kwargs=dict(
                        vendor_part_id=vendor_part_id, tool_name=tool_name, args=args, tool_call_id=tool_call_id
                    ),
                )
            )
        return super().handle_tool_call_delta(
            vendor_part_id=vendor_part_id, tool_name=tool_name, args=args, tool_call_id=tool_call_id
        )

    def handle_tool_call_part(
        self,
        *,
        vendor_part_id: Hashable | None,
        tool_name: str,
        args: str | dict[str, Any],
        tool_call_id: str | None = None,
    ) -> ModelResponseStreamEvent:
        if self.on_call is not None:
            self.on_call(
                PartManagerCall(
                    'handle_tool_call_part',
                    kwargs=dict(
                        vendor_part_id=vendor_part_id, tool_name=tool_name, args=args, tool_call_id=tool_call_id
                    ),
                )
            )
        return super().handle_tool_call_part(
            vendor_part_id=vendor_part_id, tool_name=tool_name, args=args, tool_call_id=tool_call_id
        )


@dataclass
class CacheUsage(Usage):
    on_add: Callable[[Usage], None] | None = None

    def __add__(self, other: Usage):
        if self.on_add and isinstance(other, Usage):
            self.on_add(other)
        return super().__add__(other)


# TODO: Make StreamedResponse implement a protocol, and have this implement the same protocol..
@dataclass
class RecordingStreamedResponse:
    wrapped: StreamedResponse
    cache_events: list[PartManagerCall | Usage] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.wrapped._parts_manager = CacheModelResponsePartsManager(on_call=self.cache_events.append)  # pyright: ignore[reportPrivateUsage]
        self.wrapped._usage = CacheUsage(on_add=self.cache_events.append)  # pyright: ignore[reportPrivateUsage]

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream the response as an async iterable of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s."""
        return self.wrapped.__aiter__()

    def get(self) -> ModelResponse:
        """Build a [`ModelResponse`][pydantic_ai.messages.ModelResponse] from the data received from the stream so far."""
        return self.wrapped.get()

    def usage(self) -> Usage:
        """Get the usage of the response so far. This will not be the final usage until the stream is exhausted."""
        return self.wrapped.usage()

    @property
    def model_name(self) -> str:
        return self.wrapped.model_name

    @property
    def timestamp(self) -> datetime:
        return self.wrapped.timestamp


@dataclass
class CachedStreamedResponse(StreamedResponse):
    cache_events: list[PartManagerCall | Usage]
    cached_model_name: str
    cached_timestamp: datetime

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for cache_event in self.cache_events:
            if isinstance(cache_event, PartManagerCall):
                maybe_event = getattr(self._parts_manager, cache_event.call)(**cache_event.kwargs)
                if maybe_event is not None:
                    yield maybe_event
            elif isinstance(cache_event, Usage):
                self._usage += cache_event

    @property
    def model_name(self) -> str:
        return self.cached_model_name

    @property
    def timestamp(self) -> datetime:
        return self.cached_timestamp


@dataclass
class RecordingModel(WrapperModel):
    """Model which wraps another model so that requests are stored and loaded from a cache."""

    cache: ModelResponseCache

    def __init__(
        self,
        wrapped: Model | KnownModelName,
        path: Path = Path('pydantic_ai_requests.json'),
        stream_path: Path = Path('pydantic_ai_stream_requests.json'),
        mode: Literal['read', 'write', 'readwrite', 'error', 'disabled'] = 'readwrite',
    ) -> None:
        super().__init__(wrapped)
        self.cache = ModelResponseCache(path=path, stream_path=stream_path, mode=mode)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        inputs = RequestInputs(messages, model_settings, model_request_parameters)
        cached_outputs = self.cache.get(inputs)
        if cached_outputs:
            return cached_outputs.response, cached_outputs.usage
        response, usage = await super().request(messages, model_settings, model_request_parameters)
        self.cache.set(inputs, CachedRequestOutputs(response, usage))
        return response, usage

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        inputs = RequestInputs(messages, model_settings, model_request_parameters)
        cached_outputs = self.cache.cached_stream_requests.get(inputs)
        if cached_outputs:
            yield CachedStreamedResponse(
                cache_events=cached_outputs.events,
                cached_model_name=cached_outputs.model_name,
                cached_timestamp=cached_outputs.timestamp,
            )
            return

        async with super().request_stream(messages, model_settings, model_request_parameters) as response_stream:
            recording_response_stream = RecordingStreamedResponse(response_stream)
            yield recording_response_stream  # type: ignore  # TODO: Use a protocol so this isn't a type error

        self.cache.set_stream(
            inputs,
            CachedStreamRequestOutputs(
                model_name=response_stream.model_name,
                timestamp=response_stream.timestamp,
                events=recording_response_stream.cache_events,
            ),
        )

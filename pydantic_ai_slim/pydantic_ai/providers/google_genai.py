from __future__ import annotations as _annotations

import os

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers import Provider

try:
    from google import genai
except ImportError as _import_error:
    raise ImportError(
        'Please install the `google-genai` package to use the Google Vertex AI provider, '
        'you can use the `gemini` optional group — `pip install "pydantic-ai-slim[gemini]"`'
    ) from _import_error


class GoogleGenAIProvider(Provider[genai.Client]):
    """Provider for Google API."""

    @property
    def name(self):
        return 'google-gla'

    @property
    def base_url(self) -> str:
        return 'https://generativelanguage.googleapis.com/v1beta/models/'

    @property
    def client(self) -> genai.Client:
        return self._client

    def __init__(self, api_key: str | None = None, client: genai.Client | None = None) -> None:
        """Create a new Google GLA provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `GEMINI_API_KEY` environment variable
                will be used if available.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise UserError(
                'Set the `GEMINI_API_KEY` environment variable or pass it via `GoogleGLAProvider(api_key=...)`'
                'to use the Google GLA provider.'
            )

        self._client = client or genai.Client(api_key=api_key)

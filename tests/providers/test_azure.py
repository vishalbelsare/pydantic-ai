import pytest
from inline_snapshot import snapshot
from openai import AsyncAzureOpenAI

from pydantic_ai.models.openai import OpenAIModel

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.azure import AzureProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_azure_provider():
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_version='2023-03-15-preview',
        api_key='1234567890',
    )
    assert isinstance(provider, AzureProvider)
    assert provider.name == 'azure'
    assert provider.base_url == snapshot('https://project-id.openai.azure.com/')
    assert isinstance(provider.client, AsyncAzureOpenAI)


def test_azure_provider_with_openai_model():
    model = OpenAIModel(
        model_name='gpt-4o',
        provider=AzureProvider(
            azure_endpoint='https://project-id.openai.azure.com/',
            api_version='2023-03-15-preview',
            api_key='1234567890',
        ),
    )
    assert isinstance(model, OpenAIModel)
    assert isinstance(model.client, AsyncAzureOpenAI)

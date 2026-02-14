import os
from pathlib import Path

import pytest

from dify_plugin.config.integration_config import IntegrationConfig
from dify_plugin.core.entities.plugin.request import (
    ModelActions,
    ModelInvokeLLMRequest,
    PluginInvokeType,
)
from dify_plugin.entities.model import ModelType
from dify_plugin.entities.model.llm import LLMResultChunk
from dify_plugin.integration.run import PluginRunner


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable is required")
    return value


def _test_models() -> list[str]:
    models = os.getenv("VOLCENGINE_TEST_MODELS", "").strip()
    if models:
        return [m.strip() for m in models.split(",") if m.strip()]
    return [os.getenv("VOLCENGINE_BASE_MODEL", "Doubao-Seed-1.8")]


@pytest.mark.parametrize("model_name", _test_models())
def test_llm_invoke(model_name: str) -> None:
    api_key = _required_env("VOLCENGINE_API_KEY")
    endpoint_id = _required_env("VOLCENGINE_ENDPOINT_ID")

    plugin_path = os.getenv("PLUGIN_FILE_PATH")
    if not plugin_path:
        plugin_path = str(Path(__file__).parent.parent)

    payload = ModelInvokeLLMRequest(
        user_id="test_user",
        provider="volcengine_maas",
        model_type=ModelType.LLM,
        model=model_name,
        credentials={
            "auth_method": "api_key",
            "volc_api_key": api_key,
            "endpoint_id": endpoint_id,
            "base_model_name": model_name,
            "volc_region": os.getenv("VOLCENGINE_REGION", "cn-beijing"),
            "api_endpoint_host": os.getenv("VOLCENGINE_API_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3"),
        },
        prompt_messages=[{"role": "user", "content": "Say hello in one word."}],
        model_parameters={"max_tokens": 64},
        stop=None,
        tools=None,
        stream=True,
    )

    with PluginRunner(config=IntegrationConfig(), plugin_package_path=plugin_path) as runner:
        results: list[LLMResultChunk] = []
        for result in runner.invoke(
            access_type=PluginInvokeType.Model,
            access_action=ModelActions.InvokeLLM,
            payload=payload,
            response_type=LLMResultChunk,
        ):
            results.append(result)

        assert len(results) > 0, f"No results received for model {model_name}"

        full_content = "".join(
            r.delta.message.content for r in results if r.delta.message.content
        )
        assert len(full_content) > 0, f"Empty content for model {model_name}"

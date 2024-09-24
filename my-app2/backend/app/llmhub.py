from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from typing import Dict
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


class TSIEmbedding(OpenAIEmbedding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._query_engine = self._text_engine = self.model_name


def llm_config_from_env() -> Dict:
    from llama_index.core.constants import DEFAULT_TEMPERATURE

    model = os.getenv("MODEL", DEFAULT_MODEL)
    temperature = os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
    max_tokens = os.getenv("LLM_MAX_TOKENS")
    api_key = os.getenv("T_SYSTEMS_LLMHUB_API_KEY")
    api_base = os.getenv("T_SYSTEMS_LLMHUB_BASE_URL")

    config = {
        "model": model,
        "api_key": api_key,
        "api_base": api_base,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
    }
    return config


def embedding_config_from_env() -> Dict:
    from llama_index.core.constants import DEFAULT_EMBEDDING_DIM

    model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    dimension = os.getenv("EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM)
    api_key = os.getenv("T_SYSTEMS_LLMHUB_API_KEY")
    api_base = os.getenv("T_SYSTEMS_LLMHUB_BASE_URL")

    config = {
        "model_name": model,
        "dimension": int(dimension) if dimension is not None else None,
        "api_key": api_key,
        "api_base": api_base,
    }
    return config


def init_llmhub():
    from llama_index.llms.openai_like import OpenAILike

    llm_configs = llm_config_from_env()
    embedding_configs = embedding_config_from_env()

    #Settings.embed_model = TSIEmbedding(**embedding_configs)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Ollama(model="llama3:latest", request_timeout=120.0)
+
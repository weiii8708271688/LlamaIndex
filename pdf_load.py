"""import nest_asyncio
nest_asyncio.apply()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

parser = LlamaParse(
    api_key="llx-...",
    result_type="markdown",
    verbose=True,
)




def gen_documents(title):
    documents = parser.load_data("./my_file.pdf")"""

from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
)
from llama_index.core import SummaryIndex
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.callbacks import CallbackManager
import nest_asyncio
nest_asyncio.apply()
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import os 
from pathlib import Path
from llama_index.embeddings.llamafile import LlamafileEmbedding
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import load_index_from_storage, StorageContext


Settings.llm = Ollama(model="llama3:latest", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
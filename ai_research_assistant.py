import arxiv
import os
import logging
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline
import asyncio, nest_asyncio
import logging
import requests
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import threading
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
logging.basicConfig(level=logging.INFO, filename='ai_research_assistant.log', filemode='a+')
from agent_a import AgentA
from agent_b import AgentB
from agent_c import AgentC


paper_docs = None
class AIResearchAssistant:
    def __init__(self):
        nest_asyncio.apply()
        load_dotenv()
        op = input("要開啟OPENAI LLM嗎？(y/n)")
        if op == 'y':
            self.llm = OpenAI(temperature=0, model="gpt-3.5-turbo", api_key=os.getenv('OPENAI_API_KEY'))
        else:
            self.llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
        Settings.llm = self.llm
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.parser = LlamaParse(
            api_key=os.getenv('LLAMA_PARSE_API_KEY'),
            result_type="markdown",
            verbose=True,
        )
        self.title_list = os.listdir('./data/pdf')
        
        self.chat_history = []

        self.agent_a = AgentA(self.llm).create_agent()
        
        self.agent_b = AgentB(self.llm).create_agent()

        self.agent_c = AgentC(self.llm, self.agent_a, self.agent_b).create_agent()


    async def chat(self, user_input: str) -> str:
        print(f"收到了 User說 {user_input}")
        
        response = self.agent_c.chat(user_input)
        self.chat_history.append(f"User: {user_input}")
        self.chat_history.append(f"Agent: {response.response}")
        return response.response
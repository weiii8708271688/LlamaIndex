from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import os
from llama_index.llms.ollama import Ollama
import prompt.agent_b_prompt as agent_b_prompt

class AgentB:
    def __init__(self, llm):
        self.llm = llm
        self.ask_about_papers_tool = FunctionTool.from_defaults(
            fn=self.ask_about_papers,
            description="Ask questions about a specific academic paper. Requires the paper name (no need .pdf JUST paper name) and a question. The paper must exist in the ./storage directory."
        )
        

    def create_agent(self):
        

        b_agent = ReActAgent.from_tools(
            [self.ask_about_papers_tool],
            llm=self.llm,
            verbose=True,
            max_iterations=10,
            context=agent_b_prompt.FULL_PROMPT
        )
        agent_b = FunctionTool.from_defaults(
            fn=b_agent.chat,
            name="PaperQueryAssistant_AgentB",
            description="An advanced tool for querying and analyzing scientific paper information.",
            tool_metadata=ToolMetadata(
                name="PaperQueryAssistant_AgentB",
                description="""
                    This tool helps you retrieve detailed information about scientific papers. Usage:
                    1. Input your question or query in the format: {"message": "your question"}
                    2. Questions can include paper summaries, research methods, result analysis, or comparisons of related works.
                    3. The tool understands and responds to complex queries about research across various scientific domains.
                    4. For more details, you can ask follow-up questions.

                    Always include the 'message' parameter when using this tool, for example:
                    PaperQueryAssistant_AgentB({"message": "Tell me about the latest applications of AI in medical diagnostics"})
                """
            )
        )
        return agent_b
        return FunctionTool(
            query_engine=b_agent,
            metadata=ToolMetadata(
                name="Agent_B",
                description="A comprehensive academic paper analysis tool. It first checks for paper existence, then provides detailed analysis if the paper is found in the ./storage directory."
            )
        )
    
    def ask_about_papers(self, paper_name: str, question: str) -> str:
        filepath = f"./storage/{paper_name}"
        if not os.path.exists(filepath):
            return f"NO PAPER {paper_name} FOUND, Here are list of papers {os.listdir('./storage')}. You may find the paper you are looking for in the list above."
        self.vector_index_chunk = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=filepath),
        )
        self.vector_retriever_chunk = self.vector_index_chunk.as_retriever(similarity_top_k=5)

        # 創建 BM25 檢索器
        self.bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.vector_index_chunk.docstore, similarity_top_k=5
        )

        # 創建融合檢索器
        self.retriever = QueryFusionRetriever(
            [self.vector_retriever_chunk, self.bm25_retriever],
            similarity_top_k=5,
            num_queries=4,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
        )

        # 創建檢索查詢引擎
        self.query_engine_chunk = RetrieverQueryEngine.from_args(self.retriever)
        response = self.query_engine_chunk.query(question)
        return response.response

    
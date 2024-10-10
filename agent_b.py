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

class AgentB:
    def __init__(self, llm):
        self.llm = llm
        self.ask_about_papers_tool = FunctionTool.from_defaults(
            fn=self.ask_about_papers,
            description="Ask questions about a specific academic paper. Requires the paper name (no need .pdf JUST paper name) and a question. The paper must exist in the ./storage directory."
        )
        self.generate_paper_search_prompt_tool = FunctionTool.from_defaults(
            fn=self.generate_paper_search_prompt,
            description="Query the paper database to check for paper existence and get file names. Always use this first. "
        )

    def create_agent(self):
        system_prompt = """
        You are an advanced AI agent specialized in academic paper analysis and information retrieval. Follow these steps strictly:

        1. ALWAYS start by using the generate_paper_search_prompt tool to check if the paper exists in the database.
        2. If the paper exists (response is not 'NO'), ALWAYS proceed to use the ask_about_papers tool for in-depth analysis or to answer specific questions about that paper. You must provide both the paper name and the user's question to the ask_about_papers tool.
        3. If the paper doesn't exist, inform the user and suggest they verify the paper title or search for alternative papers.

        Remember:
        - You must use both tools in this order for every query about a specific paper.
        - When using ask_about_papers, make sure to include both the paper name (from the generate_paper_search_prompt result) and the user's question.
        - Provide clear, concise, and academically-oriented responses.
        - If asked about multiple papers, repeat this process for each paper systematically.

        Your goal is to assist users in finding and understanding academic papers efficiently and accurately.
        """

        b_agent = ReActAgent.from_tools(
            [self.generate_paper_search_prompt_tool, self.ask_about_papers_tool],
            llm=self.llm,
            verbose=True,
            max_iterations=10,
            system_prompt=system_prompt
        )

        return QueryEngineTool(
            query_engine=b_agent,
            metadata=ToolMetadata(
                name="Retrieval_Agent_Tool",
                description="A comprehensive academic paper analysis tool. It first checks for paper existence, then provides detailed analysis if the paper is found in the ./storage directory."
            )
        )
    
    def ask_about_papers(self, paper_name: str, question: str) -> str:
        filepath = f"./storage/{paper_name}"
        if not os.path.exists(filepath):
            return f"NO PAPER {paper_name} FOUND"
        self.vector_index_chunk = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=filepath),
        )
        self.vector_retriever_chunk = self.vector_index_chunk.as_retriever(similarity_top_k=2)

        # 創建 BM25 檢索器
        self.bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.vector_index_chunk.docstore, similarity_top_k=2
        )

        # 創建融合檢索器
        self.retriever = QueryFusionRetriever(
            [self.vector_retriever_chunk, self.bm25_retriever],
            similarity_top_k=2,
            num_queries=4,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
        )

        # 創建檢索查詢引擎
        self.query_engine_chunk = RetrieverQueryEngine.from_args(self.retriever)
        response = self.query_engine_chunk.query(question)
        return response.response

    def generate_paper_search_prompt(self, user_input: str):
        self.title_list = os.listdir("data/pdf")
        chat_engine = SimpleChatEngine.from_defaults(llm=Ollama(model="llama3.1:latest", request_timeout=120.0))
        response = chat_engine.chat(f"The database contains the following paper files:\n{self.title_list}\n"
            f"The paper the user wants to find is: {user_input}\n"
            "If this paper doesn't exist, output only 'NO' without any other sentences.\n"
            "If the paper exists, output the complete file name(s) of the paper(s), "
            "without any path. If there are multiple files with the same name, "
            "output all matching complete file names.")
        
        if response.response != "NO":
            return f"Paper found: {response.response}. Proceed to ask questions about this paper."
        else:
            return "NO"
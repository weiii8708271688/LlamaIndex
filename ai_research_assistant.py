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
import arxiv

class AIResearchAssistant:
    def __init__(self):
        load_dotenv()
        self.llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
        Settings.llm = self.llm
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        self.parser = LlamaParse(
            api_key=os.getenv('LLAMA_PARSE_API_KEY'),
            result_type="markdown",
            verbose=True,
        )
        self.title_list = os.listdir('./data/pdf')
        self.node_parser = SentenceSplitter()
        
        self.chat_history = []
        self.download_paper_tool = FunctionTool.from_defaults(
            fn=self.download_paper,
            description="Downloads and indexes an academic paper from arXiv given its ID."
        )

        self.search_paper_tool = FunctionTool.from_defaults(
            fn=self.search_paper,
            description="Searches for academic papers on arXiv based on a query string."
        )

        self.ask_about_papers_tool = FunctionTool.from_defaults(
            fn=self.ask_about_papers,
            description="Ask questions about the academic papers."
        )
        self.a_agent = self.create_a_agent()
        self.b_agent = self.create_b_agent()

    def download_paper(self, paper_id: str) -> str:
        try:
            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results())
            
            pdf_path = f"data/pdf/{paper.title}.pdf"
            if os.path.exists(pdf_path):
                return f"Paper {paper.title} [{paper_id}] already downloaded before."
            paper.download_pdf(filename=pdf_path)
            paper_docs = self.parser.load_data(pdf_path)
            base_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=f"./storage/all_datas"),
            )
            nodes = self.node_parser.get_nodes_from_documents(paper_docs)
            base_index.insert_nodes(nodes)
            base_index.storage_context.persist(persist_dir=f"./storage/all_datas")
            
            self.title_list = os.listdir('./data/pdf')
            return f"Successfully downloaded paper {paper.title} [{paper_id}] and stored its index in the storage folder"
        except Exception as e:
            return f"Error downloading paper {paper_id}: {str(e)}"

    def search_paper(self, query: str) -> str:
        try:
            search = arxiv.Search(query=query, max_results=5)
            results = [f"Title: {result.title}, ID: {result.entry_id.split('/')[-1]}" for result in search.results()]
            return "\n".join(results) if results else "No papers found."
        except Exception as e:
            return f"Error searching for papers with query '{query}': {str(e)}"

    def ask_about_papers(self, question: str) -> str:
        base_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"./storage/all_datas"),
        )
        query_engine = base_index.as_query_engine(llm=Settings.llm)
        response = query_engine.query(question)
        return response.response

    def create_a_agent(self):
        a_agent = ReActAgent.from_tools(
            [self.download_paper_tool, self.search_paper_tool], 
            llm=self.llm, 
            verbose=True, 
            max_iterations=10, 
            system_prompt="An agent specialized in interacting with the arXiv repository for academic paper retrieval and information gathering."
        )
        return QueryEngineTool(
            query_engine=a_agent, 
            metadata=ToolMetadata(
                name="arXiv_research_assistant",
                description="A comprehensive tool for academic research assistance, specializing in arXiv repository interactions.There has two main tools to use: 1. download_paper: Download and index academic papers from arXiv based on their IDs. 2. search_paper: Search 5 papers for academic papers on arXiv based on a query string."
            )
        )

    def create_b_agent(self):
        b_agent = ReActAgent.from_tools(
            [self.ask_about_papers_tool],
            llm=self.llm,
            verbose=True, 
            max_iterations=10, 
            system_prompt="An agent specialized in analyzing and answering questions about academic papers.")
    
        return QueryEngineTool(
            query_engine=b_agent, 
            metadata=ToolMetadata(
                name="academic_paper_analysis_tool",
                description="A powerful academic paper analysis tool capable of answering in-depth questions about multiple academic papers.There has one main tool to use: ask_about_papers: Ask questions about the academic papers."
            )
        )

    def create_c_agent(self):
        return ReActAgent.from_tools(
            [self.a_agent, self.b_agent],
            llm=self.llm,
            verbose=True,
            max_iterations=10,
            system_prompt=f"""You are an intelligent conversation host and task manager. Your role is to facilitate effective communication between the user and specialized agents, while also being capable of handling tasks independently when appropriate.

            Key Responsibilities:
            1. Analyze and understand the user's requests or queries.
            2. Determine the most suitable approach for each task:
            a. Utilize Agent A (arXiv_research_assistant) for tasks related to searching and downloading papers from arXiv.
            b. Utilize Agent B (academic_paper_analysis_tool) for tasks related to analyzing and answering questions about downloaded papers.
            c. Handle simpler tasks or queries directly using your own knowledge and capabilities.
            3. Manage the flow of conversation, ensuring clarity and coherence.
            4. Synthesize information from multiple sources (agents, your own knowledge, user input) when necessary.
            5. Provide clear, concise, and relevant responses to the user.

            Important Notes:
            - If you want to use Agent A for a task, please go to Agent B first to check if the paper has been downloaded.
            - Not every task requires the use of Agent A or Agent B. Use your judgment to determine when to involve them.
            - You have access to the following chat history for context: {self.chat_history}
            - If a task seems simple or within your capabilities, you may choose to handle it directly without involving the specialized agents.
            - Always prioritize the user's needs and the efficiency of task completion.

            Remember, your goal is to provide the helpful and appropriate response to each user query, whether that involves coordinating with specialized agents or utilizing your own capabilities."""
        )

    def chat(self, user_input: str) -> str:
        """c_agent = self.create_c_agent()
        response = c_agent.chat(user_input)
        self.chat_history.append(f"User: {user_input}")
        self.chat_history.append(f"Agent: {response.response}")"""
        response = "Test response"
        
        return response

    def get_paper_titles(self):
        return self.title_list
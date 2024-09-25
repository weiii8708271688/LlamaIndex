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

# Initialize LLM and embedding model
llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initialize LlamaParse
parser = LlamaParse(
    api_key="llx-",
    result_type="markdown",
    verbose=True,
)

node_parser = SentenceSplitter()

def download_paper(paper_id: str) -> str:
    try:
        search = arxiv.Search(id_list=[paper_id])
        paper = next(search.results())
        
        pdf_path = f"data/pdf/{paper.title}.pdf"
        paper.download_pdf(filename=pdf_path)
        paper_docs = parser.load_data(pdf_path)
        base_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"./storage/all_datas"),
        )
        nodes = node_parser.get_nodes_from_documents(paper_docs)
        base_index.insert_nodes(nodes)
        base_index.storage_context.persist(persist_dir=f"./storage/all_datas")
        
        logging.info(f"Successfully downloaded and indexed paper: {paper.title} [{paper_id}]")
        return f"Successfully downloaded paper {paper.title} [{paper_id}] and stored its index in the storage folder"
    except Exception as e:
        logging.error(f"Error downloading paper {paper_id}: {str(e)}")
        return f"Error downloading paper {paper_id}: {str(e)}"

def search_paper(query: str) -> str:
    try:
        search = arxiv.Search(query=query, max_results=5)
        results = [f"Title: {result.title}, ID: {result.entry_id.split('/')[-1]}" for result in search.results()]
        return "\n".join(results) if results else "No papers found."
    except Exception as e:
        logging.error(f"Error searching for papers with query '{query}': {str(e)}")
        return f"Error searching for papers with query '{query}': {str(e)}"
    
def ask_about_papers(question: str)->str:
    base_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=f"./storage/all_datas"),
    )
    query_engine = base_index.as_query_engine(llm=Settings.llm)
    response = query_engine.query(question)
    return response.response

download_paper_tool = FunctionTool.from_defaults(
    fn=download_paper,
    description="Downloads and indexes an academic paper from arXiv given its ID."
)

search_paper_tool = FunctionTool.from_defaults(
    fn=search_paper,
    description="Searches for academic papers on arXiv based on a query string."
)

ask_about_papers_tool = FunctionTool.from_defaults(
    fn=ask_about_papers,
    description="Ask questions about the academic papers."
)

def create_a_agent():
    a_agent = ReActAgent.from_tools(
        [download_paper_tool, search_paper_tool], 
        llm=llm, 
        verbose=True, 
        max_iterations=10, 
        metadata=ToolMetadata(
            name="arXiv_interaction_agent",
            description="An agent specialized in interacting with the arXiv repository for academic paper retrieval and information gathering."
        )
    )
    return QueryEngineTool(
        query_engine=a_agent, 
        metadata=ToolMetadata(
            name="arXiv_research_assistant",
            description="A comprehensive tool for academic research assistance, specializing in arXiv repository interactions.There has two main tools to use: 1. download_paper: Download and index academic papers from arXiv based on their IDs. 2. search_paper: Search 5 papers for academic papers on arXiv based on a query string."
        )
    )

def create_b_agent():
    """all_tools = []
    all_tools.append(search_paper_tool)
    paper_titles = os.listdir('./storage')
    for paper_title in paper_titles:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"./storage/{paper_title}"),
        )
        vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)
        all_tools.append(QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"vector_tool_{paper_title}",
                description=f"Specialized tool for answering questions about the academic paper '{paper_title}'."
            ),
        ))"""
    b_agent = ReActAgent.from_tools([ask_about_papers_tool], llm=llm, verbose=True, max_iterations=5)
    
    return QueryEngineTool(
        query_engine=b_agent, 
        metadata=ToolMetadata(
            name="academic_paper_analysis_tool",
            description="A powerful academic paper analysis tool capable of answering in-depth questions about multiple academic papers.There has one main tool to use: ask_about_papers: Ask questions about the academic papers."
        )
    )

def create_c_agent(a_agent, b_agent, chat_history):
    return ReActAgent.from_tools(
        [a_agent, b_agent],
        llm=llm,
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
- Not every task requires the use of Agent A or Agent B. Use your judgment to determine when to involve them.
- You have access to the following chat history for context: {chat_history}
- If a task seems simple or within your capabilities, you may choose to handle it directly without involving the specialized agents.
- Always prioritize the user's needs and the efficiency of task completion.

Remember, your goal is to provide the most helpful and appropriate response to each user query, whether that involves coordinating with specialized agents or utilizing your own capabilities."""
    )

def main():
    a_agent = create_a_agent()
    b_agent = create_b_agent()
    chat_history = []

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            break

        c_agent = create_c_agent(a_agent, b_agent, chat_history)
        response = c_agent.chat(user_input)
        print(f"Agent: {response}")

        chat_history.append(f"User: {user_input}")
        chat_history.append(f"Agent: {response}")
        

        # Update b_agent only if new papers have been added
        if any(file.endswith('.pdf') for file in os.listdir('./storage')):
            b_agent = create_b_agent()

if __name__ == "__main__":
    main()
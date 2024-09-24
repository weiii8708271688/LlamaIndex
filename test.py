from llama_index.core.tools import FunctionTool

from llama_index.core.agent import ReActAgent
import arxiv
import os
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
)
llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
Settings.llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")



parser = LlamaParse(
    api_key=os.getenv('LLX_API_KEY'),
    result_type="markdown",
    verbose=True,
)
from llama_parse import LlamaParse



def download_paper(paper_id: str) -> str:
    try:
        search = arxiv.Search(id_list=[paper_id])
        paper = next(search.results())
        
        # 創建下載目錄
        os.makedirs("downloaded_papers", exist_ok=True)
        
        # 下載PDF
        paper.download_pdf(filename=f"data/pdf/{paper.title}.pdf")
        paper_docs = parser.load_data(f"./data/pdf/{paper.title}.pdf")
        # build index
        paper_index = VectorStoreIndex.from_documents(paper_docs)

        # persist index
        paper_index.storage_context.persist(persist_dir=f"./storage/{paper.title}")

        #然後這邊要把這個index弄到B agent裡面 B agent應該要是一個全域變數之類的
        
        return(f"成功下載論文: {paper.title} [{paper_id}]")
    except Exception as e:
        return(f"下載論文 {paper_id} 時發生錯誤: {str(e)}")

def search_paper(query: str) -> str:
    try:
        search = arxiv.Search(query=query, max_results = 5)
        results = []
        for result in search.results():
            results.append(f"Title: {result.title}, ID: {result.entry_id.split('/')[-1]}")
        return "\n".join(results) if results else "No papers found."
    except Exception as e:
        return f"Error searching for papers with query '{query}': {str(e)}"
    
def read_paper(paper_id: str) -> str:
    try:
        search = arxiv.Search(id_list=[paper_id])
        paper = next(search.results())
        ret = []
        ret.append(paper.authors)
        ret.append(paper.title)
        ret.append(paper.summary)  
        ret.append(paper.published)
        ret.append(paper.updated)
        
        return ret
    except Exception as e:
        return f"Error reading paper with ID '{paper_id}': {str(e)}"

download_paper_tool = FunctionTool.from_defaults(fn=download_paper,
                                                 description="""The download_paper function is designed to download academic papers from the arXiv repository. Here's a detailed description of its functionality:

                                                                Purpose: This function downloads a specific paper from arXiv given its ID and saves it as a PDF file.
                                                                Parameters:

                                                                paper_id (str): The arXiv ID of the paper to be downloaded. ID is all number there is no letter in it.


                                                                Workflow:

                                                                It uses the arxiv.Search class to search for the paper using the provided ID.
                                                                It retrieves the first (and should be the only) result from the search.
                                                                It creates a directory named "downloaded_papers" if it doesn't already exist.
                                                                It then downloads the PDF of the paper, saving it in the "data" directory with the paper's title as the filename.


                                                                Error Handling:

                                                                The function is wrapped in a try-except block to catch and handle any exceptions that may occur during the download process.
                                                                If an error occurs, it prints an error message including the paper ID and the specific error encountered.


                                                                Output:

                                                                On successful download, it prints a success message with the paper ID.
                                                                On failure, it prints an error message detailing the issue.


                                                                Note:

                                                                The function doesn't return any value (returns None).
                                                                It assumes the existence of an 'arxiv' module and appropriate permissions to create directories and write files.



                                                                This function is useful in automating the process of downloading academic papers from arXiv, which can be particularly helpful in research tasks, literature reviews, or building a corpus of academic papers for further analysis.
                                                                """
                                                 )


search_paper_tool = FunctionTool.from_defaults(fn=search_paper,
                                               description="""The search_paper function is designed to search for academic papers on the arXiv repository based on a query string. Here's a detailed description of its functionality:

                                                                Purpose: This function searches for papers on arXiv using a query string and returns a list of titles and IDs of the matching papers.
                                                                Parameters:

                                                                query (str): The search query string. string must translate to English first


                                                                Workflow:

                                                                It uses the arxiv.Search class to search for papers using the provided query string.string must translate to English first
                                                                It iterates through the search results and collects the titles and IDs of the matching papers.
                                                                It returns a formatted string containing the titles and IDs of the matching papers.


                                                                Error Handling:

                                                                The function is wrapped in a try-except block to catch and handle any exceptions that may occur during the search process.
                                                                If an error occurs, it returns an error message including the query string and the specific error encountered.


                                                                Output:

                                                                On successful search, it returns a formatted string containing the titles and IDs of the matching papers.
                                                                On failure, it returns an error message detailing the issue.


                                                                Note:

                                                                The function assumes the existence of an 'arxiv' module and appropriate permissions to access the internet.


                                                                This function is useful in automating the process of searching for academic papers on arXiv, which can be particularly helpful in research tasks, literature reviews, or building a corpus of academic papers for further analysis.
                                                                """
                                                    )

read_paper_tool = FunctionTool.from_defaults(fn=read_paper,
                                             description="""This function retrieves information about an academic paper from the arXiv database using its unique identifier (paper_id).paper_id only contain number and version. It returns a list containing the following details about the paper:

                                                            Authors
                                                            Title
                                                            Summary (abstract)
                                                            Publication date
                                                            Last update date

                                                            The function uses the arxiv library to search for and fetch the paper details. If successful, it extracts the relevant information and returns it as a list. In case of any errors (e.g., invalid paper ID, network issues), it returns an error message string instead.
                                                            Use this function when you need to access basic metadata about a specific arXiv paper. Provide the paper's arXiv ID as input."""
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceSplitter



def create_b_agent(options='b'):
    all_tools = []
    paper_titles = os.listdir('./data/pdf')
    paper_titles = [title.split('.pdf')[0] for title in paper_titles if title.endswith('.pdf')]
    for paper_title in paper_titles:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"./storage/{paper_title}"),
        )
        vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)
        all_tools.append(QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description = (
        f"Useful for answering questions about the academic paper titled '{paper_title}'. "
        "This tool can provide information on various aspects of the paper, including but not limited to:"
        "\n- The main research question or hypothesis"
        "\n- Methodology and experimental design"
        "\n- Key findings and results"
        "\n- Theoretical framework and background"
        "\n- Implications and conclusions"
        "\n- Related work and literature review"
        "\n- Limitations and future research directions"
        "\nUse a specific question about the paper as input to this tool."
                ),
            ),
        ),)
    b_agent = ReActAgent.from_tools(all_tools, llm=llm, verbose=True, max_iterations=10)
    return b_agent

def create_a_agent():
    all_tools = []
    all_tools.append(download_paper_tool)
    all_tools.append(search_paper_tool)
    all_tools.append(read_paper_tool)
    a_agent = ReActAgent.from_tools(all_tools, llm=llm, verbose=True, max_iterations=10, metadata=ToolMetadata(name="", description=""))
    return a_agent


a_agent = None # 這個是主要的腦
b_agent = None # 這個是找資料的進大腦的
c_agent = None # 這個是主持人
a_agent = create_a_agent()
b_agent = create_b_agent()

c_agent = ReActAgent.from_tools([a_agent, b_agent], llm=llm, verbose=True, max_iterations=10, system_prompt="You are the host. You are responsible for managing the conversation between the user and the agents.")

while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = c_agent.chat(text_input)
    print(f"Agent: {response}")
    b_agent = create_b_agent('a')
    CHAT_HISTORY = c_agent.chat_history
    c_agent = ReActAgent.from_tools([a_agent, b_agent], chat_history=CHAT_HISTORY ,llm=llm, verbose=True, max_iterations=10, system_prompt="You are the host. You are responsible for managing the conversation between the user and the agents. not every task will use agent a or agent b, it will depend on the task.Maybe you can ask llm to help you with the task without using agent.")



# What was Lyft's revenue growth in 2021?
# What was Uber's revenue growth in 2021?
# Base on above chat_history. Compare the revenue growth of Uber and Lyft in 2021.
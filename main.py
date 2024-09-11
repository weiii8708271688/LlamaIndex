
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

parser = LlamaParse(
    api_key="llx-tIeANhGTiFwsFqCxWiL0tUe07FF4KshEtckrClY7J5vX2r8z",
    result_type="markdown",
    verbose=True,
)


paper_titles = os.listdir('./data/pdf')
paper_titles = [title.split('.pdf')[0] for title in paper_titles if title.endswith('.pdf')]
paper_docs = {}
for paper_title in paper_titles:
    paper_docs[paper_title] = parser.load_data(f"./data/pdf/{paper_title}.pdf")



paper_index = {}
for paper_title in paper_titles:

    # build index
    paper_index[paper_title] = VectorStoreIndex.from_documents(paper_docs[paper_title])

    # persist index
    paper_index[paper_title].storage_context.persist(persist_dir=f"./storage/{paper_title}")





node_parser = SentenceSplitter()


# Build agents dictionary
agents = {}
query_engines = {}

# this is for the baseline
all_nodes = []


for idx, paper_title in enumerate(paper_titles):
    nodes = node_parser.get_nodes_from_documents(paper_docs[paper_title])
    all_nodes.extend(nodes)

    if not os.path.exists(f"./data/{paper_title}"):
        # build vector index
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(
            persist_dir=f"./data/{paper_title}"
        )
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"./data/{paper_title}"),
        )

    # build summary index
    summary_index = SummaryIndex(nodes)
    print(vector_index)
    print('----------------------')
    print(summary_index)
    # define query engines
    vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)
    summary_query_engine = summary_index.as_query_engine(llm=Settings.llm)

    # define tools
    query_engine_tools = [
        QueryEngineTool(
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
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description = (
                    f"Useful for any requests that require a holistic summary of EVERYTHING about the academic paper '{paper_title}'. "
                    "This tool provides a comprehensive overview of the entire paper, including:"
                    "\n- The main research question and objectives"
                    "\n- Key methodologies and theoretical framework"
                    "\n- Overall findings and their significance"
                    "\n- Major conclusions and implications"
                    "\n- The paper's contribution to its field"
                    "\nFor questions about more specific sections or detailed aspects of the paper, please use the vector_tool."
                ),
            ),
        ),
    ]

    # build agent
    function_llm = Ollama(model="llama3:latest", request_timeout=120.0)
    agent = ReActAgent.from_tools(
        
        query_engine_tools,
        max_iterations=100,
        llm=function_llm,
        verbose=True,
        system_prompt=f"""\
You are a specialized agent designed to answer queries about {paper_title}.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
""",
    )

    agents[paper_title] = agent
    query_engines[paper_title] = vector_index.as_query_engine(
        similarity_top_k=2
    )


# define tool for each document agent
all_tools = []
for paper_title in paper_titles:
    paper_summary = (
        f"This content contains paper articles about {paper_title}. Use"
        f" this tool if you want to answer any questions about {paper_title}.\n"
    )
    doc_tool = QueryEngineTool(
        query_engine=agents[paper_title],
        metadata=ToolMetadata(
            name=f"tool_{paper_title}",
            description=paper_summary,
        ),
    )
    all_tools.append(doc_tool)


# define an "object" index and retriever over these tools
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

top_agent = ReActAgent.from_tools(
    max_iterations=100,
    tool_retriever=obj_index.as_retriever(similarity_top_k=3),
    system_prompt=""" \
You are an agent designed to answer queries about a set of given paper.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    verbose=True,
)


base_index = VectorStoreIndex(all_nodes)
base_query_engine = base_index.as_query_engine(similarity_top_k=4)

response = top_agent.query("Tell me about ReDel: A Toolkit for LLM-Powered Recursive Multi-Agent Systems")
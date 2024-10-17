from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
Settings.llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
import  llama_index.core.chat_engine
import os

### Recipe
### Build a recursive retriever that retrieves using small chunks
### but passes associated larger chunks to the generation stage

# load data
pdfs = os.listdir("data/pdf")
documents = SimpleDirectoryReader(
  input_dir="data/pdf/"
).load_data()

# build parent chunks via NodeParser
node_parser = SentenceSplitter(chunk_size=1024)
base_nodes = node_parser.get_nodes_from_documents(documents)

# define smaller child chunks
sub_chunk_sizes = [256, 512]
sub_node_parsers = [
    SentenceSplitter(chunk_size=c, chunk_overlap=20) for c in sub_chunk_sizes
]
all_nodes = []
for base_node in base_nodes:
    for n in sub_node_parsers:
        sub_nodes = n.get_nodes_from_documents([base_node])
        sub_inodes = [
            IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
        ]
        all_nodes.extend(sub_inodes)
    # also add original node to node
    original_node = IndexNode.from_text_node(base_node, base_node.node_id)
    all_nodes.append(original_node)

# define a VectorStoreIndex with all of the nodes
vector_index_chunk = VectorStoreIndex(
    all_nodes
)
vector_index_chunk.storage_context.persist(persist_dir="./storage/all_datas")

"""
from llama_index.core import Settings, VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.retrievers.bm25 import BM25Retriever
vector_index_chunk = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./storage/all_datas_v2"),
    )
vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=2)
bm25_retriever = BM25Retriever.from_defaults(
    docstore=vector_index_chunk.docstore, similarity_top_k=2
)
# build RecursiveRetriever
all_nodes_dict = {n.node_id: n for n in all_nodes}
retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_chunk},
    node_dict=all_nodes_dict,
    verbose=True,
)
retriever = QueryFusionRetriever(
    [vector_retriever_chunk, bm25_retriever],
    similarity_top_k=2,
    num_queries=4,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=True,
    verbose=True,
    # query_gen_prompt="...",  # we could override the query generation prompt here
)

# build RetrieverQueryEngine using recursive_retriever
query_engine_chunk = RetrieverQueryEngine.from_args(
    retriever
)
query_engine_chunk2 = RetrieverQueryEngine.from_args(
    vector_retriever_chunk
)
# perform inference with advanced RAG (i.e. query engine)
response = query_engine_chunk.query(
    "Tell me about ColPali and LoRa. And tell me about their differences.Answer me use traditional chinese ."
)




response2 = query_engine_chunk.query(
    "Tell me about ColPali and LoRa. And tell me about their differences.Answer me use traditional chinese ."
)

print(response)

print('---------------------------------')
print(response2)



"""
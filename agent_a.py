from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import Settings
import arxiv
import requests
import xml.etree.ElementTree as ET
import asyncio
import threading
import time
from pathlib import Path

class AgentA:
    def __init__(self, llm,):
        self.llm = llm
        self.download_paper_tool = FunctionTool.from_defaults(
            fn=self.download_paper,
            description="Downloads and indexes an academic paper from arXiv given its ID."
        )
        self.search_paper_tool = FunctionTool.from_defaults(
            fn=self.search_paper,
            description="Searches for academic papers on arXiv based on a query string."
        )

    def create_agent(self):
        system_prompt = """
            You are an advanced AI agent specialized in academic paper retrieval and information gathering from the arXiv repository. Your primary functions are:

            1. Searching for academic papers based on user queries.
            2. Downloading and indexing papers from arXiv using their unique identifiers.

            Key Capabilities:
            1. Paper Search:
            - Use the 'search_paper' tool to find relevant papers on arXiv.
            - Return up to 5 most relevant results, including titles and arXiv IDs.
            - If no papers are found, suggest refining the search query.

            2. Paper Download:
            - Use the 'download_paper' tool to retrieve and index papers by their arXiv ID.
            - Ensure the paper is not already in the database before initiating a download.
            - Handle potential errors during download or processing gracefully.

            Operational Guidelines:
            - Always start with a search if the user doesn't provide a specific arXiv ID.
            - When downloading, confirm the success of both the download and indexing process.
            - If a paper is already in the database, inform the user and offer to provide information about it.
            - Be mindful of potential API rate limits and inform the user if multiple requests are needed.
            - Provide clear, concise responses about the status of each operation (search or download).

            Interaction Style:
            - Maintain a professional and academic tone.
            - Be proactive in suggesting related papers or additional searches when appropriate.
            - If a user's query is ambiguous, ask for clarification before proceeding.
            - Offer brief explanations of your actions to keep the user informed of the process.

            Remember: Your goal is to efficiently assist users in finding and accessing relevant academic papers from arXiv, enhancing their research capabilities.
            """
        a_agent = ReActAgent.from_tools(
            [self.download_paper_tool, self.search_paper_tool], 
            llm=self.llm, 
            verbose=True, 
            max_iterations=10, 
            system_prompt=system_prompt
        )
        return QueryEngineTool(
            query_engine=a_agent, 
            metadata=ToolMetadata(
                name="Search_Agent_Tool",
                description="A comprehensive tool for academic research assistance, specializing in arXiv repository interactions. There are two main tools to use: 1. download_paper: Download and index academic papers from arXiv based on their IDs. 2. search_paper: Search 5 papers for academic papers on arXiv based on a query string."
            )
        )
    
    def download_paper(self, paper_id: str) -> str:
        try:
            
            print(f"Starting download of paper {paper_id}")
            global paper_docs
            # 使用 arXiv API 獲取論文信息
            api_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
            response = requests.get(api_url)
            response.raise_for_status()
            
            # 解析 XML 響應
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = root.find('atom:entry', ns)
            if entry is None:
                return f"Paper {paper_id} not found on arXiv."
            
            title = entry.find('atom:title', ns).text
            pdf_url = entry.find('atom:link[@title="pdf"]', ns).attrib['href']
            
            # 構建檔案路徑，確保文件名合法
            
            safe_title = "".join(c for c in title if c.isalnum() or c in [' ', '-', '_']).rstrip()
            filename = f"{safe_title}.pdf"
            pdf_path = Path(f"data/pdf/{filename}")
            if pdf_path.exists():
                print(f"Paper '{safe_title}' [{paper_id}] already exists.")
                return f"Paper '{safe_title}' [{paper_id}] already downloaded before."
            
            return "Paper download not implemented yet."
            # 下載 PDF 文件
            print(f"Downloading PDF for paper {paper_id}")
            pdf_response = requests.get(pdf_url)
            pdf_response.raise_for_status()
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(pdf_response.content)
            
            # 解析 PDF 文件
            print(f"Parsing PDF for paper {paper_id}")
            self.run_async_parse(pdf_path)
            
            # 加載或創建索引
            print(f"Loading or creating index for paper {paper_id}")
            try:
                base_index = load_index_from_storage(
                    StorageContext.from_defaults(persist_dir="./storage/all_datas"),
                )
            except Exception:
                base_index = VectorStoreIndex([])
            
            # 創建節點並插入索引
            print(f"Creating nodes and inserting into index for paper {paper_id}")
            nodes = self.node_parser.get_nodes_from_documents(paper_docs)
            base_index.insert_nodes(nodes)
            
            # 保存索引
            print(f"Saving index for paper {paper_id}")
            base_index.storage_context.persist(persist_dir="./storage/all_datas")
            
            # 更新論文列表
            self.title_list = [f.name for f in Path("./data/pdf").glob("*.pdf")]
            
            end_time = time.time()
            print(f"Completed download and processing of paper {paper_id}")
            return f"Successfully downloaded paper '{safe_title}' [{paper_id}] and stored its index in the storage folder"
        except Exception as e:
            print(f"Error processing paper {paper_id}: {str(e)}")
            return f"Error processing paper {paper_id}: {str(e)}"
        
    def search_paper(self, query: str) -> str:
        try:
            search = arxiv.Search(query=query, max_results=10)
            results = [f"Title: {result.title}, ID: {result.entry_id.split('/')[-1]}" for result in search.results()]
            return "\n".join(results) if results else "No papers found."
        except Exception as e:
            return f"Error searching for papers with query '{query}': {str(e)}"
        

    def run_async_parse(self, pdf_path):
        async def async_parse():
            global paper_docs
            paper_docs = await self.parser.aload_data(str(pdf_path))

        def run_in_thread(loop):
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(async_parse())

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=run_in_thread, args=(loop,))
        thread.start()
        thread.join()
        return None
    
    def load(self, filename):
        documents = SimpleDirectoryReader(
        input_files=[f"data/pdf/{filename}"]
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
        vector_index_chunk.storage_context.persist(persist_dir=f"./storage/{filename[:-4]}")
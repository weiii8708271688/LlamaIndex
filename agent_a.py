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
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.chat_engine import SimpleChatEngine
import os
from llama_index.llms.ollama import Ollama
import prompt.agent_a_prompt as agent_a_prompt
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import IndexNode
import asyncio, nest_asyncio
nest_asyncio.apply()

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

        self.generate_paper_search_prompt_tool = FunctionTool.from_defaults(
            fn=self.generate_paper_search_prompt,
            description="Query the paper database to check for paper existence and get file names. Always use this first. "
        )

    def create_agent(self):
        
        a_agent = ReActAgent.from_tools(
            [self.download_paper_tool, self.search_paper_tool, self.generate_paper_search_prompt_tool], 
            llm=self.llm, 
            verbose=True, 
            max_iterations=10, 
            context=agent_a_prompt.SYSTEM_PROMPT
        )
        a_agent = FunctionTool.from_defaults(
                fn=a_agent.chat,
                name="PaperDownloadAssistant_AgentA",
                description="""Advanced tool for arXiv paper search, download, and management. Has 'meaaage' one parameter."""
                )
        return a_agent
        
    
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
            filename = f"{safe_title}_{paper_id}.pdf"
            pdf_path = Path(f"data/pdf/{filename}")
            if pdf_path.exists():
                print(f"Paper '{safe_title}' [{paper_id}] already exists.")
                return f"Paper '{safe_title}' [{paper_id}] already downloaded before."
            
            # 下載 PDF 文件
            print(f"Downloading PDF for paper {paper_id}")
            pdf_response = requests.get(pdf_url)
            pdf_response.raise_for_status()
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(pdf_response.content)
            
            # 解析 PDF 文件
            print(f"Parsing PDF for paper {paper_id}")
            self.load(filename)
            
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
            search = arxiv.Search(query=query, max_results=3)
            results = [f"Title: {result.title}, ID: {result.entry_id.split('/')[-1]}" for result in search.results()]
            return "\n".join(results) if results else "No papers found."
        except Exception as e:
            return f"Error searching for papers with query '{query}': {str(e)}"
        
    
    def load(self, filename):
        global paper_docs
        paper_docs = SimpleDirectoryReader(
        input_files=[f"data/pdf/{filename}"]
        ).load_data()

        # build parent chunks via NodeParser
        node_parser = SentenceSplitter(chunk_size=1024)
        base_nodes = node_parser.get_nodes_from_documents(paper_docs)

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
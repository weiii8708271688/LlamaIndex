from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent






multiply_tool = FunctionTool.from_defaults(fn=multiply)

llm = Ollama(model="llama3:latest", request_timeout=120.0)

agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)

agent.chat("what is 50*60")


"""import arxiv
import os
from typing import List

def download_paper(paper_id: str) -> None:

    try:
        search = arxiv.Search(id_list=[paper_id])
        paper = next(search.results())
        
        # 創建下載目錄
        os.makedirs("downloaded_papers", exist_ok=True)
        
        # 下載PDF
        paper.download_pdf(filename=f"data/{paper.title}.pdf")
        print(f"成功下載論文: {paper_id}")
    except Exception as e:
        print(f"下載論文 {paper_id} 時發生錯誤: {str(e)}")

def main(paper_ids: List[str]) -> None:

    for paper_id in paper_ids:
        download_paper(paper_id)

if __name__ == "__main__":
    paper_ids = [
        "2106.09685",
        "2408.08921",
        "2407.01449",
        "2407.13278",
        "2302.04761",
        "2401.03955",
        "2103.00020",
        "2408.05933",
        "2408.02248",
        "2407.19994",
        "2407.12036"
    ]
    
    main(paper_ids)"""
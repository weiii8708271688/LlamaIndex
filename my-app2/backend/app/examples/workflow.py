import asyncio
from typing import AsyncGenerator, List, Optional
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.core.chat_engine.types import ChatMessage
from app.agents.single import AgentRunEvent, AgentRunResult, FunctionCallingAgent




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





def create_workflow(chat_history: Optional[List[ChatMessage]] = None):
    searcher = FunctionCallingAgent(
        name="searcher",
        role="專業文獻搜尋專家",
        system_prompt="你是一個專業的文獻搜尋專家。你的任務是根據給定的主題或關鍵詞搜索相關的學術文獻。請提供文獻的標題、作者、發表年份和簡短摘要。",
        chat_history=chat_history,
    )
    reader = FunctionCallingAgent(
        name="reader",
        role="學術文獻分析專家",
        system_prompt="你是一個學術文獻分析專家。你的任務是仔細閱讀和分析給定的文獻摘要，提取關鍵信息，並識別主要觀點和研究方法。",
        tools=
        chat_history=chat_history,
    )
    moderator = FunctionCallingAgent(
        name="moderator",
        role="研究項目協調員",
        system_prompt="你是一個研究項目的協調員。你的任務是與用戶互動，理解他們的研究需求，整合其他專家的意見，並為用戶提供有價值的研究建議和總結。",
        chat_history=chat_history,
    )
    workflow = ResearchWorkflow(timeout=99999)
    workflow.add_workflows(searcher=searcher, reader=reader, moderator=moderator)
    return workflow

class SearchEvent(Event):
    input: str

class ReadEvent(Event):
    input: str

class ModerateEvent(Event):
    input: str

class ResearchWorkflow(Workflow):
    @step()
    async def start(self, ctx: Context, ev: StartEvent) -> SearchEvent:
        ctx.data["streaming"] = getattr(ev, "streaming", False)
        ctx.data["research_topic"] = ev.input
        return SearchEvent(input=f"搜索以下研究主題的相關文獻：{ev.input}")

    @step()
    async def search(self, ctx: Context, ev: SearchEvent, searcher: FunctionCallingAgent) -> ReadEvent:
        result: AgentRunResult = await self.run_agent(ctx, searcher, ev.input)
        literature_list = result.response.message.content
        return ReadEvent(input=f"分析以下文獻列表：{literature_list}")

    @step()
    async def read(self, ctx: Context, ev: ReadEvent, reader: FunctionCallingAgent) -> ModerateEvent:
        result: AgentRunResult = await self.run_agent(ctx, reader, ev.input)
        analysis = result.response.message.content
        return ModerateEvent(input=f"根據以下分析與用戶討論研究方向：{analysis}")

    @step()
    async def moderate(self, ctx: Context, ev: ModerateEvent, moderator: FunctionCallingAgent) -> StopEvent:
        result = await self.run_agent(ctx, moderator, ev.input, streaming=ctx.data["streaming"])
        return StopEvent(result=result)

    async def run_agent(
        self,
        ctx: Context,
        agent: FunctionCallingAgent,
        input: str,
        streaming: bool = False,
    ) -> AgentRunResult | AsyncGenerator:
        task = asyncio.create_task(agent.run(input=input, streaming=streaming))
        async for event in agent.stream_events():
            ctx.write_event_to_stream(event)
        return await task
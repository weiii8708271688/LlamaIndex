import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ai_research_assistant import AIResearchAssistant  # 導入您的 AIResearchAssistant 類
from fastapi.responses import StreamingResponse
import asyncio
import json
from typing import List, Dict, Any
app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源，您可能想要限制這個
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VercelCompatibleResponse:
    TEXT_PREFIX = "0:"
    DATA_PREFIX = "8:"

    @classmethod
    def convert_text(cls, token: str) -> str:
        escaped_token = json.dumps(token)
        return f"{cls.TEXT_PREFIX}{escaped_token}\n"

    @classmethod
    def convert_data(cls, data: Dict[str, Any]) -> str:
        data_str = json.dumps(data)
        return f"{cls.DATA_PREFIX}[{data_str}]\n"

async def stream_response(response: str):
    yield VercelCompatibleResponse.convert_text("")  # 開始流
    
    for chunk in response.split():  # 簡單按單詞分割，您可能需要更複雜的邏輯
        yield VercelCompatibleResponse.convert_text(chunk)
        await asyncio.sleep(0.1)  # 模擬延遲
    
    # 模擬事件輸出
    event_data = {
        "type": "agent",
        "data": {"agent": "論文小幫手", "text": "Processing complete"}
    }
    yield VercelCompatibleResponse.convert_data(event_data)
    
    # 模擬建議問題
    suggested_questions = ["What is the main topic of this paper?", "Can you summarize the key findings?"]
    question_data = {
        "type": "suggested_questions",
        "data": suggested_questions
    }
    yield VercelCompatibleResponse.convert_data(question_data)

# 初始化您的 AI 助手
assistant = AIResearchAssistant()

class Message(BaseModel):
    role: str
    content: str

class ChatData(BaseModel):
    messages: List[Message]

@app.post("/api/chat")
async def chat(request: Request, data: ChatData):
    try:
        # 獲取最後一條消息
        last_message = data.messages[-1].content
        
        # 使用您的 AI 助手處理消息
        response = assistant.chat(last_message)
        print('-------------------')
        print(response)
        print('-------------------')
        # 返回響應
        return StreamingResponse(stream_response(response), media_type="text/event-stream")
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
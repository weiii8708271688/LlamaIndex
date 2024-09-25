import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ai_research_assistant import AIResearchAssistant  # 導入您的 AIResearchAssistant 類
from fastapi.responses import StreamingResponse
import asyncio
import json
app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源，您可能想要限制這個
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def stream_response(response: str):
    # 模擬流式響應
    for chunk in response.split():  # 這裡簡單地按單詞切分，您可能需要更複雜的邏輯
        yield f"data: {json.dumps({'response': chunk})}\n\n"
        await asyncio.sleep(0.1)  # 添加一些延遲以模擬真實的流式響應
    yield f"data: [DONE]\n\n"

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
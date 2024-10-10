from ai_research_assistant import AIResearchAssistant
from llama_index.core.chat_engine import ContextChatEngine
import asyncio




async def main():
    assistant = AIResearchAssistant()
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = await assistant.chat(user_input)
        print(response)

if __name__ == "__main__":
    asyncio.run(main())
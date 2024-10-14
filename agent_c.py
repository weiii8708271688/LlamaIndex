from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.tools import FunctionTool, ToolMetadata
import prompt.agent_c_prompt as agent_c_prompt

class AgentC:
    def __init__(self, llm, agent_a, agent_b):
        self.llm = llm
        self.agent_a = agent_a
        self.agent_b = agent_b

    def create_agent(self):
        
        c_agent = ReActAgent.from_tools(
            [self.agent_a, self.agent_b],
            llm=self.llm,
            verbose=True,
            max_iterations=10,
            context=agent_c_prompt.FULL_PROMPT
        )
        
        return c_agent
    
    def chattool(self):
        chat_engine = SimpleChatEngine.from_defaults(llm = self.llm)
        chat_tool = FunctionTool.from_defaults(
            fn=chat_engine.chat,
            name="SimpleChatTool",
            description="A tool that uses SimpleChatEngine to process messages and generate responses.",
            tool_metadata=ToolMetadata(
                name="SimpleChatTool",
                description="Use this tool for general conversation or when other specialized tools are not applicable."
            )
        )
        return chat_tool
    
    def create_enhance_response_tool(self):
        def enhance_response(original_response: str, context: str) -> str:
            prompt = f"""
            As a professional academic writer, your task is to enhance the following response 
            to make it more professional, academically sound, and comprehensive. The response 
            is related to the following context: {context}

            Original response:
            {original_response}

            Please rewrite the response to:
            1. Use more precise and academic language
            2. Add relevant academic references or citations if appropriate
            3. Provide a more structured and logical flow of ideas
            4. Include any additional relevant information that might be helpful
            5. Ensure the tone is professional and suitable for an academic audience

            Enhanced response:
            """

            enhanced_response = self.llm.complete(prompt)
            return enhanced_response.text

        return FunctionTool.from_defaults(
            fn=enhance_response,
            name="EnhanceResponseTool",
            description="A tool to enhance responses, making them more professional and academically sound.",
        )
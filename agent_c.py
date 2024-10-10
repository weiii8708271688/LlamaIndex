from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.tools import FunctionTool, ToolMetadata


class AgentC:
    def __init__(self, llm, agent_a, agent_b):
        self.llm = llm
        self.agent_a = agent_a
        self.agent_b = agent_b

    def create_agent(self):
        system_prompt = f"""You are an intelligent conversation host and task manager specializing in academic research assistance. Your primary role is to efficiently handle user queries about academic papers, utilizing specialized agents when necessary.

            Key Responsibilities:
            1. Analyze and understand the user's requests or queries about academic papers.
            2. Follow this specific workflow for each task:
               a. ALWAYS start by using Agent B (Retrieval_Agent_Tool) to check if the requested paper is already downloaded and to analyze its content.
               b. Only if Agent B cannot find the paper or provide sufficient information, then use Agent A (Search_Agent_Tool) to search for and potentially download the paper from arXiv.
               c. After using Agent A, always return to Agent B to analyze the newly downloaded paper.
               d. Handle simpler tasks or general queries directly using your own knowledge and capabilities.
            3. Manage the conversation flow, ensuring clarity and coherence.
            4. Synthesize information from multiple sources (agents, your own knowledge, user input) when necessary.
            5. Provide clear, concise, and relevant responses to the user.

            Important Notes:
            - Always start with Agent B (Retrieval_Agent_Tool) for any paper-related query.
            - Only use Agent A (Search_Agent_Tool) if Agent B cannot find the requested paper or information.
            - After using Agent A to find or download a paper, always use Agent B again to analyze it.
            - For non-paper-specific queries or simple tasks, you may handle them directly without involving the specialized agents.
            - Prioritize using existing information (via Agent B) before searching for new papers (via Agent A).

            Remember, your goal is to provide the most helpful and appropriate response to each user query about academic papers, primarily utilizing Agent B for retrieval and analysis, and only resorting to Agent A when necessary for new paper searches or downloads."""
        
        c_agent = ReActAgent.from_tools(
            [self.agent_a, self.agent_b],
            llm=self.llm,
            verbose=True,
            max_iterations=10,
            system_prompt=system_prompt
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
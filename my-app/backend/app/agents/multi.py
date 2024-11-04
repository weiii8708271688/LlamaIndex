import asyncio
from typing import Any, List

from llama_index.core.tools.types import ToolMetadata, ToolOutput
from llama_index.core.tools.utils import create_schema_from_function
from llama_index.core.workflow import Context, Workflow

from app.agents.single import (
    AgentRunResult,
    ContextAwareTool,
    FunctionCallingAgent,
)
from app.agents.planner import StructuredPlannerAgent


class AgentCallTool(ContextAwareTool): #這個工具允許一個代理委派任務給另一個代理，實現了代理之間的協作。
    def __init__(self, agent: Workflow) -> None:
        self.agent = agent
        name = f"call_{agent.name}"

        async def schema_call(input: str) -> str:
            pass

        # create the schema without the Context
        
        fn_schema = create_schema_from_function(name, schema_call)
        print(f"fn_schema: {fn_schema}")
        self._metadata = ToolMetadata(
            name=name,
            description=(
                f"Use this tool to delegate a sub task to the {agent.name} agent."
                + (f" The agent is an {agent.role}." if agent.role else "")
            ),
            fn_schema=fn_schema,
        )

    # overload the acall function with the ctx argument as it's needed for bubbling the events
    async def acall(self, ctx: Context, input: str) -> ToolOutput:
        task = asyncio.create_task(self.agent.run(input=input))
        # bubble all events while running the agent to the calling agent
        async for ev in self.agent.stream_events():
            ctx.write_event_to_stream(ev)
        ret: AgentRunResult = await task
        response = ret.response.message.content
        try:
            print(f"{self.agent.name} memory {self.agent.memory.model_dump_json(indent=4)}")
        except Exception as e:
            print(f"Error while getting memory: {e}")

        print(f"{self.agent.name} have been called")
        return ToolOutput(
            content=str(response),
            tool_name=self.metadata.name,
            raw_input={"args": input, "kwargs": {}},
            raw_output=response,
        )
        


class AgentCallingAgent(FunctionCallingAgent):#這個類的目的是創建一個可以調用其他代理的代理。
    def __init__(
        self,
        *args: Any,
        name: str,
        agents: List[FunctionCallingAgent] | None = None,
        **kwargs: Any,
    ) -> None:
        agents = agents or []
        tools = [AgentCallTool(agent=agent) for agent in agents]
        super().__init__(*args, name=name, tools=tools, **kwargs)
        # call add_workflows so agents will get detected by llama agents automatically
        self.add_workflows(**{agent.name: agent for agent in agents})
        print("here is the AgentCallingAgent have been called")


class AgentOrchestrator(StructuredPlannerAgent): #這個類用於協調多個代理的工作。
    def __init__(
        self,
        *args: Any,
        name: str = "orchestrator",
        agents: List[FunctionCallingAgent] | None = None,
        **kwargs: Any,
    ) -> None:
        agents = agents or []
        tools = [AgentCallTool(agent=agent) for agent in agents]
        super().__init__(
            *args,
            name=name,
            tools=tools,
            **kwargs,
        )
        # call add_workflows so agents will get detected by llama agents automatically
        self.add_workflows(**{agent.name: agent for agent in agents})

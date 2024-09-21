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


class AgentCallTool(ContextAwareTool):
    def __init__(self, agent: Workflow) -> None:
        self.agent = agent
        name = f"call_{agent.name}"

        async def schema_call(input: str) -> str:
            pass

        # create the schema without the Context
        fn_schema = create_schema_from_function(name, schema_call)
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
        return ToolOutput(
            content=str(response),
            tool_name=self.metadata.name,
            raw_input={"args": input, "kwargs": {}},
            raw_output=response,
        )


class AgentCallingAgent(FunctionCallingAgent):
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


class AgentOrchestrator(StructuredPlannerAgent):
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

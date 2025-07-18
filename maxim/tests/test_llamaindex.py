import unittest
import os
import dotenv
import asyncio

from llama_index.core.tools import BaseTool
from tavily import AsyncTavilyClient
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
)

from maxim import Maxim
from maxim.logger.llamaindex import instrument_llamaindex

dotenv.load_dotenv()

baseUrl = os.getenv("MAXIM_BASE_URL")

class TestLlamaIndex(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # This is a hack to ensure that the Maxim instance is not cached
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

        self.maxim = Maxim({"base_url": baseUrl})
        self.logger = self.maxim.logger()
        instrument_llamaindex(self.logger)
        
    def test_llamaindex_simple_agent(self):
        llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2, max_tokens=1024)
        agent = FunctionAgent(
            name="SimpleAgent",
            description="Useful for answering questions.",
            llm=llm,
        )
        
        async def run_workflow():
            res = await agent.run("What is the capital of Spain?")
            return res

        result = asyncio.run(run_workflow())
        print(result.result)
        
    def test_llamaindex_simple_agent_with_system_prompt(self):
        llm = OpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2, max_tokens=1024)
        agent = FunctionAgent(
            name="SimpleAgent",
            description="Useful for answering questions.",
            system_prompt="You are a simple agent that can answer questions.",
            llm=llm,
        )
        
        async def run_workflow():
            res = await agent.run("What is the capital of France?")
            return res

        result = asyncio.run(run_workflow())
        print(result.result)

    def test_llamaindex_complex_agent(self):
        llm = OpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

        async def search_web(query: str) -> str:
            """Useful for using the web to answer questions."""
            client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            return str(await client.search(query))


        async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
            """Useful for recording notes on a given topic. Your input should be notes with a title to save the notes under."""
            current_state = await ctx.store.get("state")
            if "research_notes" not in current_state:
                current_state["research_notes"] = {}
            current_state["research_notes"][notes_title] = notes
            await ctx.store.set("state", current_state)
            return "Notes recorded."


        async def write_report(ctx: Context, report_content: str) -> str:
            """Useful for writing a report on a given topic. Your input should be a markdown formatted report."""
            current_state = await ctx.store.get("state")
            current_state["report_content"] = report_content
            await ctx.store.set("state", current_state)
            return "Report written."


        async def review_report(ctx: Context, review: str) -> str:
            """Useful for reviewing a report and providing feedback. Your input should be a review of the report."""
            current_state = await ctx.store.get("state")
            current_state["review"] = review
            await ctx.store.set("state", current_state)
            return "Report reviewed."

        research_agent = FunctionAgent(
            name="ResearchAgent",
            description="Useful for searching the web for information on a given topic and recording notes on the topic.",
            system_prompt=(
                "You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
                "Once notes are recorded and you are satisfied, you should hand off control to the WriteAgent to write a report on the topic. "
                "You should have at least some notes on a topic before handing off control to the WriteAgent."
            ),
            llm=llm,
            tools=[search_web, record_notes],
            can_handoff_to=["WriteAgent"],
        )

        write_agent = FunctionAgent(
            name="WriteAgent",
            description="Useful for writing a report on a given topic.",
            system_prompt=(
                "You are the WriteAgent that can write a report on a given topic. "
                "Your report should be in a markdown format. The content should be grounded in the research notes. "
                "Once the report is written, you should get feedback at least once from the ReviewAgent."
            ),
            llm=llm,
            tools=[write_report],
            can_handoff_to=["ReviewAgent", "ResearchAgent"],
        )

        review_agent = FunctionAgent(
            name="ReviewAgent",
            description="Useful for reviewing a report and providing feedback.",
            system_prompt=(
                "You are the ReviewAgent that can review the write report and provide feedback. "
                "Your review should either approve the current report or request changes for the WriteAgent to implement. "
                "If you have feedback that requires changes, you should hand off control to the WriteAgent to implement the changes after submitting the review."
            ),
            llm=llm,
            tools=[review_report],
            can_handoff_to=["WriteAgent"],
        )


        agent_workflow = AgentWorkflow(
            agents=[research_agent, write_agent, review_agent],
            root_agent=research_agent.name,
            initial_state={
                "research_notes": {},
                "report_content": "Not written yet.",
                "review": "Review required.",
            },
        )
        
        async def run_workflow():
            res = await agent_workflow.run("What are the capital cities of the Lord of the Rings kingdoms?")
            print(res.result)

        asyncio.run(run_workflow())

    def tearDown(self) -> None:
        """Clean up after each test."""
        self.maxim.cleanup()

if __name__ == '__main__':
    unittest.main()

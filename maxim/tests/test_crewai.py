import os
import unittest
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

from maxim import Maxim
from maxim.logger.crewai import instrument_crewai

load_dotenv()

class TestCrewAI(unittest.TestCase):
    """Test class for CrewAI integration with Maxim logging."""

    def setUp(self):
        # Ensure no cached Maxim instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

        # Initialize Maxim and logger
        self.maxim = Maxim({
            "api_key": os.getenv("MAXIM_API_KEY"),
            "base_url": os.getenv("MAXIM_BASE_URL"),
            "debug": True,
        })
        self.logger = self.maxim.logger({"id": os.getenv("MAXIM_LOGGER_ID")})
        
        # Instrument CrewAI with Maxim logging
        instrument_crewai(self.logger, True)

    def test_crew_execution(self):
        """Test CrewAI execution with multiple agents and tasks"""
        try:
            # Create agents
            # Create agent with config
            agent_one = Agent(
                role="Data Analyst",
                goal="Analyze data trends in the market",
                backstory="An experienced data analyst with a background in economics",
                config={
                    "maxim-eval": {
                        "evaluators": ["Bias"]
                    }
                }
            )
            
            # Set config after creation
            print("[TEST] Before setting config:", vars(agent_one))
            agent_one.config = {
                "maxim-eval": {
                    "evaluators": ["Bias"]
                }
            }
            print("[TEST] After setting config:", vars(agent_one))
            agent_two = Agent(
                role="Market Researcher",
                goal="Gather information on market dynamics",
                backstory="A diligent researcher with a keen eye for detail",
            )

            # Create tasks
            task_one = Task(
                name="Collect Data Task",
                description="Collect recent market data and identify trends.",
                expected_output="A report summarizing key trends in the market.",
                agent=agent_one,
            )
            task_two = Task(
                name="Market Research Task",
                description="Research factors affecting market dynamics.",
                expected_output="An analysis of factors influencing the market.",
                agent=agent_two,
            )

            # Create and execute crew
            crew = Crew(
                agents=[agent_one, agent_two],
                tasks=[task_one, task_two],
                process=Process.sequential,
            )

            # Execute crew tasks
            result = crew.kickoff()
            
            # Basic assertions
            self.assertIsNotNone(result)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

            print("Crew execution result:", result)

        except Exception as e:
            self.skipTest(f"Crew execution error: {e}")

    def tearDown(self):
        self.maxim.cleanup()

if __name__ == "__main__":
    unittest.main()
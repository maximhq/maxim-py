import unittest
import os

from maxim.logger import Logger
from maxim.logger.agno import instrument_agno
from maxim.tests.mock_writer import inject_mock_writer


class TestAgnoIntegration(unittest.TestCase):
    def test_agno_generation_logging(self):
        # Import Agno agent and model
        try:
            from agno.agent.agent import Agent
            from agno.models.ollama import Ollama
        except ImportError:
            self.skipTest("Agno or Ollama not available")

        # Setup Maxim logger with mock writer
        logger = Logger(
            {"id": os.getenv("MAXIM_LOG_REPO_ID", "test-repo")},
            api_key="test-api-key",
            base_url="https://app.getmaxim.ai",
        )
        mock_writer = inject_mock_writer(logger)
        instrument_agno(logger)

        # Create agent and run
        agent = Agent(model=Ollama())
        response = agent.run(message="Hello from Agno!", generation_name="hello")
        logger.flush()
        logs_raw = mock_writer.get_flushed_logs()
        logs = list(logs_raw) if logs_raw is not None else []

        # Find the generation create log
        gen_create = next(
            (
                log
                for log in logs
                if getattr(log, "entity", None)
                and getattr(log.entity, "value", None) == "generation"
                and getattr(log, "action", None) == "create"
            ),
            None,
        )
        self.assertIsNotNone(gen_create, "No generation create log found")
        data = getattr(gen_create, "data", {}) or {}
        self.assertTrue(data.get("messages"), "Generation messages should not be empty")
        self.assertIsInstance(data.get("messages"), list)
        self.assertTrue(
            any(m.get("role") == "user" for m in data.get("messages")),
            "Should have user message",
        )
        self.assertTrue(
            data.get("modelParameters") is not None,
            "Model parameters should not be None",
        )

        # Find the generation result log
        gen_result = next(
            (
                log
                for log in logs
                if getattr(log, "entity", None)
                and getattr(log.entity, "value", None) == "generation"
                and getattr(log, "action", None) == "result"
            ),
            None,
        )
        self.assertIsNotNone(gen_result, "No generation result log found")
        # Optionally check result format
        result_data = (getattr(gen_result, "data", {}) or {}).get("result", {})
        self.assertTrue(result_data, "Generation result should not be empty")


if __name__ == "__main__":
    unittest.main()

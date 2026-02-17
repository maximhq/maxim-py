import os
import socket
import unittest
from uuid import uuid4

import dotenv

from maxim.tests.openai.realtime.agent_to_agent_voice_automation import (
    AutomationConfig,
    run_duplex_voice_automation,
)

dotenv.load_dotenv()


def _openai_dns_available() -> bool:
    try:
        socket.getaddrinfo("api.openai.com", 443)
        return True
    except OSError:
        return False


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
@unittest.skipUnless(os.getenv("MAXIM_API_KEY"), "MAXIM_API_KEY not set")
@unittest.skipUnless(os.getenv("MAXIM_BASE_URL"), "MAXIM_BASE_URL not set")
@unittest.skipUnless(os.getenv("MAXIM_LOG_REPO_ID"), "MAXIM_LOG_REPO_ID not set")
@unittest.skipUnless(_openai_dns_available(), "OpenAI DNS/network unavailable")
class TestAgentToAgentVoiceAutomationIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_automation_two_turn_smoke(self):
        summary = await run_duplex_voice_automation(
            AutomationConfig(
                turns=2,
                session_name=f"A2A Smoke {uuid4().hex[:8]}",
                turn_timeout_sec=60.0,
                debug_events=False,
            )
        )
        self.assertGreaterEqual(summary.turns_completed, 2)
        self.assertGreaterEqual(summary.per_agent_response_done_count["agent_a"], 1)
        self.assertGreaterEqual(summary.per_agent_response_done_count["agent_b"], 1)

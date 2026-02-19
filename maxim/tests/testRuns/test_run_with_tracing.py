"""
E2E test for test runs with yields_output_with_tracing and logging integration.

This test mirrors the TypeScript SDK test at maxim-js/src/lib/tests/testRuns/testRun.ts.

Requirements:
- MAXIM_API_KEY: Maxim API key
- MAXIM_WORKSPACE_ID: Workspace ID for the test run
- MAXIM_LOG_REPO_ID: Log repository ID for tracing
- MAXIM_BASE_URL: (optional) Maxim API base URL

The test makes HTTP requests to the chat endpoint - ensure that service is running.
Set CHAT_URL (default: http://localhost:3001/api/v1/chat) if your agent uses a different path.

Run from project root:
  python -m maxim.tests.testRuns.test_run_with_tracing
"""

import os
import dotenv

import httpx
from maxim import Config, Maxim
from maxim.dataset import create_data_structure
from maxim.models import Data, YieldedOutput, YieldedOutputCost, YieldedOutputMeta, YieldedOutputTokenUsage

dotenv.load_dotenv()

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing {name} environment variable")
    return value


def main() -> None:
    api_key = _require_env("MAXIM_API_KEY")
    workspace_id = _require_env("MAXIM_WORKSPACE_ID")
    log_repo_id = _require_env("MAXIM_LOG_REPO_ID")
    base_url = os.environ.get("MAXIM_BASE_URL", "https://app.getmaxim.ai")

    maxim = Maxim(
        Config(
            api_key=api_key,
            base_url=base_url,
            debug=True,
            raise_exceptions=True,
        )
    )

    data_structure = create_data_structure({
        "Input": "INPUT",
        "Expected Output": "EXPECTED_OUTPUT",
        "targetLanguage": "VARIABLE",
        "nativeLanguage": "VARIABLE",
        "difficulty": "VARIABLE",
    })

    data: Data = [
        {
            "Input": "How to say Hello",
            "Expected Output": (
                'To say "Hello" in Spanish, you say "Hola." \n\nCan you try saying it? '
                '\nAlso, you can use "Hola" in different situations, just like in English! '
                '\n\nFor example: \n\n- "Hola, ¿cómo estás?" (Hello, how are you?) '
                '\n\nWould you like to learn how to respond to that question?'
            ),
            "targetLanguage": "Klingon",
            "nativeLanguage": "English",
            "difficulty": "beginner",
        },
        {
            "Input": "How to say Good",
            "Expected Output": (
                'Great question! In Spanish, "good" is "bueno." \n\n'
                "Here's how you can use it in a sentence:\n\n"
                '- "Es bueno." (It is good.)\n'
                '- "El libro es bueno." (The book is good.)\n\n'
                'Can you try to use "bueno" in a sentence'
            ),
            "targetLanguage": "Klingon",
            "nativeLanguage": "English",
            "difficulty": "beginner",
        },
    ]

    maxim_logger = maxim.logger({"id": log_repo_id})
    chat_url = os.environ.get("CHAT_URL", "http://localhost:3001/api/v1/chat")

    def output_function(row: dict, trace_id: str) -> YieldedOutput:
        """Output function that receives trace_id for logging integration."""
        with httpx.Client(timeout=30, http2=False) as client:
            response = client.post(
                chat_url,
                json={
                    "message": row["Input"],
                    "targetLanguage": row["targetLanguage"],
                    "nativeLanguage": row["nativeLanguage"],
                    "difficulty": row["difficulty"],
                },
                headers={"trace-id": trace_id},
            )
        response.raise_for_status()

        response_data = response.json()
        content = response_data.get("data", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response_data.get("data", {}).get("usage", {})

        total_tokens = usage.get("total_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)

        return YieldedOutput(
            data=content,
            meta=YieldedOutputMeta(
                usage=YieldedOutputTokenUsage(
                    total_tokens=total_tokens,
                    completion_tokens=completion_tokens,
                    prompt_tokens=prompt_tokens,
                ),
                cost=YieldedOutputCost(
                    input_cost=prompt_tokens * 0.0015,
                    output_cost=completion_tokens * 0.002,
                    total_cost=total_tokens * 0.00175,
                ),
            ),
        )

    test_run = (
        maxim.create_test_run("testing tests + logs", workspace_id)
        .with_data_structure(data_structure)
        .with_data(data)
        .with_concurrency(2)
        .yields_output_with_tracing(
            output_function,
            maxim_logger,
            disable_default_trace_creation=True,
        )
    )

    result = test_run.run()
    print(f"Test run completed: {result}")


if __name__ == "__main__":
    main()

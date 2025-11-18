# Maxim SDK

<div style="display: flex; justify-content: center; align-items: center;margin-bottom:20px;">
<img src="https://cdn.getmaxim.ai/third-party/sdk.png">
</div>

This is Python SDK for enabling Maxim observability. [Maxim](https://www.getmaxim.ai) is an enterprise grade evaluation and observability platform.

## How to integrate

### Install

```
pip install maxim-py
```

### Documentation

You can find detailed documentation and available integrations [here](https://www.getmaxim.ai/docs/sdk/python/overview).

### OpenAI Responses (one-line integration)

Plug-and-play observability for the OpenAI Responses API. Works for both sync and streaming calls and captures model, parameters, messages, output text, tool calls, and errors without changing your OpenAI usage.

Sync example:

```python
from openai import OpenAI
from maxim import Maxim
from maxim.logger.openai import MaximOpenAIClient

maxim = Maxim({"api_key": os.getenv("MAXIM_API_KEY")})
logger = maxim.logger({"id": str(os.getenv("MAXIM_LOG_REPO_ID"))})
oai = OpenAI()  # reads OPENAI_API_KEY from environment
client = MaximOpenAIClient(oai, logger=logger)

res = client.responses.create(
    model="gpt-4.1-mini",
    input="Write a one-sentence summary of Maxim."
)

print(res.output_text)
```

Streaming example:

```python
from openai import OpenAI
from maxim import Maxim
from maxim.logger.openai import MaximOpenAIClient

maxim = Maxim({"api_key": os.getenv("MAXIM_API_KEY")})
logger = maxim.logger({"id": str(os.getenv("MAXIM_LOG_REPO_ID"))})
oai = OpenAI()
client = MaximOpenAIClient(oai, logger=logger)

with client.responses.stream(
    model="gpt-4.1-mini",
    input="Stream a short poem about observability."
) as stream:
    for event in stream:
        print(event, end="")
    final = stream.get_final_response()
```

Async examples:

Non-streaming (async):

```python
import os
import asyncio
from openai import AsyncOpenAI
from maxim import Maxim
from maxim.logger.openai import MaximOpenAIAsyncClient


async def main():
    maxim = Maxim({"api_key": os.getenv("MAXIM_API_KEY")})
    logger = maxim.logger({"id": str(os.getenv("MAXIM_LOG_REPO_ID"))})
    oai = AsyncOpenAI()  # reads OPENAI_API_KEY from environment
    client = MaximOpenAIAsyncClient(oai, logger=logger)

    res = await client.responses.create(
        model="gpt-4.1-mini",
        input="Write a one-sentence summary of Maxim.",
    )
    print(res.output_text)


if __name__ == "__main__":
    asyncio.run(main())
```

Streaming (async):

```python
import os
import asyncio
from openai import AsyncOpenAI
from maxim import Maxim
from maxim.logger.openai import MaximOpenAIAsyncClient


async def main():
    maxim = Maxim({"api_key": os.getenv("MAXIM_API_KEY")})
    logger = maxim.logger({"id": str(os.getenv("MAXIM_LOG_REPO_ID"))})
    oai = AsyncOpenAI()
    client = MaximOpenAIAsyncClient(oai, logger=logger)

    async with client.responses.stream(
        model="gpt-4.1-mini",
        input="Stream a short poem about observability.",
    ) as stream:
        async for event in stream:
            print(event, end="")
        final = await stream.get_final_response()


if __name__ == "__main__":
    asyncio.run(main())
```

Optional headers to control trace metadata (pass via `extra_headers` on `create`/`stream`):

```python
extra_headers = {
    "x-maxim-trace-id": "my-trace-id",
    "x-maxim-generation-name": "homepage-summary",
    "x-maxim-session-id": "abc123",
}

res = client.responses.create(
    model="gpt-4.1-mini",
    input="Hello",
    extra_headers=extra_headers,
)
```

### Cookbook

See [cookbook/agno_agent.py](cookbook/agno_agent.py) for an example of tracing an Agno agent.

#### Langchain module compatibility

|                                                             | Anthropic | Bedrock Anthropic | Bedrock Meta | OpenAI | Azure                                        |
| ----------------------------------------------------------- | --------- | ----------------- | ------------ | ------ | -------------------------------------------- |
| Chat (0.3.x)                                                | ✅        | ✅                | ✅           | ✅     | ✅                                           |
| Chat (0.1.x)                                                | ✅        | ✅                | ✅           | ✅     | ✅                                           |
| Tool call (0.3.x)                                           | ✅        | ✅                | ❓           | ✅     | ✅                                           |
| Tool call (0.1.x)                                           | ✅        | ✅                | ✅           | ✅     | ✅                                           |
| Chain (via LLM) (0.3.x)                                     | ✅        | ✅                | ✅           | ✅     | ✅                                           |
| Chain (via LLM) (0.1.x)                                     | ✅        | ✅                | ✅           | ✅     | ✅                                           |
| Streaming (0.3.x)                                           | ✅        | ✅                | ✅           | ✅     | ✳️ Token usage is not supported by Langchain |
| Streaming (0.1.x) Token usage is not supported by Langchain | ✳️        | ✳️                | ✳️           | ✳️     | ✳️                                           |
| Agent (0.3.x)                                               | ⛔️       | ⛔️               | ⛔️          | ⛔️    | ⛔️                                          |
| Agent (0.1.x)                                               | ⛔️       | ⛔️               | ⛔️          | ⛔️    | ⛔️                                          |

> Please reach out to us if you need support for any other package + provider + classes.

### Litellm

| completion | acompletion | fallback | Prompt Management |
| ---------- | ----------- | -------- | ----------------- |
| ✅         | ✅          | ⛔️      | ⛔️               |

### LiveKit (Alpha)

| Provider             | Audio | ToolCalls | Video |
| -------------------- | ----- | --------- | ----- |
| OpenAI (RealtimeAPI) | ✅    | ⛔️       | ⛔️   |
| Gemini (RealtimeAPI) | ✅    | ⛔️       | ⛔️   |

# Setting up this repository

1. Clone [this](https://github.com/maximhq/maxim-py) repository.
2. Make sure you have installed [uv](https://docs.astral.sh/uv/) on your computer.
3. Run `uv sync`.

## Version changelog

### 3.13.6
- feat: Adds various voice observability integrations
  - ElevenLabs STT-TTS: Adds ElevenLabs STT-TTS logging
  - OpenAI Realtime: Extends the OpenAI one-line integration to support realtime behavior
  - Audio support for OpenAI Agents: Extends `MaximOpenAITraceProcessor` to work with voice agents

### 3.13.5
- feat: Adds ability to override `startTimestamp` and `endTimestamp` for traces and sessions

### 3.13.4
- feat: Adds support for chat.completions.parse for OpenAI SDK

### 3.13.3
- feat: Adds callbacks for langchain tracer to capture trace ids and other events

### 3.13.2
- feat: Adds ability to add tags to test runs using the `with_tags` method

### 3.13.1

- feat: Adds OpenAI Responses one-line integration for the async client via `MaximOpenAIAsyncClient.responses` (non-streaming and streaming), matching sync behavior and `extra_headers` support.

### 3.13.0

- feat: Introduces OpenAI Responses one-line integration (sync and streaming) via `MaximOpenAIClient.responses`

### 3.12.1

- fix: Fixes Google ADK integration to fix trace user input message.

### 3.12.0

- feat: Added support for OpenAI Responses API format in addition to Chat Completion API
- feat: Added TTL-based caching (60s) for prompt version number single-condition fetches
- feat: Added new `prompt_version_number()` method in QueryBuilder for convenient version-specific queries
- improvement: Enhanced `generation_parser` to detect and handle OpenAI Responses API structure

### 3.11.4

- fix: Fixes race condition in LiveKit realtime tracing
- fix: Fixes import errors for Gemini and Google realtime session imports

### 3.11.3

- fix: Fixed Nested Spans issue for Google ADK

### 3.11.2

- fix: Fixed Google ADK integration to support spans for agent hand offs and tool calls.

### 3.11.1

- feat: Added `ContainerManager` to Langchain to manage containers
- fix: Fixes `trace.end` in Langchain integration `MaximLangchainTracer`

### 3.11.0

- feat: Added observability for google adk

### 3.10.10

- feat: Added case for `commit_user_turn` for LiveKit logs

### 3.10.9

- fix: Fixed session audio silences for LiveKit Realtime session implementation

### 3.10.8

- feat: Added Pydantic AI Single Line Integration

### 3.10.7

- fix: Fixes local data test runs

### 3.10.6

- feat: Added smolagents single line integration for observability
- fix: Fixes CrewAI Single Line Integration, modified `handle_non_streaming_wrapper` function signature as expected by crewai

### 3.10.5

- feat: Adds `Agent` instrumentation for LiveKit
- fix: Fixes import breakage

### 3.10.4 (💥 Yanked)

- feat: Disables internal logs for LiveKit

### 3.10.3 (💥 Yanked)

- **feat** Added file support to `add_dataset_entries`
- **Breaking Change** Updated dataclasses of `Variables` and `DatasetEntry`

### 3.10.2

- feat: Adds node-level evals support for CrewAI
- feat: Adds LlamaIndex instrumentation
- fix: Refactors `generation_parser` validation to extend to both `dict` and `ChatMessageToolCall` types

### 3.10.1

- feat: Added STT-LLM-TTS pipeline tracing integration for LiveKit

### 3.10.0

- **feat**: Added comprehensive multimodal content support for vision-enabled models
  - **MULTIMODAL**: Full support for text and image content in prompt messages
  - **TYPES**: New `CompletionRequestContent`, `CompletionRequestTextContent`, `CompletionRequestImageUrlContent` types
  - **ENHANCED**: Updated `Message` class to handle both string and multimodal content arrays
- **feat**: Added `deployment_id` support to prompt configurations
  - **ENHANCEMENT**: Optional `deployment_id` field now available in `Prompt`, `RunnablePrompt`, and `PromptVersionConfig` classes for prompts with `Azure` provider
- **improvement**: Enhanced `RunnablePrompt.run()` with proper `image_urls` parameter support
  - **MULTIMODAL**: Better integration for vision-enabled model workflows

### 3.9.14

- chore: Import fixes for crewai integration.

### 3.9.13

- fix: Fixes tool call parsing for streaming
- fix: Some minor fixes in OpenAI, Fireworks wrappers.

### 3.9.12

- feat: Adds one line integration for Fireworks Build SDK

### 3.9.11

- feat: Adds support for PROMPT_TYPE_CUSTOM for langchain

### 3.9.10

- fix: removes signal handlers from the package.

### 3.9.9

- feat: Adds one line integration for Together AI SDK
- feat: Adds one line integration for Groq SDK
- feat: Adds sink support for logger to write to multiple repos.

### 3.9.8

- feat: Adds one line integration for tracing Agno agents.

### 3.9.7

- improvement: Adds support for uploading large payloads as part of logs.

### 3.9.6

- improvement: Increased connection pool max size to 20 for more connections at high throughput.
- improvement: Moves network stack from requests to httpx for better stability

### 3.9.5

- feat: Adds new query type for querying prompts, prompt chains and folders called mutli-select. [Learn more](https://www.getmaxim.ai/docs/offline-evals/via-sdk/prompts/querying-prompts#querying-a-single-prompt).

### 3.9.4

- chore: Changed scribe (Maxim Logger) default level to warning.

### 3.9.3

- improvement: Adds try except for all LiveKit callbacks call and gracefully moves forward with tracing.

### 3.9.2

- improvement: Improves the network connection error handling for connection pool.

### 3.9.1

- fix: Fixes `enable_prompt_management` method bug.

### 3.9.0

- fix: Some minor fixes in Maxim log repo checks, Anthropic client and Gemini client
- feat: Improved memory management and better interrupt detection for LiveKit + Gemini.
- feat: LiveKit + Tool call support is live.
- feat: LiveKit support is now in beta (up from Alpha).

### 3.8.5

- chore: Adds session_id and room_id

### 3.8.4

- fix: Fixes gemini + langchain integration to capture None finish_reason and usage_metadata

### 3.8.3

- fix: Fixes chunk and chat auto-tracing for gemini client

### 3.8.2

- feat: Adds one line integration for Portkey AI
- fix: Fixes tool call parsing for OpenAI one line integration.
- feat: Adds auto attachment parsing for vision models
- fix: Exposes YieldedOutputTokens and YieldedOutputCost classes

### 3.8.1

- feat: Adds files support for test_runs
- feat: Adds mistral tracing support

### 3.8.0

- breaking change: We have renamed a few entities used in test runs.

### 3.7.3

- chore: Updates default log level of scribe to debug

### 3.7.2

- feat: LiveKit support for Google and OpenAI
- chore: improvements in crewai logging integration
- chore: deprecated span.output() method removed
- feat: LiveKit one line integration (alpha)

### 3.7.1

- fix: Signal registration only happens if the current thread is main thread

### 3.7.0

- feat: Prompt, PromptVersionConfig, and RunnablePrompt now expose a `provider` field to indicate the LLM provider (e.g., 'openai', 'anthropic').
- improvement: Maxim SDK listens for `atexit` and termination signals, and triggers cleanup automatically

### 3.6.4

- chore: crewai python 3.9+ support

### 3.6.3

- fix: now reports metadata errors silently
- fix: minor fixes to crew-ai instrumentation

### 3.6.2

- fix: minor version check fixes.

### 3.6.1

- feat: crew-ai intercept
- fix: trace tag fixes for langchain

### 3.6.0

- feat: Adds attachment support (beta): [Read more](https://www.getmaxim.ai/docs/observe/how-to/log-your-application/add-attachments)

### 3.5.8

- fix: Fixes anthropic one line integration
- chore: Fixes anthropic messages parsing for streaming client

### 3.5.7

- chore: Added extra guards for lang-graph evaluation config.

### 3.5.6

- chore: Adds new Maxim SDK level logger. You can set specific level for Maxim SDK by `logging.getLogger('maxim').setLevel(logging.DEBUG)`
- fix: minor cleanups on Langchain tracer.

### 3.5.5

- feat: adds special handling for apps running on AWS lambda. As runtime execution is unpredictable, logger.flush() now pushes logs immediately vs submitting to a thread pool worker.
- chore: all logs emitted form the SDK now has [MaximSDK] prefix.

### 3.5.4

- fix: adds special handling when langchain streaming response handler raises an exception in user-land.

### 3.5.3

- fix: fixes empty cache issue for prompt management

### 3.5.2 (💥 Yanked)

- chore: adds max limit to the commit log queue size
- chore: auto flush when the writer level max in memory message size reaches
- chore: adds global container list for langchain tracer to use it across multiple instances

### 3.5.1 (💥 Yanked)

- chore: adds custom trace-id support for MaximOpenAIClient

### 3.5.0 (💥 Yanked)

- feat: adds error component
- deprecate: old Config classes - now all logging constructs support TypedDict
- feat: adds one line integration for OpenAI
- fix: fixes agents import in OpenAI SDK

### 3.4.17

- fix: fixes langchain callback tracer

### 3.4.16

- chore: adds support to wrap LiteLLMProxy tracer for docker deployments

### 3.4.15

- feat: adds new MaximLiteLLMProxyTracer file for LitellmProxy logger

### 3.4.14 (💥 Yanked)

- feat: adds new MaximLiteLLMProxyTracer file for LitellmProxy logger

### 3.4.13

- chore: adds ID validation on client side.

### 3.4.12

- feat: adds support for bedrock client

### 3.4.11

- chore: Some minor bugfixes for langhchain handler (its them not us)
- fix: handles some network level exceptions during connection resets.

### 3.4.10

- feat: OpenAI agents tracing out of beta. And we are also on OpenAI docs -

### 3.4.9

- feat: adds support for running test runs using prompt chains

### 3.4.8

- fix: fixes testruns using dataset, and runs using local workflows.

### 3.4.7 (🚧 Yanked: In favor of broken test-runs via SDK)

- fix: fixes incorrect import for TypedDict

### 3.4.6 (🚧 Yanked: In favor of broken test-runs via SDK)

- feat: openai-agents adds session support, adds error support for llm calls

### 3.4.5 (🚧 Yanked: In favor of broken test-runs via SDK)

- feat: OpenAI agents tracing support (beta)

### 3.4.4 (🚧 Yanked: In favor of broken test-runs via SDK)

- feat: Generation messages adds support for dicts

### 3.4.3 (🚧 Yanked: In favor of broken test-runs via SDK)

- fix: Handles optional PromptResponse fields gracefully.

### 3.4.2 (🚧 Yanked: In favor of broken test-runs via SDK)

- feat: Adds support for prompt and prompt chain run

### 3.4.1 (🚧 Yanked: In favor of broken test-runs via SDK)

- fix: Resolved duplicate serialization of metadata entries

### 3.4.0 (🚧 Yanked: In favor of broken test-runs via SDK)

- Breaking change: Prompt and Prompt chain object properties are now with snake cases
- fix: Prompt chain nodes are properly parsed in all cases

### 3.3.9 (🚧 Yanked: In favor of broken test-runs via SDK)

- fix: fixes litellm pre_api_call message parsing

### v3.3.8 (🚧 Yanked: In favor of broken test-runs via SDK)

- fix: updates create test run api to use v2 api

### v3.3.7 (🚧 Yanked: In favor of broken test-runs via SDK)

- fix: handles marking test run as failed if the test run raises error at any point after creating it on the platform.
- feat: adds support for `context_to_evaluate` in `with_prompt_version_id` and `with_workflow_id` (by passing it as the second parameter) to be able to choose whichever variable or dataset column to use as context to evaluate, as opposed to only having the dataset column as context through the `CONTEXT_TO_EVALUATE` datastructure mapping.

### v3.3.6

- fix: fixes garbled message formatting when invalid testrun config is passed to the TestRunBuilder

### v3.3.5

- chore: now sdk propagates system errors in formatted structures (specifically for test runs)

### v3.3.4

- fix: add missing deps in the requirement

### v3.3.3

- chore: minor bug fixes

### v3.3.2

- feat: adds support for gemini outputs
- feat: adds local evaluator support for test runs

### v3.3.1

- chore: Litellm failure exceptions will be sent to the default logger.

### v3.3.0

- feat: Adds litellm support (Beta)

### v3.2.3

- fix: Fixes duplicate container ids for langchain tracer

### v3.2.2

- fix: Langgraph capture fixes
- chore: Adds missing docstrings

### v3.2.1

- fix: Adds support for dict as an output to yields_output function during test runs.

### v3.2.0

- fix: Fixed dependency issues

### v3.1.0 (🚧 Yanked)

- feat: Adds new flow to trigger test runs via Python SDK
- fix: Minor bug fixes

### v3.0.1 [Breaking changes](https://www.getmaxim.ai/docs/sdk/python/upgrading-to-v3)

- beta release
- feat: New decorators support for tracing, langchain and langgraph

### v3.0.0rc6

- feat: Adds new decorator for langgraph. @langgraph_agent
- feat: Adds support for chains in langchain tracer
- fix: Some minor bug fixes

### v3.0.0rc5

- chore: Keeps logger till function call context is present

### v3.0.0rc4

- fix: Fixes automatic retrieval capture from vector dbs

### v3.0.0rc3

- fix: Fixes langchain_llm_call to handle chat models

### v3.0.0rc2

- fix: Minor bug fixes

### v3.0.0rc1

- Check [upgrade steps](https://www.getmaxim.ai/docs/sdk/python/upgrading-to-v3)
- feat: Adds new decorators flow to simplify tracing
- chore: apiKey and baseUrl parameters in MaximConfig are now api_key and base_url respectively.

### v2.0.0 (Breaking changes)

- feat: Jinja 2.0 variables support

### v1.5.13

- fix: Fixes issue where model was None for some prompt versions.

### v1.5.12

- fix: Fixes edge case of race condition while fetching prompts, prompt chains and folders.

### v1.5.11

- fix: Fixes import of dataclasse

### v1.5.10

- feat: Adds new config called `raise_exceptions`. Unless this is set to `True`, the SDK will not raise any exceptions.

### v1.5.9

- Chore - Removes raising alert when repo not found

### v1.5.8

- fix - Removes a no-op command for retrieval
- fix - Fixes retrieval output command

### v1.5.7

- feat - Supports 0.1.x langchain

### v1.5.6

- chore - Improved langchain support

### v1.5.5

- chore - Improves cleanups for log writer for quick returns.

### v1.5.4

- chore - Improved fs access checks.
- chore - Fixes threading locks for periodic syncs in Python3.9

### v1.5.3

- chore - Adds lambda env support for SDK with no access to filesystem.

### v1.5.2

- feat - Adds support to new langchain_openai.AzureChatOpenAI class in langchain tracer

### v1.5.1

- fix - Adds Python 3.9 compatibility

### v1.5.0

- chore - Updates connection pool to use session that enforces re-connects before making API calls.

### v1.4.5

- chore - Adds backoff retries to failed REST calls.

### v1.4.4

- chore - langchain becomes optional dependency

### v1.4.3

- fix - connection pooling for network calls.
- fix - connection close issue.

### v1.4.2 (🚧 Yanked)

- fix - connection close issue

### v1.4.1

- Adds validation for provider in generation

### v1.4.0

- Now generation.result accepts
  - OpenAI chat completion object
  - Azure OpenAI chat completion object
  - Langchain LLMResult, AIMessage object

### v1.3.4

- fix: Fixes message_parser

### v1.3.2

- fix: Fixes utility function for langchain to parse AIMessage into Maxim logger completion result

### v1.3.1

- feat: Adds tool call parsing support for Langchain tracer

### v1.3.0

- feat: Adds support for ChatCompletion in generations
- feat: Adds type safety for retrieval results

### v1.2.7

- fix: Bug fix where input sent with trace.config was getting overridden with None

### v1.2.6

- chore: Adds `trace.set_input` and `trace.set_output` methods to control what to show in logs dashboard

### v1.2.5

- chore: Removes one no_op command while creating spans
- fix: Minor bug fixes

### v1.2.1

- fix: Fixed MaximLangchainTracer error logging flow.

### v1.2.0

- feat: Adds langchain support
- chore: Adds local parsers to validate payloads on client side

### v1.1.0

- fix: Minor bug fixes around log writer cleanup

### v1.0.0

- Public release

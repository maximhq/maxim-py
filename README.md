# Maxim SDK

<div style="display: flex; justify-content: center; align-items: center;margin-bottom:20px;">
<img src="https://cdn.getmaxim.ai/third-party/sdk.png">
</div>

This is Python SDK for enabling Maxim observability. [Maxim](https://www.getmaxim.ai?ref=npm) is an enterprise grade evaluation and observability platform.

## How to integrate

### Install

```
pip install maxim-py
```

### Initialize Maxim logger

```console
MAXIM_API_KEY=
MAXIM_LOG_REPO_ID=
```

```python
# SDK picks up env variables

from maxim import Maxim
maxim = Maxim()
```

### Start sending traces

```python
# Initializing logger
logger = maxim.logger()
# Initializing a new trace
trace = logger.trace(TraceConfig(id="trace-id",name="trace-name",tags={"key":"value"}))
# Creating the generation
generation = trace.generation(GenerationConfig(id=str(uuid4()), model="text-davinci-002", provider="azure", model_parameters={"temperature": 0.7, "max_tokens": 100}))
# Making LLM call
completion = self.client.completions.create(
   model="text-davinci-002",
   prompt="Translate the following English text to French: 'Hello, how are you?'",
   max_tokens=100,
   temperature=0.7
)
# Updating generation
generation.result(completion)
# Ending trace
trace.end()
```

## Integrations with other frameworks

### Langchain

We have built in Langchain tracer support

```python
logger = self.maxim.logger(LoggerConfig(id=repoId))
trace_id = str(uuid4())
trace = logger.trace(TraceConfig(
   id=trace_id, name="pre-defined-trace"))

model = OpenAI(callbacks=[MaximLangchainTracer(logger)],api_key=openAIKey)
messages = [
   (
         "system",
         "You are a helpful assistant that translates English to French. Translate the user sentence.",
   ),
   ("human", "I love programming."),
]
model.invoke(messages, config={
   "metadata": {
         "maxim": {
            "trace_id": trace_id,
            "generation_name": "get-answer",
            "generation_tags": {
               "test": "123"
            }
         }
   }
})
trace.event(id=str(uuid4()), name="test event")
trace.end()
```

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

| Provider | Audio | ToolCalls | Video | 
| ---------- | ---------- | ----------- | -------- 
| OpenAI (RealtimeAPI)  | ✅         | ⛔️          | ⛔️      
| Gemini (RealtimeAPI)  | ✅         | ⛔️          | ⛔️      


## Version changelog


### 3.7.2

- LiveKit support for Google and OpenAI
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

### v3.0.1 [Breaking changes](https://www.getmaxim.ai/docs/sdk/upgrading-to-v3)

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

- Check [upgrade steps](https://www.getmaxim.ai/docs/sdk/upgrading-to-v3)
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

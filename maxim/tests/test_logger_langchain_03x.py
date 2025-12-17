############################################
# This file uses langchain 0.3.x
###########################################

import base64
import logging
import os
import unittest
from pathlib import Path
from time import sleep
from uuid import uuid4

import dotenv
from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_anthropic import AnthropicLLM, ChatAnthropic
from langchain_community.llms.openai import AzureOpenAI, OpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from maxim.logger.components.generation import GenerationConfig
from maxim.logger.components.span import SpanConfig
from maxim.logger.components.trace import TraceConfig
from maxim.logger.langchain.tracer import MaximLangchainTracer
from maxim.logger.logger import LoggerConfigDict
from maxim.maxim import Maxim

# Load environment variables from .env file
dotenv.load_dotenv()

awsAccessKeyId = os.getenv("BEDROCK_ACCESS_KEY_ID")
awsAccessKeySecret = os.getenv("BEDROCK_SECRET_ACCESS_KEY")
azureOpenAIBaseUrl = os.getenv("AZURE_OPENAI_BASE_URL")
azureOpenAIKey = os.getenv("AZURE_OPENAI_KEY")
openAIKey = os.getenv("OPENAI_API_KEY")
apiKey = os.getenv("MAXIM_API_KEY")
baseUrl = os.getenv("MAXIM_BASE_URL")
repoId = os.getenv("MAXIM_LOG_REPO_ID")
anthropicApiKey = os.getenv("ANTHROPIC_API_KEY")

# # Callback handler for langchain
logging.basicConfig(level=logging.INFO)


@tool
def addition_tool(a, b):
    """Add two integers together."""
    return a + b


@tool
def subtraction_tool(a, b):
    """Subtract two integers."""
    return a - b


class TestLoggingUsingLangchain(unittest.TestCase):
    def setUp(self):
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.maxim = Maxim({"api_key": apiKey, "base_url": baseUrl})

    def test_generation_chat_prompt(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = OpenAI(callbacks=[MaximLangchainTracer(logger)], api_key=openAIKey)
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "I love programming."),
        ]
        result = model.invoke(messages)
        print(f"result {result}")

    def test_generation_chat_prompt_with_external_trace(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        trace_id = str(uuid4())
        trace = logger.trace(TraceConfig(id=trace_id, name="pre-defined-trace"))

        model = OpenAI(callbacks=[MaximLangchainTracer(logger)], api_key=openAIKey)
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "I love programming."),
        ]
        model.invoke(
            messages,
            config={
                "metadata": {
                    "maxim": {
                        "trace_id": trace_id,
                        "generation_name": "get-answer",
                        "generation_tags": {"test": "123"},
                    }
                }
            },
        )
        trace.event(id=str(uuid4()), name="test event")
        trace.end()

    def test_generation_chat_prompt_chat_model(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = ChatOpenAI(callbacks=[MaximLangchainTracer(logger)], api_key=openAIKey)
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "How are you doing."),
        ]
        model.invoke(messages)

    def test_generation_chat_prompt_azure_chat_model(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = AzureChatOpenAI(
            api_key=azureOpenAIKey,
            model="gpt-4o-mini",
            azure_endpoint=azureOpenAIBaseUrl,
            callbacks=[MaximLangchainTracer(logger)],
            api_version="2024-02-01",
        )
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "How are you doing."),
        ]
        result = model.invoke(
            messages, config={"metadata": {"maxim": {"trace_tags": {"a": "asdasd"}}}}
        )
        print(result)
        logger.flush()

    def test_generation_chat_prompt_azure_chat_model_with_streaming(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = AzureChatOpenAI(
            api_key=azureOpenAIKey,
            model="gpt-4o-mini",
            azure_endpoint=azureOpenAIBaseUrl,
            callbacks=[MaximLangchainTracer(logger)],
            api_version="2024-02-01",
            streaming=True,
        )
        messages = [
            (
                "system",
                "You are essay writer",
            ),
            ("human", "what's up with rahul dravid?"),
        ]
        for chunk in model.stream(messages):
            # print(chunk)
            pass
        logger.flush()

    # def test_generation_chat_prompt_bedrock_sonnet_35_chat_model(self):
    #     logger = self.maxim.logger(LoggerConfig(id=repoId))
    #     model = ChatBedrock(
    #         region="us-east-1",
    #         aws_access_key_id=awsAccessKeyId,
    #         aws_secret_access_key=awsAccessKeySecret,
    #         model_kwargs={"temperature": 0.1},
    #         provider="anthropic",
    #         model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    #         callbacks=[MaximLangchainTracer(logger)],
    #     )
    #     messages = [
    #         (
    #             "system",
    #             "You are a helpful assistant that translates English to French. Translate the user sentence.",
    #         ),
    #         ("human", "How are you doing."),
    #     ]
    #     result = model.invoke(messages)
    #     print(f"Model response: {result}")
    #     logger.flush()

    # def test_generation_chat_prompt_bedrock_sonnet_35_chat_mode_streaming(self):
    #     logger = self.maxim.logger(LoggerConfig(id=repoId))
    #     model = ChatBedrock(
    #         region="us-east-1",
    #         aws_access_key_id=awsAccessKeyId,
    #         aws_secret_access_key=awsAccessKeySecret,
    #         model_kwargs={"temperature": 0.1},
    #         provider="anthropic",
    #         model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    #         callbacks=[MaximLangchainTracer(logger)],
    #         streaming=True,
    #     )
    #     messages = [
    #         (
    #             "system",
    #             "You are a helpful assistant that translates English to French. Translate the user sentence.",
    #         ),
    #         ("human", "How are you doing."),
    #     ]
    #     for chunk in model.stream(messages):
    #         print(chunk)
    #         pass
    #     logger.flush()

    # def test_generation_chat_prompt_bedrock_sonnet_3_model_with_tool_call(self):
    #     logger = self.maxim.logger(LoggerConfig(id=repoId))
    #     model = ChatBedrock(
    #         region="us-east-1",
    #         model_kwargs={"temperature": 0.1},
    #         provider="anthropic",
    #         model="anthropic.claude-3-sonnet-20240229-v1:0",
    #         callbacks=[MaximLangchainTracer(logger)],
    #     )
    #     messages = [
    #         (
    #             "system",
    #             "You are a math expert",
    #         ),
    #         ("human", "add 3 and 3"),
    #     ]
    #     llm = model.bind_tools([addition_tool, subtraction_tool])
    #     result = llm.invoke(messages)
    #     print(f"Model response: {result}")
    #     logger.flush()

    # def test_generation_chat_prompt_bedrock_sonnet_35_model_with_tool_call(self):
    #     logger = self.maxim.logger(LoggerConfig(id=repoId))
    #     model = ChatBedrock(
    #         region="us-east-1",
    #         model_kwargs={"temperature": 0.1},
    #         provider="anthropic",
    #         model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    #         callbacks=[MaximLangchainTracer(logger)],
    #     )
    #     messages = [
    #         (
    #             "system",
    #             "You are a math expert",
    #         ),
    #         ("human", "add 3 and 3"),
    #     ]
    #     llm = model.bind_tools([addition_tool, subtraction_tool])
    #     result = llm.invoke(messages)
    #     print(f"Model response: {result}")
    #     logger.flush()

    # def test_generation_chat_prompt_bedrock_sonnet_3_chat_model(self):
    #     logger = self.maxim.logger(LoggerConfig(id=repoId))
    #     model = ChatBedrock(
    #         region="us-east-1",
    #         aws_access_key_id=awsAccessKeyId,
    #         aws_secret_access_key=awsAccessKeySecret,
    #         model_kwargs={"temperature": 0.1},
    #         provider="anthropic",
    #         model="anthropic.claude-3-sonnet-20240229-v1:0",
    #         callbacks=[MaximLangchainTracer(logger)],
    #     )
    #     messages = [
    #         (
    #             "system",
    #             "You are a helpful assistant that translates English to French. Translate the user sentence.",
    #         ),
    #         ("human", "How are you doing."),
    #     ]
    #     result = model.invoke(messages)
    #     print(f"Model response: {result}")
    #     logger.flush()

    # def test_generation_chat_prompt_bedrock_sonnet_3_chat_model_old_chain(self):
    #     logger = self.maxim.logger(LoggerConfig(id=repoId))
    #     model = ChatBedrock(
    #         region="us-east-1",
    #         model_kwargs={"temperature": 0.1},
    #         provider="anthropic",
    #         model="anthropic.claude-3-sonnet-20240229-v1:0",
    #         callbacks=[MaximLangchainTracer(logger)],
    #     )
    #     messages = [
    #         (
    #             "system",
    #             "You are a helpful assistant that translates English to French. Translate the user sentence.",
    #         ),
    #         ("human", "{question}"),
    #     ]
    #     llmChain = LLMChain(
    #         llm=model, prompt=ChatPromptTemplate.from_messages(messages)
    #     )
    #     question = "How are you doing?"
    #     result = llmChain.invoke(input={"question": question})
    #     print(f"Model response: {result}")
    #     logger.flush()

    # def test_generation_chat_prompt_bedrock_sonnet_3_chat_model_without_runnable_sequence(
    #     self,
    # ):
    #     logger = self.maxim.logger(LoggerConfig(id=repoId))
    #     model = ChatBedrock(
    #         region="us-east-1",
    #         model_kwargs={"temperature": 0.1},
    #         provider="anthropic",
    #         model="anthropic.claude-3-sonnet-20240229-v1:0",
    #         callbacks=[MaximLangchainTracer(logger)],
    #     )
    #     messages = [
    #         (
    #             "system",
    #             "You are a helpful assistant that translates English to French. Translate the user sentence.",
    #         ),
    #         ("human", "{question}"),
    #     ]
    #     prompt = ChatPromptTemplate.from_messages(messages=messages)
    #     chain = prompt | model | StrOutputParser()
    #     question = "How are you doing?"
    #     result = chain.invoke(input={"question": question})
    #     print(f"Model response: {result}")
    #     logger.flush()

    # def test_generation_chat_prompt_bedrock_sonnet_3_chat_model_with_runnable_sequence(
    #     self,
    # ):
    #     logger = self.maxim.logger(LoggerConfig(id=repoId))
    #     model = ChatBedrock(
    #         region="us-east-1",
    #         model_kwargs={"temperature": 0.1},
    #         provider="anthropic",
    #         model="anthropic.claude-3-sonnet-20240229-v1:0",
    #         callbacks=[MaximLangchainTracer(logger)],
    #     )
    #     messages = [
    #         (
    #             "system",
    #             "You are a helpful assistant that translates English to French. Translate the user sentence.",
    #         ),
    #         ("human", "{question}"),
    #     ]
    #     prompt = ChatPromptTemplate.from_messages(messages=messages)
    #     chain = RunnableSequence(prompt, model)
    #     question = "How are you doing?"
    #     result = chain.invoke(input={"question": question})
    #     print(f"Model response: {result}")
    #     logger.flush()

    # def test_generation_chat_prompt_bedrock_haiku_3_chat_model(self):
    #     logger = self.maxim.logger(LoggerConfig(id=repoId))
    #     model = ChatBedrock(
    #         region="us-east-1",
    #         model_kwargs={"temperature": 0.1},
    #         provider="anthropic",
    #         model="anthropic.claude-3-haiku-20240307-v1:0",
    #         callbacks=[MaximLangchainTracer(logger)],
    #     )
    #     messages = [
    #         (
    #             "system",
    #             "You are a helpful assistant that translates English to French. Translate the user sentence.",
    #         ),
    #         ("human", "How are you doing."),
    #     ]
    #     result = model.invoke(messages)
    #     print(f"Model response: {result}")
    #     logger.flush()

    # def test_generation_chat_prompt_bedrock_llama_chat_model(self):
    #     logger = self.maxim.logger(LoggerConfig(id=repoId))
    #     model = ChatBedrock(
    #         region="us-east-1",
    #         aws_access_key_id=awsAccessKeyId,
    #         aws_secret_access_key=awsAccessKeySecret,
    #         model_kwargs={"temperature": 0.1},
    #         provider="meta",
    #         model="meta.llama3-8b-instruct-v1:0",
    #         callbacks=[MaximLangchainTracer(logger)],
    #     )
    #     messages = [
    #         (
    #             "system",
    #             "You are a helpful assistant that translates English to French. Translate the user sentence.",
    #         ),
    #         ("human", "How are you doing."),
    #     ]
    #     result = model.invoke(messages)
    #     print(f"Model response: {result}")
    #     logger.flush()

    # def test_generation_chat_prompt_bedrock_llama_chat_model_streaming(self):
    #     logger = self.maxim.logger(LoggerConfig(id=repoId))
    #     model = ChatBedrock(
    #         region="us-east-1",
    #         aws_access_key_id=awsAccessKeyId,
    #         aws_secret_access_key=awsAccessKeySecret,
    #         model_kwargs={"temperature": 0.1},
    #         provider="meta",
    #         model="meta.llama3-8b-instruct-v1:0",
    #         callbacks=[MaximLangchainTracer(logger)],
    #         streaming=True,
    #     )
    #     messages = [
    #         (
    #             "system",
    #             "You are essay writer",
    #         ),
    #         ("human", "write a long essay about sachin tendulkar"),
    #     ]
    #     for chunk in model.stream(messages):
    #         # print(chunk)
    #         pass

    #     logger.flush()

    # def test_generation_chat_prompt_bedrock_llama_chat_model_tool_call(self):
    #     logger = self.maxim.logger(LoggerConfig(id=repoId))
    #     model = ChatBedrock(
    #         region="us-east-1",
    #         aws_access_key_id=awsAccessKeyId,
    #         aws_secret_access_key=awsAccessKeySecret,
    #         model_kwargs={"temperature": 0.1},
    #         provider="meta",
    #         model="meta.llama3-8b-instruct-v1:0",
    #         callbacks=[MaximLangchainTracer(logger)],
    #     )
    #     messages = [
    #         (
    #             "system",
    #             "You are a math expert",
    #         ),
    #         ("human", "add 99.9 and 11"),
    #     ]
    #     llm = model.bind_tools([addition_tool, subtraction_tool])
    #     result = llm.invoke(messages)
    #     print(f"Model response: {result}")
    #     logger.flush()

    def test_generation_chat_prompt_anthropic_llm(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = AnthropicLLM(
            api_key=anthropicApiKey,
            model_name="claude-4-sonnet",
            callbacks=[MaximLangchainTracer(logger)],
        )
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "How are you doing."),
        ]
        result = model.invoke(messages)
        print(result)
        logger.flush()

    def test_generation_chat_prompt_anthropic_sonnet_chat_model(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = ChatAnthropic(
            api_key=anthropicApiKey,
            callbacks=[MaximLangchainTracer(logger)],
            model_name="claude-4-sonnet",
            timeout=10,
            stop=None,
        )
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "How are you doing."),
        ]
        result = model.invoke(messages)
        print(f"Result = {result}")
        logger.flush()

    def test_generation_chat_prompt_anthropic_sonnet_chat_model_streaming(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = ChatAnthropic(
            api_key=anthropicApiKey,
            callbacks=[MaximLangchainTracer(logger)],
            model_name="claude-4-sonnet",
            timeout=10,
            stop=None,
            stream_usage=True,
            streaming=True,
        )
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "How are you doing."),
        ]
        for chunk in model.stream(messages):
            print(chunk)
        logger.flush()

    def test_generation_chat_prompt_openai_chat_model_with_tool_call(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = model = ChatOpenAI(
            callbacks=[MaximLangchainTracer(logger)],
            api_key=openAIKey,
            model="gpt-4o-mini",
        )
        llm = model.bind_tools([addition_tool, subtraction_tool])
        messages = [
            (
                "system",
                "You are a meth teacher",
            ),
            ("human", "What is addition of 4 and 5"),
        ]
        result = llm.invoke(messages)
        print(f"Result = {result}")
        logger.flush()

    def test_generation_chat_prompt_openai_chat_model_with_tool_call_with_streaming(
        self,
    ):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = model = ChatOpenAI(
            callbacks=[MaximLangchainTracer(logger)],
            api_key=openAIKey,
            model="gpt-4o-mini",
            streaming=True,
        )
        llm = model.bind_tools([addition_tool, subtraction_tool])
        messages = [
            (
                "system",
                "You are a meth teacher",
            ),
            ("human", "What is addition of 4 and 5"),
        ]

        for chunk in llm.stream(messages):
            print(chunk)
        logger.flush()

    def test_generation_chat_prompt_openai_chat_model_with_streaming(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = model = ChatOpenAI(
            callbacks=[MaximLangchainTracer(logger)],
            api_key=openAIKey,
            model="gpt-4o-mini",
            streaming=True,
        )
        llm = model.bind_tools([addition_tool, subtraction_tool])
        messages = [
            (
                "system",
                "You are essay writer",
            ),
            ("human", "write a long essay about sachin tendulkar"),
        ]

        for chunk in llm.stream(messages, stream_usage=True):
            # print(chunk)
            pass
        logger.flush()

    def test_generation_chat_prompt_anthropic_sonnet_chat_model_with_tool_call(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = ChatAnthropic(
            api_key=anthropicApiKey,
            callbacks=[MaximLangchainTracer(logger)],
            model_name="claude-4-sonnet",
            timeout=10,
            stop=None,
        )
        llm = model.bind_tools([addition_tool, subtraction_tool])
        messages = [
            (
                "system",
                "You are a meth teacher",
            ),
            ("human", "What is addition of 4 and 5"),
        ]
        result = llm.invoke(messages)
        print(f"Result = {result}")
        logger.flush()

    def test_generation_chat_prompt_anthropic_3_sonnet_chat_model(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = ChatAnthropic(
            api_key=anthropicApiKey,
            callbacks=[MaximLangchainTracer(logger)],
            model_name="claude-4-sonnet",
            timeout=10,
            stop=None,
        )
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "How are you doing."),
        ]
        result = model.invoke(messages)
        print(f"Result = {result}")
        logger.flush()

    def test_generation_chat_prompt_anthropic_haiku_chat_model(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = ChatAnthropic(
            api_key=anthropicApiKey,
            callbacks=[MaximLangchainTracer(logger)],
            model_name="claude-3-haiku-20240307",
            timeout=10,
            stop=None,
        )
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "How are you doing."),
        ]
        result = model.invoke(messages)
        print(f"Result = {result}")
        logger.flush()

    def test_generation_chat_prompt_azure_chat_model_old_class(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = AzureOpenAI(
            api_key=azureOpenAIKey,
            model="gpt-4o",
            azure_endpoint=azureOpenAIBaseUrl,
            callbacks=[MaximLangchainTracer(logger)],
            api_version="2024-02-01",
        )
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "How are you doing."),
        ]
        result = model.invoke(messages)
        print(result)
        logger.flush()
        sleep(5)

    def test_generation_chat_prompt_chat_model_with_span(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        trace = logger.trace(TraceConfig(id=str(uuid4()), name="test-trace"))
        span = trace.span(SpanConfig(id=str(uuid4()), name="test-span"))
        model = ChatOpenAI(callbacks=[MaximLangchainTracer(logger)], api_key=openAIKey)
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "I love programming from chat."),
        ]
        model.invoke(messages, config={"metadata": {"maxim": {"span_id": span.id}}})
        span.event(id=str(uuid4()), name="done llm call")
        span.end()
        trace.end()

    def test_generation_chat_prompt_chat_model_error(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = ChatOpenAI(
            model="gpt-4.1", callbacks=[MaximLangchainTracer(logger)], api_key=openAIKey
        )
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "I love programming from chat error."),
        ]
        with self.assertRaises(Exception):
            model.invoke(messages)

    def test_langchain_generation_with_chain(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4o",
            n=3,
            api_key=openAIKey,
            callbacks=[MaximLangchainTracer(logger)],
        )
        system_template = SystemMessagePromptTemplate.from_template(
            "You are an expert in Data Science and Machine Learning"
        )
        user_template = HumanMessagePromptTemplate.from_template("{user_prompt}")
        template = ChatPromptTemplate.from_messages([system_template, user_template])
        chain = LLMChain(llm=llm, prompt=template)
        user_prompt = "How to handle outliers in dirty datasets"
        # Run the Langchain chain with the user prompt
        result = chain.run({"user_prompt": user_prompt})
        print(result)

    def test_langchain_generation_with_azure_chain(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        llm = AzureChatOpenAI(
            api_key=azureOpenAIKey,
            model="gpt-4o",
            azure_endpoint=azureOpenAIBaseUrl,
            api_version="2024-02-01",
            callbacks=[MaximLangchainTracer(logger)],
        )
        system_template = SystemMessagePromptTemplate.from_template(
            "You are an expert in Data Science and Machine Learning"
        )
        user_template = HumanMessagePromptTemplate.from_template("{user_prompt}")
        template = ChatPromptTemplate.from_messages([system_template, user_template])
        chain = LLMChain(
            llm=llm, prompt=template, callbacks=[MaximLangchainTracer(logger)]
        )
        user_prompt = "How to handle outliers in dirty datasets"
        # Run the Langchain chain with the user prompt
        result = chain.run({"user_prompt": user_prompt})
        print(result)

    def test_langchain_generation_with_azure_multi_prompt_chain(self):
        logger = self.maxim.logger({
            "id": repoId,
        })
        llm = AzureChatOpenAI(
            api_key=azureOpenAIKey,
            model="gpt-4o",
            azure_endpoint=azureOpenAIBaseUrl,
            api_version="2024-02-01",
            callbacks=[MaximLangchainTracer(logger)],
        )

        # Define the system message template
        system_template = SystemMessagePromptTemplate.from_template(
            "You are an expert in Data Science and Machine Learning"
        )

        # Define the user message template
        user_template = HumanMessagePromptTemplate.from_template("{user_prompt}")

        # Create a prompt template for outlier handling
        outlier_template = ChatPromptTemplate.from_messages(
            [system_template, user_template]
        )

        # Create a prompt template for data cleaning
        cleaning_template = ChatPromptTemplate.from_messages(
            [
                system_template,
                HumanMessagePromptTemplate.from_template("{cleaning_prompt}"),
            ]
        )

        # Create two separate LLM chains
        outlier_chain = LLMChain(llm=llm, prompt=outlier_template)
        cleaning_chain = LLMChain(llm=llm, prompt=cleaning_template)

        # User prompts for both tasks
        user_prompt_outliers = "How to handle outliers in dirty datasets"
        user_prompt_cleaning = "What are the best practices for cleaning data?"

        # Run the Langchain chains with the user prompts
        result_outliers = outlier_chain.run({"user_prompt": user_prompt_outliers})
        result_cleaning = cleaning_chain.run({"cleaning_prompt": user_prompt_cleaning})

        # Print results from both chains
        print("Outlier Handling Result:", result_outliers)
        print("Data Cleaning Result:", result_cleaning)

    def test_langchain_tools(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4o",
            n=3,
            api_key=openAIKey,
            callbacks=[MaximLangchainTracer(logger)],
        )
        llm_with_tools = llm.bind_tools([addition_tool, subtraction_tool])
        query = "whats addition of 3 and 2"
        result = llm_with_tools.invoke(query)

    def test_langchain_tools_with_chat_openai_chain(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        prompt = PromptTemplate(
            input_variables=["first_int", "second_int"],
            template="What is {first_int} multiplied by {second_int}?",
        )
        llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4o",
            n=3,
            api_key=openAIKey,
            callbacks=[MaximLangchainTracer(logger)],
        ).bind_tools([addition_tool, subtraction_tool])
        # Create an LLM chain with the prompt
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        response = llm_chain.run(first_int=4, second_int=5)
        print(response)

    def test_custom_result_with_generation_chat_prompt(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        model = OpenAI(api_key=openAIKey)
        trace = logger.trace(TraceConfig(id=str(uuid4()), name="test-trace"))
        generation = trace.generation(
            GenerationConfig(
                id=str(uuid4()),
                name="test-generation",
                provider="openai",
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": "You are a helpful assistant that translates English to French. Translate the user sentence.",
                    }
                ],
            )
        )
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "I love programming."),
        ]
        result = model.invoke(messages)
        print(result)
        generation.result(result)
        trace.end()

    def test_simple_langchain_chain(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        llm = ChatOpenAI(temperature=0.7, model="gpt-4o", n=3, api_key=openAIKey)
        system_template = SystemMessagePromptTemplate.from_template(
            "You are an expert in Data Science and Machine Learning"
        )
        user_template = HumanMessagePromptTemplate.from_template("{user_prompt}")
        template = ChatPromptTemplate.from_messages([system_template, user_template])
        chain = template | llm
        user_prompt = "How are you"
        # Run the Langchain chain with the user prompt
        result = chain.invoke(
            {"user_prompt": user_prompt},
            config={"callbacks": [MaximLangchainTracer(logger)]},
        )
        print(result)

    def test_multi_node_langchain_chain(self):
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )
        llm = ChatOpenAI(temperature=0.7, model="gpt-4o", n=3, api_key=openAIKey)

        # Define a simple function to use in the chain
        def capitalize_text(text: str) -> str:
            return text.upper()

        # Create a RunnableFunction from the capitalize_text function
        capitalize_node = RunnableLambda(capitalize_text)

        # Create a prompt template
        prompt_template = PromptTemplate.from_template(
            "Summarize the following text: {capitalized_text}"
        )

        # Build the multi-node chain
        chain = {"capitalized_text": capitalize_node} | prompt_template | llm

        # Run the chain with input
        input_text = "This is a test of a multi-node Langchain."
        result = chain.invoke(
            input_text 
        )

        print(result)

    def test_human_message_with_image_file_attachment(self):
        """Test LangChain chat model with image file attached in human message using raw bytes.

        This test demonstrates LangChain's support for raw bytes using the 'media' content type.
        No base64 encoding is required - just read the file as bytes and pass directly.
        """
        # Get the path to the test image file
        test_files_dir = Path(__file__).parent / "files"
        image_path = test_files_dir / "png_image.png"

        # Read the file as raw bytes - no base64 encoding needed!
        file_bytes = image_path.read_bytes()

        # Create a multimodal message with text and image using raw bytes
        message_with_image = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "What do you see in this image? Describe it briefly.",
                },
                {
                    "type": "media",
                    "mime_type": "image/png",
                    "data": file_bytes,  # Raw bytes for LangChain
                },
            ]
        )

        model = ChatOpenAI(
            api_key=openAIKey,
            model="gpt-4o-mini",
        )

        result = model.invoke([message_with_image])
        print(f"Result with image attachment: {result}")

    def test_human_message_with_image_url_attachment(self):
        """Test LangChain chat model with image URL attached in human message"""
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )

        # Use a public image URL
        image_url = "https://pyxis.nymag.com/v1/imgs/7e2/b83/01a7d3094f5856a53f409a59b9d16e392e-22-transformers-fighting.jpg"

        # Create a multimodal message with text and image URL
        message_with_image = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "What do you see in this image? Describe it briefly.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ]
        )

        model = ChatOpenAI(
            callbacks=[MaximLangchainTracer(logger)],
            api_key=openAIKey,
            model="gpt-4o-mini",
        )

        result = model.invoke([message_with_image])
        print(f"Result with image URL attachment: {result}")

        logger.flush()

    def test_human_message_with_multiple_image_attachments(self):
        """Test LangChain chat model with multiple images attached in human message.

        This test demonstrates mixing raw bytes (local file) with URL-based images.
        """
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )

        # Get the path to the test image file
        test_files_dir = Path(__file__).parent / "files"
        image_path = test_files_dir / "png_image.png"

        # Read the file as raw bytes
        file_bytes = image_path.read_bytes()

        # Use a public image URL as second image
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"

        # Create a multimodal message with text and multiple images
        # Mix raw bytes (local file) with URL-based image
        message_with_images = HumanMessage(
            content=[
                {
                    "type": "media",
                    "mime_type": "image/png",
                    "data": file_bytes,  # Raw bytes for local file
                },
            ]
        )

        model = ChatOpenAI(
            callbacks=[MaximLangchainTracer(logger)],
            api_key=openAIKey,
            model="gpt-4o-mini",
        )

        result = model.invoke([message_with_images])
        print(f"Result with multiple image attachments: {result}")

        logger.flush()

    def test_human_message_with_audio_file_attachment(self):
        """Test LangChain chat model with audio file attached in human message using raw bytes.

        This test demonstrates attaching audio files using the 'media' content type.
        Supported formats for media in LangChain:
        1. URL: {"type": "image_url", "image_url": {"url": "https://..."}}
        2. Raw bytes: {"type": "media", "mime_type": "audio/wav", "data": bytes}
        3. Base64 data URL: {"type": "image_url", "image_url": {"url": "data:audio/wav;base64,..."}}
        """
        logger = self.maxim.logger(
            {
                "id": repoId,
            }
        )

        # Get the path to the test audio file
        test_files_dir = Path(__file__).parent / "files"
        audio_path = test_files_dir / "wav_audio.wav"

        # Read the file as raw bytes
        file_bytes = audio_path.read_bytes()

        # Create a multimodal message with text and audio using raw bytes
        message_with_audio = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "What do you hear in this audio? Describe it briefly.",
                },
                {
                    "type": "media",
                    "mime_type": "audio/wav",
                    "data": file_bytes,  # Raw bytes for LangChain
                },
            ]
        )

        # Use gpt-4o-audio-preview for audio input support
        model = ChatOpenAI(
            callbacks=[MaximLangchainTracer(logger)],
            api_key=openAIKey,
            model="gpt-4o-audio-preview",
        )

        result = model.invoke([message_with_audio])
        print(f"Result with audio attachment: {result}")

        logger.flush()

    def test_generation_result_callback(self):
        """Test that generation.result callback is called with correct values"""
        logger = self.maxim.logger(LoggerConfigDict(id=repoId))

        # Track callback events
        callback_events = []

        def callback(event_type: str, event_data: dict):
            callback_events.append((event_type, event_data))
            # If this is a generation.result event, add cost metadata
            if event_type == "generation.result":
                generation_id = event_data.get("generation_id")
                token_usage = event_data.get("token_usage", {})

                if generation_id and token_usage:
                    # Calculate cost: each token * 10
                    prompt_tokens = token_usage.get("prompt_tokens", 0) or 0
                    completion_tokens = token_usage.get("completion_tokens", 0) or 0
                    total_tokens = token_usage.get("total_tokens", 0) or 0

                    cost = {
                        "input": prompt_tokens * 10,
                        "output": completion_tokens * 10,
                        "total": total_tokens * 10,
                    }

                    # Add cost to the generation
                    logger.generation_add_cost(generation_id, cost)

        tracer = MaximLangchainTracer(logger, callback=callback)
        model = ChatOpenAI(
            callbacks=[tracer],
            api_key=openAIKey,
            model="gpt-3.5-turbo",
            temperature=0,
        )

        generation_name = "test-generation"
        messages = [
            ("system", "You are a helpful assistant."),
            ("human", "Say hello in one word."),
        ]

        model.invoke(
            messages,
            config={
                "metadata": {
                    "maxim": {
                        "generation_name": generation_name,
                    }
                }
            },
        )

        # Verify generation.result callback was called
        generation_result_events = [
            (event_type, event_data)
            for event_type, event_data in callback_events
            if event_type == "generation.result"
        ]

        self.assertGreater(
            len(generation_result_events),
            0,
            "generation.result callback should be called at least once",
        )

        print(f"{baseUrl}")
        print(f"Generation result events: {generation_result_events}")

        # Check the first generation.result event
        event_type, event_data = generation_result_events[0]
        self.assertEqual(event_type, "generation.result")
        self.assertIn("generation_id", event_data)
        self.assertIn("generation_name", event_data)
        self.assertIn("token_usage", event_data)

        # Verify generation_name matches
        self.assertEqual(event_data["generation_name"], generation_name)

        # Verify token_usage structure
        token_usage = event_data["token_usage"]
        self.assertIsInstance(token_usage, dict)
        # Token usage should have at least one of these keys
        self.assertTrue(
            any(
                key in token_usage
                for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
            ),
            "token_usage should contain at least one token count field",
        )

        # Verify generation_id is a string
        self.assertIsInstance(event_data["generation_id"], str)
        self.assertGreater(len(event_data["generation_id"]), 0)

        # Flush logs to ensure they're sent to Maxim
        logger.flush()

    def tearDown(self) -> None:
        self.maxim.cleanup()
        return super().tearDown()

############################################
# This file uses langchain 0.3.x
###########################################

import logging
import os
import unittest
from time import sleep
from uuid import uuid4

from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.runnable import RunnableLambda 
from langchain_anthropic import ChatAnthropic
from langchain_community.llms.openai import AzureOpenAI, OpenAI
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from maxim.logger.components.generation import GenerationConfig
from maxim.logger.components.span import SpanConfig
from maxim.logger.components.trace import TraceConfig
from maxim.logger.langchain.tracer import MaximLangchainTracer
from maxim.logger.logger import LoggerConfig
from maxim.maxim import Maxim
from dotenv import load_dotenv
load_dotenv()

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
        self.maxim = Maxim({ "base_url": baseUrl, "api_key": apiKey })
        self.logger = self.maxim.logger(LoggerConfig(id=repoId))

    def test_generation_chat_prompt(self):
        model = OpenAI(callbacks=[MaximLangchainTracer(self.logger)], api_key=openAIKey)
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
        trace_id = str(uuid4())
        trace = self.logger.trace(TraceConfig(id=trace_id, name="pre-defined-trace"))

        model = OpenAI(callbacks=[MaximLangchainTracer(self.logger)], api_key=openAIKey)
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
        model = ChatOpenAI(callbacks=[MaximLangchainTracer(self.logger)], api_key=openAIKey)
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "How are you doing."),
        ]
        model.invoke(messages)

    def test_generation_chat_prompt_azure_chat_model(self):
        model = AzureChatOpenAI(
            api_key=azureOpenAIKey,
            model="gpt-4o",
            azure_endpoint=azureOpenAIBaseUrl,
            callbacks=[MaximLangchainTracer(self.logger)],
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
        self.logger.flush()

    def test_generation_chat_prompt_azure_chat_model_with_streaming(self):
        model = AzureChatOpenAI(
            api_key=azureOpenAIKey,
            model="gpt-4o",
            azure_endpoint=azureOpenAIBaseUrl,
            callbacks=[MaximLangchainTracer(self.logger)],
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
        self.logger.flush()

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

    # def test_generation_chat_prompt_anthropic_llm(self):
    #     model = AnthropicLLM(
    #         api_key=anthropicApiKey,
    #         model_name="claude-3-5-sonnet-20240620",
    #         callbacks=[MaximLangchainTracer(self.logger)],
    #     )
    #     messages = [
    #         (
    #             "system",
    #             "You are a helpful assistant that translates English to French. Translate the user sentence.",
    #         ),
    #         ("human", "How are you doing."),
    #     ]
    #     result = model.invoke(messages)
    #     print(result)
    #     self.logger.flush()
    #
    # def test_generation_chat_prompt_anthropic_sonnet_chat_model(self):
    #     model = ChatAnthropic(
    #         api_key=anthropicApiKey,
    #         callbacks=[MaximLangchainTracer(self.logger)],
    #         model_name="claude-3-5-sonnet-20240620",
    #         timeout=10,
    #         stop=None,
    #     )
    #     messages = [
    #         (
    #             "system",
    #             "You are a helpful assistant that translates English to French. Translate the user sentence.",
    #         ),
    #         ("human", "How are you doing."),
    #     ]
    #     result = model.invoke(messages)
    #     print(f"Result = {result}")
    #     self.logger.flush()

    def test_generation_chat_prompt_anthropic_sonnet_chat_model_streaming(self):
        model = ChatAnthropic(
            api_key=anthropicApiKey,
            callbacks=[MaximLangchainTracer(self.logger)],
            model_name="claude-3-haiku-20240307",
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
        self.logger.flush()

    def test_generation_chat_prompt_openai_chat_model_with_tool_call(self):
        model = model = ChatOpenAI(
            callbacks=[MaximLangchainTracer(self.logger)],
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
        self.logger.flush()

    def test_generation_chat_prompt_openai_chat_model_with_tool_call_with_streaming(
        self,
    ):
        model = model = ChatOpenAI(
            callbacks=[MaximLangchainTracer(self.logger)],
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
        self.logger.flush()

    def test_generation_chat_prompt_openai_chat_model_with_streaming(self):
        model = model = ChatOpenAI(
            callbacks=[MaximLangchainTracer(self.logger)],
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
        self.logger.flush()

    def test_generation_chat_prompt_anthropic_sonnet_chat_model_with_tool_call(self):
        model = ChatAnthropic(
            api_key=anthropicApiKey,
            callbacks=[MaximLangchainTracer(self.logger)],
            model_name="claude-3-5-sonnet-20240620",
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
        self.logger.flush()

    def test_generation_chat_prompt_anthropic_3_sonnet_chat_model(self):
        model = ChatAnthropic(
            api_key=anthropicApiKey,
            callbacks=[MaximLangchainTracer(self.logger)],
            model_name="claude-3-sonnet-20240229",
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
        self.logger.flush()

    def test_generation_chat_prompt_anthropic_haiku_chat_model(self):
        model = ChatAnthropic(
            api_key=anthropicApiKey,
            callbacks=[MaximLangchainTracer(self.logger)],
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
        self.logger.flush()

    def test_generation_chat_prompt_azure_chat_model_old_class(self):
        model = AzureOpenAI(
            api_key=azureOpenAIKey,
            model="gpt-4o",
            azure_endpoint=azureOpenAIBaseUrl,
            callbacks=[MaximLangchainTracer(self.logger)],
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
        self.logger.flush()
        sleep(5)

    def test_generation_chat_prompt_chat_model_with_span(self):
        trace = self.logger.trace(TraceConfig(id=str(uuid4()), name="test-trace"))
        span = trace.span(SpanConfig(id=str(uuid4()), name="test-span"))
        model = ChatOpenAI(callbacks=[MaximLangchainTracer(self.logger)], api_key=openAIKey)
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
        model = ChatOpenAI(
            model="gpt-4o", callbacks=[MaximLangchainTracer(self.logger)], api_key=openAIKey
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
        llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4o",
            n=3,
            api_key=openAIKey,
            callbacks=[MaximLangchainTracer(self.logger)],
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
        llm = AzureChatOpenAI(
            api_key=azureOpenAIKey,
            model="gpt-4o",
            azure_endpoint=azureOpenAIBaseUrl,
            api_version="2024-02-01",
            callbacks=[MaximLangchainTracer(self.logger)],
        )
        system_template = SystemMessagePromptTemplate.from_template(
            "You are an expert in Data Science and Machine Learning"
        )
        user_template = HumanMessagePromptTemplate.from_template("{user_prompt}")
        template = ChatPromptTemplate.from_messages([system_template, user_template])
        chain = LLMChain(
            llm=llm, prompt=template, callbacks=[MaximLangchainTracer(self.logger)]
        )
        user_prompt = "How to handle outliers in dirty datasets"
        # Run the Langchain chain with the user prompt
        result = chain.run({"user_prompt": user_prompt})
        print(result)

    def test_langchain_generation_with_azure_multi_prompt_chain(self):
        llm = AzureChatOpenAI(
            api_key=azureOpenAIKey,
            model="gpt-4o",
            azure_endpoint=azureOpenAIBaseUrl,
            api_version="2024-02-01",
            callbacks=[MaximLangchainTracer(self.logger)],
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
        llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4o",
            n=3,
            api_key=openAIKey,
            callbacks=[MaximLangchainTracer(self.logger)],
        )
        llm_with_tools = llm.bind_tools([addition_tool, subtraction_tool])
        query = "whats addition of 3 and 2"
        result = llm_with_tools.invoke(query)

    def test_langchain_tools_with_chat_openai_chain(self):
        prompt = PromptTemplate(
            input_variables=["first_int", "second_int"],
            template="What is {first_int} multiplied by {second_int}?",
        )
        llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4o",
            n=3,
            api_key=openAIKey,
            callbacks=[MaximLangchainTracer(self.logger)],
        ).bind_tools([addition_tool, subtraction_tool])
        # Create an LLM chain with the prompt
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        response = llm_chain.run(first_int=4, second_int=5)
        print(response)

    def test_custom_result_with_generation_chat_prompt(self):
        model = OpenAI(api_key=openAIKey)
        trace = self.logger.trace(TraceConfig(id=str(uuid4()), name="test-trace"))
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
        llm = ChatOpenAI(temperature=0.7, model="gpt-4o", n=3, api_key=openAIKey)
        system_template = SystemMessagePromptTemplate.from_template(
            "You are an expert in Data Science and Machine Learning"
        )
        user_template = HumanMessagePromptTemplate.from_template("{user_prompt}")
        template = ChatPromptTemplate.from_messages([system_template, user_template])
        chain = template.pipe(llm)
        user_prompt = "How are you"
        # Run the Langchain chain with the user prompt
        result = chain.invoke(
            {"user_prompt": user_prompt},
            config={"callbacks": [MaximLangchainTracer(self.logger)]},
        )
        print(result)

    def test_multi_node_langchain_chain(self):
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

        # Build the multi-node chain using RunnableParallel
        from langchain.schema.runnable import RunnablePassthrough 
        chain = capitalize_node.pipe(prompt_template).pipe(llm)

        # Run the chain with input
        input_text = "This is a test of a multi-node Langchain."
        result = chain.invoke(
            input_text, config={"callbacks": [MaximLangchainTracer(self.logger)]}
        )

        print(result)

    def tearDown(self) -> None:
        self.maxim.cleanup()
        return super().tearDown()

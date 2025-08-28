# # ############################################
# # # This file uses langchain 0.2.x
# # ###########################################

# import json
# import logging
# import os
# import sys
# import unittest
# from os import path
# from time import sleep
# from uuid import UUID, uuid4

# import boto3
# from langchain.chains.llm import LLMChain
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain.prompts.chat import (AIMessagePromptTemplate,
#                                     ChatPromptTemplate,
#                                     HumanMessagePromptTemplate,
#                                     SystemMessagePromptTemplate)
# from langchain.schema.output_parser import StrOutputParser
# from langchain.schema.runnable import RunnableLambda, RunnableSequence
# from langchain_anthropic import AnthropicLLM, ChatAnthropic
# from langchain_aws import ChatBedrock
# from langchain_community.llms.openai import AzureOpenAI, OpenAI
# from langchain_core.tools import tool
# from langchain_openai import AzureChatOpenAI, ChatOpenAI
# from maxim.logger.components.generation import (GenerationConfig,
#                                                 GenerationError)
# from maxim.logger.components.span import SpanConfig
# from maxim.logger.components.trace import TraceConfig
# from maxim.logger.langchain.tracer import MaximLangchainTracer
# from maxim.logger.langchain.utils import (parse_langchain_llm_result,
#                                           parse_langchain_messages)
# from maxim.logger.logger import Logger, LoggerConfig
# from maxim.maxim import Config, Maxim

# with open(str(f"{os.getcwd()}/libs/maxim-py/maxim/tests/testConfig.json")) as f:
#     data = json.load(f)

# env = "dev"

# awsAccessKeyId = data["bedrockAccessKey"]
# awsAccessKeySecret = data["bedrockSecretKey"]
# azureOpenAIBaseUrl = data["azureOpenAIBaseUrl"]
# azureOpenAIKey = data["azureOpenAIKey"]
# openAIKey = data["openAIKey"]
# apiKey = data[env]["apiKey"]
# baseUrl = data[env]["baseUrl"]
# repoId = data[env]["repoId"]
# anthropicApiKey = data["anthropicApiKey"]

# # # Callback handler for langchain
# logging.basicConfig(level=logging.INFO)


# @tool
# def addition_tool(a, b):
#     """Add two integers together."""
#     return a+b


# @tool
# def subtraction_tool(a, b):
#     """Subtract two integers."""
#     return a-b


# class TestLoggingUsingLangchain(unittest.TestCase):
#     def setUp(self):
#         config = Config(apiKey=apiKey, baseUrl=baseUrl, debug=True)
#         self.maxim = Maxim(config)

#     def test_generation_chat_prompt(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = OpenAI(
#             callbacks=[MaximLangchainTracer(logger)], api_key=openAIKey)
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "I love programming."),
#         ]
#         result = model.invoke(messages)
#         print(f"result {result}")

#     def test_generation_chat_prompt_with_external_trace(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         trace_id = str(uuid4())
#         trace = logger.trace(TraceConfig(
#             id=trace_id, name="pre-defined-trace"))

#         model = OpenAI(
#             callbacks=[MaximLangchainTracer(logger)], api_key=openAIKey)
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "I love programming."),
#         ]
#         model.invoke(messages, config={
#             "metadata": {
#                 "maxim": {
#                     "trace_id": trace_id,
#                     "generation_name": "get-answer",
#                     "generation_tags": {
#                         "test": "123"
#                     }
#                 }
#             }
#         })
#         trace.event(id=str(uuid4()), name="test event")
#         trace.end()

#     def test_generation_chat_prompt_chat_model(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatOpenAI(callbacks=[
#             MaximLangchainTracer(logger)], api_key=openAIKey)
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "How are you doing."),
#         ]
#         model.invoke(messages)

#     def test_generation_chat_prompt_azure_chat_model(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = AzureChatOpenAI(api_key=azureOpenAIKey,
#                                 model="gpt-35-turbo-16k",
#                                 azure_endpoint=azureOpenAIBaseUrl,
#                                 callbacks=[MaximLangchainTracer(logger)],
#                                 api_version="2024-02-01"
#                                 )
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "How are you doing."),
#         ]
#         result = model.invoke(messages,
#                               config={"metadata": {
#                                   "maxim": {
#                                       "trace_tags": {
#                                           "first_tag": "value",
#                                           "second_tag": "value"
#                                       }
#                                   }
#                               }})
#         print(result)
#         logger.flush()

#     def test_generation_chat_prompt_azure_chat_model_with_streaming(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = AzureChatOpenAI(api_key=azureOpenAIKey,
#                                 model="gpt-35-turbo-16k",
#                                 azure_endpoint=azureOpenAIBaseUrl,
#                                 callbacks=[MaximLangchainTracer(logger)],
#                                 api_version="2024-02-01",
#                                 streaming=True,
#                                 )
#         messages = [
#             (
#                 "system",
#                 "You are essay writer",
#             ),
#             ("human", "what's up with rahul dravid?"),
#         ]
#         for chunk in model.stream(messages, config={"metadata": {
#                 "maxim": {"trace_tags": {"a": "asdasd"}}}}):
#             # print(chunk)
#             pass
#         logger.flush()

#     def test_generation_chat_prompt_bedrock_sonnet_35_chat_model(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatBedrock(
#             client=boto3.client("bedrock-runtime"),
#             region_name="us-east-1",
#             model_kwargs={"temperature": 0.1},
#             provider="anthropic",
#             model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
#             callbacks=[MaximLangchainTracer(logger)],
#         )
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "How are you doing."),
#         ]
#         result = model.invoke(messages)
#         print(f"Model response: {result}")
#         logger.flush()

#     def test_generation_chat_prompt_bedrock_sonnet_35_chat_mode_streaming(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatBedrock(
#             region_name="us-east-1",
#             client=boto3.client("bedrock-runtime"),
#             model_kwargs={"temperature": 0.1},
#             provider="anthropic",
#             model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
#             callbacks=[MaximLangchainTracer(logger)],
#             streaming=True,
#         )
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "How are you doing."),
#         ]
#         for chunk in model.stream(messages):
#             print(chunk)
#             pass
#         logger.flush()

#     def test_generation_chat_prompt_bedrock_sonnet_3_chat_mode_streaming(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatBedrock(
#             region_name="us-east-1",
#             client=boto3.client("bedrock-runtime"),
#             model_kwargs={"temperature": 0.1},
#             provider="anthropic",
#             model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#             callbacks=[MaximLangchainTracer(logger)],
#             streaming=True,
#         )
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "How are you doing."),
#         ]
#         for chunk in model.stream(messages, config={"metadata": {
#             "maxim": {
#                 "trace_tags": {"tag1": "tag value"}
#             }
#         }}):
#             # print(chunk)
#             pass
#         logger.flush()

#     def test_generation_chat_prompt_bedrock_sonnet_3_model_with_tool_call(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatBedrock(
#             region_name="us-east-1",
#             client=boto3.client("bedrock-runtime"),
#             model_kwargs={"temperature": 0.1},
#             provider="anthropic",
#             model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#             callbacks=[MaximLangchainTracer(logger)],
#         )
#         messages = [
#             (
#                 "system",
#                 "You are a math expert",
#             ),
#             ("human", "add 3 and 3"),
#         ]
#         llm = model.bind_tools([addition_tool, subtraction_tool])
#         result = llm.invoke(messages)
#         print(f"Model response: {result}")
#         logger.flush()

#     def test_generation_chat_prompt_bedrock_sonnet_35_model_with_tool_call(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatBedrock(
#             region_name="us-east-1",
#             client=boto3.client("bedrock-runtime"),
#             model_kwargs={"temperature": 0.1},
#             provider="anthropic",
#             model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
#             callbacks=[MaximLangchainTracer(logger)],
#         )
#         messages = [
#             (
#                 "system",
#                 "You are a math expert",
#             ),
#             ("human", "add 3 and 3"),
#         ]
#         llm = model.bind_tools([addition_tool, subtraction_tool])
#         result = llm.invoke(messages)
#         print(f"Model response: {result}")
#         logger.flush()

#     def test_generation_chat_prompt_bedrock_sonnet_3_chat_model(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatBedrock(
#             region_name="us-east-1",
#             client=boto3.client("bedrock-runtime"),
#             model_kwargs={"temperature": 0.1},
#             provider="anthropic",
#             model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#             callbacks=[MaximLangchainTracer(logger)],
#         )
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "How are you doing."),
#         ]
#         result = model.invoke(messages)
#         print(f"Model response: {result}")
#         logger.flush()

#     def test_generation_chat_prompt_bedrock_sonnet_3_chat_model_old_chain(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatBedrock(
#             region_name="us-east-1",
#             client=boto3.client("bedrock-runtime"),
#             model_kwargs={"temperature": 0.1},
#             provider="anthropic",
#             model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#             callbacks=[MaximLangchainTracer(logger)],
#         )
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "{question}")
#         ]
#         llmChain = LLMChain(
#             llm=model, prompt=ChatPromptTemplate.from_messages(messages))
#         question = "How are you doing?"
#         result = llmChain.invoke(input={"question": question})
#         print(f"Model response: {result}")
#         logger.flush()

#     def test_generation_chat_prompt_bedrock_sonnet_3_chat_model_without_runnable_sequence(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatBedrock(
#             client=boto3.client("bedrock-runtime"),
#             region_name="us-east-1",
#             model_kwargs={"temperature": 0.1},
#             provider="anthropic",
#             model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#             callbacks=[MaximLangchainTracer(logger)],
#         )
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "{question}")
#         ]
#         prompt = ChatPromptTemplate.from_messages(messages=messages)
#         chain = prompt | model | StrOutputParser()
#         question = "How are you doing?"
#         result = chain.invoke(input={"question": question})
#         print(f"Model response: {result}")
#         logger.flush()

#     def test_generation_chat_prompt_bedrock_sonnet_3_chat_model_with_runnable_sequence(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatBedrock(
#             region_name="us-east-1",
#             model_kwargs={"temperature": 0.1},
#             provider="anthropic",
#             client=boto3.client("bedrock-runtime"),
#             model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#             callbacks=[MaximLangchainTracer(logger)],
#         )
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "{question}")
#         ]
#         prompt = ChatPromptTemplate.from_messages(messages=messages)
#         chain = RunnableSequence(
#             prompt,
#             model
#         )
#         question = "How are you doing?"
#         result = chain.invoke(input={"question": question})
#         print(f"Model response: {result}")
#         logger.flush()

#     def test_generation_chat_prompt_bedrock_haiku_3_chat_model(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatBedrock(
#             client=boto3.client("bedrock-runtime"),
#             region_name="us-east-1",
#             model_kwargs={"temperature": 0.1},
#             provider="anthropic",
#             model_id="anthropic.claude-3-haiku-20240307-v1:0",
#             callbacks=[MaximLangchainTracer(logger)],
#         )
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "How are you doing."),
#         ]
#         result = model.invoke(messages)
#         print(f"Model response: {result}")
#         logger.flush()

#     def test_generation_chat_prompt_bedrock_llama_chat_model(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatBedrock(
#             client=boto3.client("bedrock-runtime"),
#             region_name="us-east-1",
#             model_kwargs={"temperature": 0.1},
#             provider="meta",
#             model_id="meta.llama3-8b-instruct-v1:0",
#             callbacks=[MaximLangchainTracer(logger)],
#         )
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "How are you doing."),
#         ]
#         result = model.invoke(messages)
#         print(f"Model response: {result}")
#         logger.flush()

#     def test_generation_chat_prompt_bedrock_llama_chat_model_streaming(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatBedrock(
#             client=boto3.client("bedrock-runtime"),
#             region_name="us-east-1",
#             model_kwargs={"temperature": 0.1},
#             provider="meta",
#             model_id="meta.llama3-8b-instruct-v1:0",
#             callbacks=[MaximLangchainTracer(logger)],
#             streaming=True
#         )
#         messages = [
#             (
#                 "system",
#                 "You are essay writer",
#             ),
#             ("human", "write a long essay about sachin tendulkar"),
#         ]
#         for chunk in model.stream(messages):
#             # print(chunk)
#             pass

#         logger.flush()

#     def test_generation_chat_prompt_bedrock_llama_chat_model_tool_call(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatBedrock(
#             client=boto3.client("bedrock-runtime"),
#             region_name="us-east-1",
#             model_kwargs={"temperature": 0.1},
#             provider="meta",
#             model_id="meta.llama3-8b-instruct-v1:0",
#             callbacks=[MaximLangchainTracer(logger)],
#         )
#         messages = [
#             (
#                 "system",
#                 "You are a math expert",
#             ),
#             ("human", "add 99.9 and 11"),
#         ]
#         llm = model.bind_tools([addition_tool, subtraction_tool])
#         result = llm.invoke(messages)
#         print(f"Model response: {result}")
#         logger.flush()

#     def test_generation_chat_prompt_anthropic_llm(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = AnthropicLLM(anthropic_api_key=anthropicApiKey,
#                              model_name="claude-3-5-sonnet-20240620",
#                              callbacks=[MaximLangchainTracer(logger)],

#                              )
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "How are you doing."),
#         ]
#         result = model.invoke(messages)
#         print(result)
#         logger.flush()

#     def test_generation_chat_prompt_anthropic_sonnet_chat_model(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatAnthropic(api_key=anthropicApiKey, callbacks=[MaximLangchainTracer(
#             logger)], model_name="claude-3-5-sonnet-20240620", timeout=10)
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "How are you doing."),
#         ]
#         result = model.invoke(messages)
#         print(f"Result = {result}")
#         logger.flush()

#     def test_generation_chat_prompt_anthropic_sonnet_chat_model_streaming(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatAnthropic(api_key=anthropicApiKey, callbacks=[MaximLangchainTracer(
#             logger)], model_name="claude-3-5-sonnet-20240620", timeout=10,  streaming=True)
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "How are you doing."),
#         ]
#         for chunk in model.stream(messages):
#             print(chunk)
#         logger.flush()

#     def test_generation_chat_prompt_openai_chat_model_with_tool_call(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = model = ChatOpenAI(callbacks=[
#             MaximLangchainTracer(logger)], api_key=openAIKey, model="gpt-4o-mini")
#         llm = model.bind_tools([addition_tool, subtraction_tool])
#         messages = [
#             (
#                 "system",
#                 "You are a meth teacher",
#             ),
#             ("human", "What is addition of 4 and 5"),
#         ]
#         result = llm.invoke(messages)
#         print(f"Result = {result}")
#         logger.flush()

#     def test_generation_chat_prompt_openai_chat_model_with_tool_call_with_streaming(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = model = ChatOpenAI(callbacks=[
#             MaximLangchainTracer(logger)], api_key=openAIKey, model="gpt-4o-mini", streaming=True)
#         llm = model.bind_tools([addition_tool, subtraction_tool])
#         messages = [
#             (
#                 "system",
#                 "You are a meth teacher",
#             ),
#             ("human", "What is addition of 4 and 5"),
#         ]

#         for chunk in llm.stream(messages):
#             print(chunk)
#         logger.flush()

#     def test_generation_chat_prompt_openai_chat_model_with_streaming(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = model = ChatOpenAI(callbacks=[
#             MaximLangchainTracer(logger)], api_key=openAIKey, model="gpt-4o-mini", streaming=True)
#         llm = model.bind_tools([addition_tool, subtraction_tool])
#         messages = [
#             (
#                 "system",
#                 "You are essay writer",
#             ),
#             ("human", "write a long essay about sachin tendulkar"),
#         ]

#         for chunk in llm.stream(messages, stream_usage=True):
#             # print(chunk)
#             pass
#         logger.flush()

#     def test_generation_chat_prompt_anthropic_sonnet_chat_model_with_tool_call(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatAnthropic(api_key=anthropicApiKey, callbacks=[MaximLangchainTracer(
#             logger)], model_name="claude-3-5-sonnet-20240620", timeout=10)
#         llm = model.bind_tools([addition_tool, subtraction_tool])
#         messages = [
#             (
#                 "system",
#                 "You are a meth teacher",
#             ),
#             ("human", "What is addition of 4 and 5"),
#         ]
#         result = llm.invoke(messages)
#         print(f"Result = {result}")
#         logger.flush()

#     def test_generation_chat_prompt_anthropic_3_sonnet_chat_model(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatAnthropic(api_key=anthropicApiKey, callbacks=[MaximLangchainTracer(
#             logger)], model_name="claude-3-sonnet-20240229", timeout=10)
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "How are you doing."),
#         ]
#         result = model.invoke(messages)
#         print(f"Result = {result}")
#         logger.flush()

#     def test_generation_chat_prompt_anthropic_haiku_chat_model(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatAnthropic(api_key=anthropicApiKey, callbacks=[MaximLangchainTracer(
#             logger)], model_name="claude-3-haiku-20240307", timeout=10)
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "How are you doing."),
#         ]
#         result = model.invoke(messages)
#         print(f"Result = {result}")
#         logger.flush()

#     def test_generation_chat_prompt_azure_chat_model_old_class(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = AzureOpenAI(api_key=azureOpenAIKey,
#                             model="gpt-35-turbo-16k",
#                             azure_endpoint=azureOpenAIBaseUrl,
#                             callbacks=[MaximLangchainTracer(logger)],
#                             api_version="2024-02-01"
#                             )
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "How are you doing."),
#         ]
#         result = model.invoke(messages)
#         print(result)
#         logger.flush()
#         sleep(5)

#     def test_generation_chat_prompt_chat_model_with_span(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         trace = logger.trace(TraceConfig(id=str(uuid4()), name="test-trace"))
#         span = trace.span(SpanConfig(id=str(uuid4()), name="test-span"))
#         model = ChatOpenAI(callbacks=[
#             MaximLangchainTracer(logger)], api_key=openAIKey)
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "I love programming from chat."),
#         ]
#         model.invoke(messages, config={
#             "metadata": {
#                 "maxim": {
#                     "span_id": span.id
#                 }
#             }
#         })
#         span.event(id=str(uuid4()), name="done llm call")
#         span.end()
#         trace.end()

#     def test_generation_chat_prompt_chat_model_error(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = ChatOpenAI(model="gpt2", callbacks=[
#             MaximLangchainTracer(logger)], api_key=openAIKey)
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "I love programming from chat error."),
#         ]
#         with self.assertRaises(Exception):
#             model.invoke(messages)

#     def test_langchain_generation_with_chain(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         llm = ChatOpenAI(temperature=0.7, model="gpt-4o",
#                          n=3, api_key=openAIKey, callbacks=[MaximLangchainTracer(logger)])
#         system_template = SystemMessagePromptTemplate.from_template(
#             "You are an expert in Data Science and Machine Learning")
#         user_template = HumanMessagePromptTemplate.from_template(
#             "{user_prompt}")
#         template = ChatPromptTemplate.from_messages(
#             [system_template, user_template])
#         chain = LLMChain(llm=llm, prompt=template)
#         user_prompt = "How to handle outliers in dirty datasets"
#         # Run the Langchain chain with the user prompt
#         result = chain.run({"user_prompt": user_prompt})
#         print(result)

#     def test_langchain_generation_with_azure_chain(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         llm = AzureChatOpenAI(api_key=azureOpenAIKey,
#                               model="gpt-35-turbo-16k",
#                               azure_endpoint=azureOpenAIBaseUrl,
#                               api_version="2024-02-01",
#                               callbacks=[
#                                   MaximLangchainTracer(logger)]
#                               )
#         system_template = SystemMessagePromptTemplate.from_template(
#             "You are an expert in Data Science and Machine Learning")
#         user_template = HumanMessagePromptTemplate.from_template(
#             "{user_prompt}")
#         template = ChatPromptTemplate.from_messages(
#             [system_template, user_template])
#         chain = LLMChain(llm=llm, prompt=template, callbacks=[
#                          MaximLangchainTracer(logger)])
#         user_prompt = "How to handle outliers in dirty datasets"
#         # Run the Langchain chain with the user prompt
#         result = chain.run({"user_prompt": user_prompt})
#         print(result)

#     def test_langchain_generation_with_azure_multi_prompt_chain(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         llm = AzureChatOpenAI(api_key=azureOpenAIKey,
#                               model="gpt-35-turbo-16k",
#                               azure_endpoint=azureOpenAIBaseUrl,
#                               api_version="2024-02-01",
#                               callbacks=[
#                                   MaximLangchainTracer(logger)]
#                               )

#         # Define the system message template
#         system_template = SystemMessagePromptTemplate.from_template(
#             "You are an expert in Data Science and Machine Learning")

#         # Define the user message template
#         user_template = HumanMessagePromptTemplate.from_template(
#             "{user_prompt}")

#         # Create a prompt template for outlier handling
#         outlier_template = ChatPromptTemplate.from_messages(
#             [system_template, user_template])

#         # Create a prompt template for data cleaning
#         cleaning_template = ChatPromptTemplate.from_messages(
#             [system_template, HumanMessagePromptTemplate.from_template("{cleaning_prompt}")])

#         # Create two separate LLM chains
#         outlier_chain = LLMChain(llm=llm, prompt=outlier_template)
#         cleaning_chain = LLMChain(llm=llm, prompt=cleaning_template)

#         # User prompts for both tasks
#         user_prompt_outliers = "How to handle outliers in dirty datasets"
#         user_prompt_cleaning = "What are the best practices for cleaning data?"

#         # Run the Langchain chains with the user prompts
#         result_outliers = outlier_chain.run(
#             {"user_prompt": user_prompt_outliers})
#         result_cleaning = cleaning_chain.run(
#             {"cleaning_prompt": user_prompt_cleaning})

#         # Print results from both chains
#         print("Outlier Handling Result:", result_outliers)
#         print("Data Cleaning Result:", result_cleaning)

#     def test_langchain_tools(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         llm = ChatOpenAI(temperature=0.7, model="gpt-4o", n=3,
#                          api_key=openAIKey, callbacks=[MaximLangchainTracer(logger)])
#         llm_with_tools = llm.bind_tools([addition_tool, subtraction_tool])
#         query = "whats addition of 3 and 2"
#         result = llm_with_tools.invoke(query)

#     def test_langchain_tools_with_chat_openai_chain(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         prompt = PromptTemplate(
#             input_variables=["first_int", "second_int"],
#             template="What is {first_int} multiplied by {second_int}?"
#         )
#         llm = ChatOpenAI(temperature=0.7, model="gpt-4o", n=3,
#                          api_key=openAIKey, callbacks=[MaximLangchainTracer(logger)]).bind_tools([addition_tool, subtraction_tool])
#         # Create an LLM chain with the prompt
#         llm_chain = LLMChain(llm=llm, prompt=prompt)
#         response = llm_chain.run(first_int=4, second_int=5)
#         print(response)

#     def test_custom_result_with_generation_chat_prompt(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         model = OpenAI(api_key=openAIKey)
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         trace = logger.trace(TraceConfig(id=str(uuid4()), name="test-trace"))
#         generation = trace.generation(GenerationConfig(id=str(uuid4()), name="test-generation", provider="openai", model="gpt-4o", messages=[
#             {"role": "user", "content": "You are a helpful assistant that translates English to French. Translate the user sentence."}]))
#         messages = [
#             (
#                 "system",
#                 "You are a helpful assistant that translates English to French. Translate the user sentence.",
#             ),
#             ("human", "I love programming."),
#         ]
#         result = model.invoke(messages)
#         print(result)
#         generation.result(result)
#         trace.end()

#     def test_simple_langchain_chain(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         llm = ChatOpenAI(temperature=0.7, model="gpt-4o",
#                          n=3, api_key=openAIKey, callbacks=[MaximLangchainTracer(logger)])
#         system_template = SystemMessagePromptTemplate.from_template(
#             "You are an expert in Data Science and Machine Learning"
#         )
#         user_template = HumanMessagePromptTemplate.from_template(
#             "{user_prompt}"
#         )
#         template = ChatPromptTemplate.from_messages(
#             [system_template, user_template]
#         )
#         chain = template | llm
#         user_prompt = "How are you"
#         # Run the Langchain chain with the user prompt
#         result = chain.invoke({"user_prompt": user_prompt}, config={
#             "metadata": {
#                 "maxim": {
#                     "trace_tags": {
#                         "first_tag": "value",
#                         "second_tag": "value"
#                     }
#                 }
#             }
#         })
#         print(result)

#     def test_multi_node_langchain_chain(self):
#         logger = self.maxim.logger(LoggerConfig(id=repoId))
#         llm = ChatOpenAI(temperature=0.7, model="gpt-4o",
#                          n=3, api_key=openAIKey)

#         # Define a simple function to use in the chain
#         def capitalize_text(text: str) -> str:
#             return text.upper()

#         # Create a RunnableFunction from the capitalize_text function
#         capitalize_node = RunnableLambda(capitalize_text)

#         # Create a prompt template
#         prompt_template = PromptTemplate.from_template(
#             "Summarize the following text: {capitalized_text}")

#         # Build the multi-node chain
#         chain = (
#             {"capitalized_text": capitalize_node}
#             | prompt_template
#             | llm
#         )

#         # Run the chain with input
#         input_text = "This is a test of a multi-node Langchain."
#         result = chain.invoke(
#             input_text,
#             config={"callbacks": [MaximLangchainTracer(logger)]}
#         )

#         print(result)

#     def tearDown(self) -> None:
#         self.maxim.cleanup()
#         return super().tearDown()

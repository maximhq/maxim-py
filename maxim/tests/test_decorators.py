import concurrent.futures
import json
import logging
import os
import time
import unittest
from uuid import uuid4

from flask import Flask, request
from langchain.chat_models.openai import ChatOpenAI

from .. import Config, Maxim
from ..decorators import current_retrieval, current_trace, retrieval, span, trace
from ..decorators.langchain import langchain_callback, langchain_llm_call
from ..logger import LoggerConfig

with open(str(f"{os.getcwd()}/libs/maxim-py/maxim/tests/testConfig.json")) as f:
    data = json.load(f)

logging.basicConfig(level=logging.INFO)
env = "beta"

openAIKey = data["openAIKey"]
apiKey = data[env]["apiKey"]
baseUrl = data[env]["baseUrl"]
repoId = data[env]["repoId"]
model = ChatOpenAI(api_key=openAIKey)


class TestDecoratorForOpenAI(unittest.TestCase):

    def setUp(self) -> None:
        print("setting up")

    def test_openai_chat_one(self):
        print("here")

        @span("secondTest")
        def secondTest():
            print("inside second test")

        def thirdTest():
            print("inside third test which is not traced")

        @trace(api_key="123")
        def testing(test):
            print("inside testing")
            secondTest()
            thirdTest()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(testing, "testing2") for _ in range(2)]
            concurrent.futures.wait(futures)


class TestDecoratorsForFlask(unittest.TestCase):

    def setUp(self) -> None:
        config = Config(api_key=apiKey, base_url=baseUrl, debug=True)
        self.maxim = Maxim(config)
        config = LoggerConfig(id=repoId)
        self.logger = self.maxim.logger(config)

    def test_trace(self):

        def sensitive_method():
            print("inside sensitive method")

        @span(name="nested_span")
        def span_2():
            print("inside span")
            sensitive_method()

        @span(name="test_span11111")
        def test_span():
            print("inside span")
            sensitive_method()
            span_2()

        @trace(logger=self.logger)
        def testing(test):
            print("inside testing")
            current_trace().set_input("test_input")
            current_trace().set_output("test_output")
            test_span()

        testing("123")

    def test_flask_api(self):
        app = Flask(__name__)

        @retrieval(
            name="knowledge_query_retrieval", evaluators=["Ragas Context Relevancy"]
        )
        def knowledge_query_one(query):
            current_retrieval().input(query)
            current_retrieval().with_variables(
                ["Ragas Context Relevancy"], {"input": query}
            )
            return [
                {"test": 123, "doc": "this is documentation"},
                {"test": 1234, "doc": "this is documentation second"},
            ]

        @langchain_llm_call(name="agent_call")
        def agent_call(query):
            context = knowledge_query_one("test")
            print("inside agent call")
            messages = [
                (
                    "system",
                    "You are a helpful assistant that translates English to French. Translate the user sentence.",
                ),
                ("human", query),
            ]
            return model.invoke(
                messages, config={"callbacks": [langchain_callback()]}
            ).content

        @app.post("/chat")
        @trace(logger=self.logger, name="chat_v1", sessionId="1234")
        def handle():
            current_trace().add_tag("test", "yes")
            current_trace().set_input(request.json["query"])
            result = agent_call(request.json["query"])
            current_trace().set_output(result)
            return result

        with app.test_client() as client:
            response = client.post(
                "/chat",
                headers={"reqId": str(uuid4())},
                json={"query": "second in session"},
            )
            self.assertEqual(response.status_code, 200)

    def tearDown(self) -> None:
        self.maxim.cleanup()
        return super().tearDown()

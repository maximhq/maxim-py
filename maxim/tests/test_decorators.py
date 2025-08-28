import concurrent.futures
import logging
import os
import unittest
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request
from langchain_openai import ChatOpenAI

from maxim import Maxim
from maxim.decorators import current_retrieval, current_trace, retrieval, span, trace
from maxim.decorators.langchain import langchain_callback, langchain_llm_call
from maxim.tests.mock_writer import inject_mock_writer

# Note: This file had a hardcoded path that may not work in all environments
# Let's use environment variables instead
logging.basicConfig(level=logging.INFO)

openAIKey = os.getenv("OPENAI_API_KEY")
apiKey = os.getenv("MAXIM_API_KEY")
baseUrl = os.getenv("MAXIM_BASE_URL")
repoId = os.getenv("MAXIM_LOG_REPO_ID")
model = ChatOpenAI(api_key=openAIKey) if openAIKey else None


class TestDecoratorForOpenAI(unittest.TestCase):

    def setUp(self) -> None:
        print("setting up")
        # This is a hack to ensure that the Maxim instance is not cached
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim({"base_url": baseUrl}).logger()
        self.mock_writer = inject_mock_writer(self.logger)

    def test_openai_chat_one(self):
        print("here")

        @span(name="secondTest")
        def secondTest():
            print("inside second test")

        def thirdTest():
            print("inside third test which is not traced")

        @trace(name="second trace", logger=self.logger)
        def testing(test):
            print("inside testing")
            secondTest()
            thirdTest()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(testing, "testing2") for _ in range(2)]
            concurrent.futures.wait(futures)

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Since we're running 2 concurrent traces, we expect 2 trace creates
        self.mock_writer.assert_entity_action_count("trace", "create", 2)

        # Each trace should have 2 spans (secondTest and the main trace)
        # So we expect 4 spans total (2 per trace)
        self.mock_writer.assert_entity_action_count("span", "create", 4)

        # Assert that we have 2 trace end logs
        self.mock_writer.assert_entity_action_count("trace", "end", 2)

    def tearDown(self) -> None:
        # Print final summary for debugging
        self.mock_writer.print_logs_summary()

        # Cleanup the mock writer
        self.mock_writer.cleanup()


class TestDecoratorsForFlask(unittest.TestCase):

    def setUp(self) -> None:
        # This is a hack to ensure that the Maxim instance is not cached
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.maxim = Maxim({"base_url": baseUrl})
        self.logger = self.maxim.logger()
        self.mock_writer = inject_mock_writer(self.logger)

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
            trace = current_trace()
            if trace is None:
                raise ValueError("current_trace is None")
            trace.set_input("test_input")
            trace.set_output("test_output")
            test_span()

        testing("123")

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Assert that we have exactly 1 trace create log
        self.mock_writer.assert_entity_action_count("trace", "create", 1)

        # We expect 2 spans: test_span11111 and nested_span
        self.mock_writer.assert_entity_action_count("span", "create", 2)

        # Assert that we have exactly 1 trace end log
        self.mock_writer.assert_entity_action_count("trace", "end", 1)

    def test_flask_api(self):
        if not model:
            self.skipTest("OpenAI API key not available")

        app = Flask(__name__)

        @retrieval(
            name="knowledge_query_retrieval", evaluators=["Ragas Context Relevancy"]
        )
        def knowledge_query_one(query):
            retrieval = current_retrieval()
            if retrieval is None:
                raise ValueError("current_retrieval is None")
            retrieval.input(query)
            retrieval.evaluate().with_variables(
                variables={"input": query}, for_evaluators=["Ragas Context Relevancy"]
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
            if model is None:
                return "OpenAI model not available"
            return model.invoke(
                messages, config={"callbacks": [langchain_callback()]}
            ).content

        @app.post("/chat")
        @trace(logger=self.logger, name="chat_v1", sessionId="1234")
        def handle():
            trace = current_trace()
            if trace is None:
                raise ValueError("current_trace is None")
            trace.add_tag("test", "yes")
            trace.set_input(request.json["query"])  # type: ignore
            result = agent_call(request.json["query"])  # type: ignore
            trace.set_output(result)
            return result

        with app.test_client() as client:
            response = client.post(
                "/chat",
                headers={"reqId": str(uuid4())},
                json={"query": "second in session"},
            )
            self.assertEqual(response.status_code, 200)

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # This test involves multiple decorators and langchain calls
        # We expect at least one trace to be created
        all_logs = self.mock_writer.get_all_logs()
        self.assertGreater(len(all_logs), 0, "Expected at least one log to be captured")

    def tearDown(self) -> None:
        # Print final summary for debugging
        self.mock_writer.print_logs_summary()

        # Cleanup the mock writer
        self.mock_writer.cleanup()
        self.maxim.cleanup()
        return super().tearDown()

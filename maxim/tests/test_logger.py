import json
import logging
import os
import time
import unittest
from dotenv import load_dotenv
from uuid import uuid4

from maxim import Config, Maxim
from maxim.logger import (
    Feedback,
    FileAttachment,
    GenerationConfig,
    GenerationError,
    LoggerConfig,
    RetrievalConfig,
    SessionConfig,
    SpanConfig,
    TraceConfig,
)


load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)


# Set global log level to debug
logging.getLogger().setLevel(logging.DEBUG)


class TestLoggerInitialization(unittest.TestCase):
    def setUp(self):
        self.maxim = Maxim()

    def test_initialize_logger_if_log_repository_exists(self):
        logger = self.maxim.logger()
        self.assertIsNotNone(logger)

    def test_should_throw_error_if_log_repository_does_not_exist(self):
        with self.assertRaises(Exception) as context:
            self.maxim.logger()
        self.assertTrue("Log repository not found" in str(context.exception))


class TestLogging(unittest.TestCase):
    def setUp(self):
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.maxim = Maxim()

    def tearDown(self) -> None:
        self.maxim.cleanup()
        return super().tearDown()

    def test_trace_with_input_output(self):
        logger = self.maxim.logger(
            {
                "auto_flush": False,
            }
        )
        trace = logger.trace(
            {
                "id": str(uuid4()),
            }
        )
        trace.set_input("testinput")
        trace.set_output("my birthday is today")
        trace.end()

    def test_should_be_able_to_create_a_trace_and_update(self):
        logger = self.maxim.logger()
        trace_config = TraceConfig(id=str(uuid4()))
        trace = logger.trace(trace_config)
        trace.set_input("testinput")
        trace.set_output("output")
        jsonObj = """
        {"fields": {"chunkId": "0", "toChunkId": "2", "mediaId": "2410130811461094762", "fromChunkId": "0"}}
        """
        trace.add_tag("test", json.dumps(json.loads(jsonObj)))
        trace.set_output("test output")
        trace.end()
        logger.cleanup()
        self.maxim.cleanup()


class TestCreatingSession(unittest.TestCase):
    def setUp(self):
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.maxim = Maxim()

    def tearDown(self) -> None:
        self.maxim.cleanup()
        return super().tearDown()

    def test_send_retrieval(self):
        logger = self.maxim.logger()
        trace_id = str(uuid4())
        trace = logger.trace({"id": trace_id, "name": "test trace"})
        trace.set_input("test retrieval input")
        trace.set_output("test output")
        trace.end()
        retrieval_id = str(uuid4())
        logger.trace_add_retrieval(
            trace_id, {"id": retrieval_id, "name": "Test Retrieval"}
        )
        logger.retrieval_input(retrieval_id, "asdasdas")
        logger.retrieval_output(
            retrieval_id, ["aasdasd", "Asadasdasdkjhajksbdjkasbdjkasbdjkaß"]
        )
        logger.retrieval_end(retrieval_id)

    def test_trace_generation(self):
        logger = self.maxim.logger()
        trace = logger.trace({"id": str(uuid4()), "name": "test trace"})
        trace.set_input("test input")
        trace.set_output("test output")
        span = trace.span(
            {
                "id": str(uuid4()),
                "name": "Test Span",
            }
        )
        span.add_tag("test", "test-span")
        trace.add_tag("userId", "123")
        trace.end()
        gen_id = str(uuid4())
        gen = trace.generation(
            {
                "id": gen_id,
                "provider": "openai",
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "Hello, how can I help you today?"}
                ],
            }
        )
        time.sleep(2)
        gen.result(
            {
                "id": "c9395a2d-8fbf-4e96-8ae9-be4820348f46",
                "object": "text_completion",
                "created": 1720359641,
                "provider": "openai",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "text": '{"Intent": "General Talk"}',
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "completion_tokens": 711,
                    "prompt_tokens": 6531,
                    "total_tokens": 66011,
                },
            }
        )
        gen.end()

    def test_trace_create_spans(self):
        logger = self.maxim.logger()
        trace_id = str(uuid4())
        trace = logger.trace({"id": trace_id, "name": "test trace"})
        trace.end()
        span1_id = str(uuid4())
        span1 = logger.trace_add_span(trace_id, {"id": span1_id, "name": "Test Span"})
        span1.add_tag("test", "test-span")
        span1.event(str(uuid4()), "test-event")
        gen_id = str(uuid4())
        gen = span1.generation(
            {
                "id": gen_id,
                "provider": "openai",
                "model": "gpt-3.5-turbo-16k",
                "messages": [
                    {"role": "user", "content": "Hello, how can I help you today?"}
                ],
            }
        )
        time.sleep(2)
        gen.result(
            {
                "id": "c9395a2d-8fbf-4e96-8ae9-be4820348f46",
                "object": "text_completion",
                "created": 1720359641,
                "provider": "openai",
                "model": "gpt-35-turbo",
                "choices": [
                    {
                        "index": 0,
                        "text": '{"Intent": "General Talk"}',
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "completion_tokens": 7,
                    "prompt_tokens": 653,
                    "total_tokens": 660,
                },
            }
        )
        gen.end()
        span1.end()
        return

    def test_trace_create_spans_with_error(self):
        logger = self.maxim.logger()
        trace_id = str(uuid4())
        trace = logger.trace({"id": trace_id, "name": "test trace"})
        trace.end()
        span1_id = str(uuid4())
        span1 = logger.trace_add_span(trace_id, {"id": span1_id, "name": "Test Span"})
        span1.add_tag("test", "test-span")
        span1.event(str(uuid4()), "test-event")
        span1.add_error({"message": "Testing error", "type": "testing-one"})
        gen_id = str(uuid4())
        gen = span1.generation(
            {
                "id": gen_id,
                "provider": "openai",
                "model": "gpt-3.5-turbo-16k",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, how can I help you today? And explain this in an elaborate manner.",
                    }
                ],
            }
        )
        time.sleep(2)
        gen.result(
            {
                "id": "c9395a2d-8fbf-4e96-8ae9-be4820348f46",
                "object": "text_completion",
                "created": 1720359641,
                "provider": "openai",
                "model": "gpt-35-turbo",
                "choices": [
                    {
                        "index": 0,
                        "text": '{"Intent": "General Talk", "Response": "I am here to assist you with any questions or tasks you may have. Please feel free to ask me anything, and I will do my best to provide helpful and accurate information."}',
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "completion_tokens": 7,
                    "prompt_tokens": 653,
                    "total_tokens": 660,
                },
            }
        )
        gen.end()
        span1.end()
        return

    def test_session_changes(self):
        logger = self.maxim.logger()
        session_id = str(uuid4())
        session = logger.session(SessionConfig(id=session_id, name="test session"))
        trace_id = str(uuid4())
        trace = logger.session_trace(
            session.id, TraceConfig(id=trace_id, tags={"testing again": "123333"})
        )
        time.sleep(2)
        trace.end()
        logger.trace_add_tag(trace.id, "test", "yes")
        logger.trace_event(trace.id, str(uuid4()), "test event")
        time.sleep(40)
        session.add_tag("test", "test tag should appear")
        session.end()

    def test_unended_session(self):
        logger = self.maxim.logger()
        session_id = str(uuid4())
        session = logger.session(SessionConfig(id=session_id, name="test session"))
        time.sleep(100)
        session.add_tag("test", "test tag should appear")
        time.sleep(100)

    def test_upload_file_attachment(self):
        logger = self.maxim.logger()
        trace = logger.trace({"id": str(uuid4())})
        trace.set_input("test input")
        trace.add_attachment(FileAttachment(path="files/text_file.txt"))
        trace.set_output("test output")
        trace.end()
        logger.flush()

    def test_adding_logs_out_of_order(self):
        logger = self.maxim.logger()
        session_id = str(uuid4())
        session = logger.session(SessionConfig(id=session_id))
        trace_id = str(uuid4())
        trace = logger.session_trace(session.id, TraceConfig(id=trace_id))
        time.sleep(2)
        trace.end()
        logger.trace_add_tag(trace.id, "test", "yes")
        logger.trace_event(trace.id, str(uuid4()), "test event")
        time.sleep(40)
        generation_id = str(uuid4())
        logger.trace_generation(
            trace.id,
            GenerationConfig(
                id=generation_id,
                name="gen1",
                provider="openai",
                model="gpt-3.5-turbo-16k",
                model_parameters={"temperature": 3},
                messages=[
                    {
                        "role": "user",
                        "content": "Hello, how can I help you today ttttt?",
                    }
                ],
            ),
        )
        time.sleep(30)
        logger.generation_result(
            generation_id,
            {
                "id": "10145d10-b2d0-42f6-b69a-9a8311f312b6",
                "object": "text_completion",
                "created": 1720353381,
                "model": "gpt-35-turbo",
                "choices": [
                    {
                        "index": 0,
                        "text": '{"title": "Sending a Greeting in PowerShell", "answer": "To send a greeting in PowerShell, you can create a cmdlet that accepts a name parameter and writes out a greeting to the user. Here\'s an example of how you can do it:\\n\\n```powershell\\nusing System.Management.Automation;\\n\\nnamespace SendGreeting\\n{\\n    [Cmdlet(VerbsCommunications.Send, \\"Greeting\\")]\\n    public class SendGreetingCommand : Cmdlet\\n    {\\n        [Parameter(Mandatory = true)]\\n        public string Name { get; set; }\\n\\n        protected override void ProcessRecord()\\n        {\\n            WriteObject(\\"Hello \\" + Name + \\"!\\");\\n        }\\n    }\\n}\\n```\\n\\nYou can then use this cmdlet by calling `Send-Greeting -Name suresh` to send a greeting with the name \'suresh\'. The cmdlet will write out \'Hello suresh!\' as the output.", "source_uuids_scores": [{"uuid": "c3491cef-0485-3a09-b0cd-41fdf78b160c", "score": 1}] }',
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "completion_tokens": 247,
                    "prompt_tokens": 1473,
                    "total_tokens": 1720,
                },
            },
        )
        time.sleep(20)
        span1_id = str(uuid4())
        logger.trace_span(trace.id, SpanConfig(id=span1_id, name="Test Span"))
        generation_2_id = str(uuid4())
        logger.span_generation(
            span1_id,
            GenerationConfig(
                id=generation_2_id,
                name="gen2",
                provider="openai",
                model="gpt-3.5-turbo-16k",
                model_parameters={"temperature": 3},
                messages=[
                    {"role": "user", "content": "Hello, how can I help you today?"}
                ],
            ),
        )
        time.sleep(4)
        span = logger.trace_span(
            trace.id, SpanConfig(id=str(uuid4()), name="Test Span 2")
        )
        logger.generation_result(
            generation_2_id,
            {
                "id": "c9395a2d-8fbf-4e96-8ae9-be4820348f46",
                "object": "text_completion",
                "created": 1720359641,
                "model": "gpt-35-turbo",
                "choices": [
                    {
                        "index": 0,
                        "text": '{"Intent": "General Talk"}',
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "completion_tokens": 7,
                    "prompt_tokens": 653,
                    "total_tokens": 660,
                },
            },
        )
        time.sleep(10)
        logger.span_add_tag(span1_id, "test", "test-span")
        logger.span_event(span1_id, str(uuid4()), "test-event")
        retrieval_id = str(uuid4())
        logger.span_retrieval(
            span1_id, RetrievalConfig(id=retrieval_id, name="Test Retrieval")
        )
        logger.retrieval_input(retrieval_id, "asdasdas")
        logger.retrieval_output(retrieval_id, ["asdasd", "asdasdasd"])
        logger.retrieval_end(retrieval_id)
        time.sleep(2)
        logger.span_end(span1_id)

    def test_should_be_able_to_create_a_session_and_trace_using_logger(self):
        logger = self.maxim.logger()
        session_id = str(uuid4())
        session = logger.session(SessionConfig(id=session_id))
        trace_id = str(uuid4())
        trace = logger.session_trace(session.id, TraceConfig(id=trace_id))
        self.assertIsNotNone(trace)
        self.assertEqual(trace.id, trace_id)
        logger.trace_add_tag(trace.id, "test", "yes")
        logger.trace_event(trace.id, str(uuid4()), "test event")
        generation_id = str(uuid4())
        logger.trace_add_generation(
            trace.id,
            GenerationConfig(
                id=generation_id,
                name="gen1",
                provider="openai",
                model="gpt-3.5-turbo-16k",
                model_parameters={"temperature": 3},
                messages=[
                    {
                        "role": "user",
                        "content": "Hello, how can I help you today ttttt?",
                    }
                ],
            ),
        )
        time.sleep(2)
        logger.generation_result(
            generation_id,
            {
                "id": "10145d10-b2d0-42f6-b69a-9a8311f312b6",
                "object": "text_completion",
                "created": 1720353381,
                "model": "gpt-35-turbo",
                "choices": [
                    {
                        "index": 0,
                        "text": '{"title": "Sending a Greeting in PowerShell", "answer": "To send a greeting in PowerShell, you can create a cmdlet that accepts a name parameter and writes out a greeting to the user. Here\'s an example of how you can do it:\\n\\n```powershell\\nusing System.Management.Automation;\\n\\nnamespace SendGreeting\\n{\\n    [Cmdlet(VerbsCommunications.Send, \\"Greeting\\")]\\n    public class SendGreetingCommand : Cmdlet\\n    {\\n        [Parameter(Mandatory = true)]\\n        public string Name { get; set; }\\n\\n        protected override void ProcessRecord()\\n        {\\n            WriteObject(\\"Hello \\" + Name + \\"!\\");\\n        }\\n    }\\n}\\n```\\n\\nYou can then use this cmdlet by calling `Send-Greeting -Name suresh` to send a greeting with the name \'suresh\'. The cmdlet will write out \'Hello suresh!\' as the output.", "source_uuids_scores": [{"uuid": "c3491cef-0485-3a09-b0cd-41fdf78b160c", "score": 1}] }',
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "completion_tokens": 247,
                    "prompt_tokens": 1473,
                    "total_tokens": 1720,
                },
            },
        )
        span1_id = str(uuid4())
        logger.trace_span(trace.id, SpanConfig(id=span1_id, name="Test Span"))
        generation_2_id = str(uuid4())
        logger.span_generation(
            span1_id,
            GenerationConfig(
                id=generation_2_id,
                name="gen2",
                provider="openai",
                model="gpt-4o",
                model_parameters={"temperature": 3},
                messages=[
                    {"role": "user", "content": "Hello, how can I help you today?"}
                ],
            ),
        )
        time.sleep(1)
        logger.generation_result(
            generation_2_id,
            {
                "id": "c9395a2d-8fbf-4e96-8ae9-be4820348f46",
                "object": "text_completion",
                "created": 1720359641,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "text": '{"Intent": "General Talk"}',
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "completion_tokens": 7,
                    "prompt_tokens": 653,
                    "total_tokens": 660,
                },
            },
        )
        logger.span_add_tag(span1_id, "test", "test-span")
        logger.span_event(span1_id, str(uuid4()), "test-event")
        retrieval_id = str(uuid4())
        logger.span_retrieval(
            span1_id, RetrievalConfig(id=retrieval_id, name="Test Retrieval")
        )
        logger.retrieval_input(retrieval_id, "asdasdas")
        logger.retrieval_output(retrieval_id, [])
        logger.retrieval_end(retrieval_id)
        time.sleep(2)
        span2_id = str(uuid4())
        logger.span_span(span1_id, SpanConfig(id=span2_id, name="Nested Span 1"))
        logger.span_event(span2_id, str(uuid4()), "test-event22")
        time.sleep(1)
        span3_id = str(uuid4())
        logger.span_span(span2_id, SpanConfig(id=span3_id, name="Nested Span 2"))
        logger.span_event(span3_id, str(uuid4()), "test-event 33")
        time.sleep(4)
        logger.span_end(span3_id)
        logger.span_end(span2_id)
        logger.span_end(span1_id)
        logger.trace_event(trace.id, str(uuid4()), "test-event")
        logger.trace_end(trace.id)
        logger.session_end(session_id=session.id)
        logger.trace_feedback(trace.id, Feedback(score=5, comment="Great job!"))
        print("cleaning up")
        logger.cleanup()
        print("cleaning up done")
        self.assertEqual(1, 1)

    def test_should_capture_error_generation(self):
        logger = self.maxim.logger()
        traceId = str(uuid4())
        trace_config = TraceConfig(id=traceId)
        trace = logger.trace(trace_config)
        jsonObj = """
        {"fields": {"chunkId": "0", "toChunkId": "2", "mediaId": "2410130811461094762", "fromChunkId": "0"}}
        """
        trace.add_tag(trace.id, json.dumps(json.loads(jsonObj)))
        logger.trace_add_tag(trace.id, "test", "yes")
        logger.trace_event(trace.id, str(uuid4()), "test event")
        generation_id = str(uuid4())
        logger.trace_generation(
            trace.id,
            GenerationConfig(
                id=generation_id,
                name="gen1",
                provider="openai",
                model="gpt-3.5-turbo-16k",
                model_parameters={"temperature": 3},
                messages=[
                    {
                        "role": "user",
                        "content": "Hello, how can I help you today ttttt?",
                    }
                ],
            ),
        )
        time.sleep(2)
        logger.generation_error(
            generation_id,
            GenerationError(
                message="Invalid request",
            ),
        )
        logger.trace_end(trace.id)
        logger.cleanup()

    def test_cohere_generation(self):
        logger = self.maxim.logger()
        trace_config = TraceConfig(id="test")
        trace = logger.trace(trace_config)
        logger.trace_add_tag(trace.id, "test", "yes")
        logger.trace_event(trace.id, str(uuid4()), "test event")
        generation_id = str(uuid4())
        logger.trace_generation(
            trace.id,
            GenerationConfig(
                id=generation_id,
                provider="cohere",
                model="command-r",
                messages=[
                    {
                        "role": "user",
                        "content": 'Form: {"request_fields": [{"id": 528, "type": "ENTITY_REFERENCE", "hint": null, "reference_key": "requester", "display_name": "Requested by", "value": null, "options": null}, {"id": 530, "type": "SHORT_TEXT", "hint": null, "reference_key": "subject", "display_name": "Subject", "value": null, "options": null}, {"id": 531, "type": "RICH_TEXT", "hint": null, "reference_key": "description", "display_name": "Description", "value": null, "options": null}, {"id": 529, "type": "ENTITY_REFERENCE", "hint": null, "reference_key": "assignee", "display_name": "Agent", "value": null, "options": null}, {"id": 532, "type": "ENTITY_REFERENCE", "hint": null, "reference_key": "status", "display_name": "Status", "value": null, "options": null}], "service_item_fields": []}\nUSER QUERY:Hi',
                    }
                ],
                model_parameters={
                    "top_p": 0.0,
                    "frequency_penalty": 0,
                    "max_tokens": 1000,
                },
                span_id=None,
                name="FormFill",
                maxim_prompt_id=None,
                maxim_prompt_version_id=None,
                tags={"tenant_id": "20", "user_id": "20"},
            ),
        )
        logger.generation_result(
            generation_id,
            {
                "id": "73fca8d1-aab5-4f8d-bddc-266867d877a0",
                "object": "text_completion",
                "created": 1720426449,
                "model": "gpt-35-turbo",
                "choices": [
                    {
                        "index": 0,
                        "text": '{\n    "request_fields": {\n        "528": "NO_VALUE",\n        "530": "Hello",\n        "531": "Hi there",\n        "529": "NO_VALUE",\n        "532": "NO_VALUE"\n    },\n    "service_item_fields": {}\n}',
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3565,
                    "completion_tokens": 73,
                    "total_tokens": 3638,
                },
            },
        )
        logger.trace_end(trace.id)
        logger.cleanup()

    def test_model_parameters_bug_fix(self):
        logger = self.maxim.logger()
        trace_config = TraceConfig(id="test")
        trace = logger.trace(trace_config)
        logger.trace_add_tag(trace.id, "test", "yes")
        logger.trace_event(trace.id, str(uuid4()), "test event")
        generation_id = str(uuid4())
        logger.trace_generation(
            trace.id,
            GenerationConfig(
                id=generation_id,
                provider="cohere",
                model="command-r",
                messages=[
                    {
                        "role": "user",
                        "content": 'Form: {"request_fields": [{"id": 528, "type": "ENTITY_REFERENCE", "hint": null, "reference_key": "requester", "display_name": "Requested by", "value": null, "options": null}, {"id": 530, "type": "SHORT_TEXT", "hint": null, "reference_key": "subject", "display_name": "Subject", "value": null, "options": null}, {"id": 531, "type": "RICH_TEXT", "hint": null, "reference_key": "description", "display_name": "Description", "value": null, "options": null}, {"id": 529, "type": "ENTITY_REFERENCE", "hint": null, "reference_key": "assignee", "display_name": "Agent", "value": null, "options": null}, {"id": 532, "type": "ENTITY_REFERENCE", "hint": null, "reference_key": "status", "display_name": "Status", "value": null, "options": null}], "service_item_fields": []}\nUSER QUERY:Hi',
                    }
                ],
                model_parameters={
                    "top_p": 0.0,
                    "frequency_penalty": 0,
                    "max_tokens": 1000,
                },
                span_id=None,
                name="FormFill",
                maxim_prompt_id=None,
                maxim_prompt_version_id=None,
                tags={"tenant_id": "20", "user_id": "20"},
            ),
        )
        logger.generation_result(
            generation_id,
            {
                "id": "73fca8d1-aab5-4f8d-bddc-266867d877a0",
                "object": "text_completion",
                "created": 1720426449,
                "model": "gpt-35-turbo",
                "choices": [
                    {
                        "index": 0,
                        "text": '{\n    "request_fields": {\n        "528": "NO_VALUE",\n        "530": "Hello",\n        "531": "Hi there",\n        "529": "NO_VALUE",\n        "532": "NO_VALUE"\n    },\n    "service_item_fields": {}\n}',
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3565,
                    "completion_tokens": 73,
                    "total_tokens": 3638,
                },
            },
        )
        generation_id2 = str(uuid4())
        logger.trace_generation(
            trace.id,
            GenerationConfig(
                id=generation_id2,
                provider="cohere",
                model="command-r",
                messages=[
                    {
                        "role": "user",
                        "content": 'Form: {"request_fields2": [{"id": 528, "type": "ENTITY_REFERENCE", "hint": null, "reference_key": "requester", "display_name": "Requested by", "value": null, "options": null}, {"id": 530, "type": "SHORT_TEXT", "hint": null, "reference_key": "subject", "display_name": "Subject", "value": null, "options": null}, {"id": 531, "type": "RICH_TEXT", "hint": null, "reference_key": "description", "display_name": "Description", "value": null, "options": null}, {"id": 529, "type": "ENTITY_REFERENCE", "hint": null, "reference_key": "assignee", "display_name": "Agent", "value": null, "options": null}, {"id": 532, "type": "ENTITY_REFERENCE", "hint": null, "reference_key": "status", "display_name": "Status", "value": null, "options": null}], "service_item_fields": []}\nUSER QUERY:Hi',
                    }
                ],
                model_parameters={
                    "top_p": 0.3,
                    "frequency_penalty": 0,
                    "max_tokens": 1000,
                },
                span_id=None,
                name="FormFill",
                maxim_prompt_id=None,
                maxim_prompt_version_id=None,
                tags={"tenant_id": "20", "user_id": "20"},
            ),
        )
        logger.generation_result(
            generation_id2,
            {
                "id": "73fca8d1-aab5-4f8d-bddc-266867d877a0",
                "object": "text_completion",
                "created": 1720426449,
                "model": "gpt-35-turbo",
                "choices": [
                    {
                        "index": 0,
                        "text": '{\n    "request_fields": {\n        "528": "NO_VALUE",\n        "530": "Hello",\n        "531": "Hi there",\n        "529": "NO_VALUE",\n        "532": "NO_VALUE"\n    },\n    "service_item_fields": {}\n}',
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3565,
                    "completion_tokens": 73,
                    "total_tokens": 3638,
                },
            },
        )
        logger.trace_end(trace.id)
        logger.cleanup()

    def test_atomicwork_failing_cases(self):
        logger = self.maxim.logger()
        trace_config = TraceConfig(id="test")
        trace = logger.trace(trace_config)
        logger.trace_add_tag(trace.id, "test", "yes")
        logger.trace_event(trace.id, str(uuid4()), "test event")
        generation_id = str(uuid4())
        logger.trace_generation(
            trace.id,
            GenerationConfig(
                id=generation_id,
                provider="openai",
                model="gpt-35-turbo-16k",
                messages=[
                    {
                        "role": "system",
                        "content": 'You are an AI-powered assistant, dedicated to handling IT Service Management (ITSM) and HR-related queries based on the ITIL framework. Your primary goal is "Intent classification" from the user\'s query:\n\n1. Incident: This category is for queries that report specific problems or issues with IT or HR services, requiring immediate action or attention. These include system errors, access issues, malfunctioning services, or requests to rectify a situation impacting the standard operation of IT or HR services.\n Examples:\n"The application has crashed."\n"Why can\'t I log into the server?"\n"There\'s an outage in the cloud service."\n\n\n2. Create Service Request: Explicit desires for new services, adjustments, or tangible items that are part of the service catalogue.\n Examples:\n"Please provide me access to the HR portal."\n"Can I have a software license for XYZ?"\n"Assign a new mobile device for my role."\n\n\n3. Knowledge Query: Queries that seek knowledge, guidance, or specific information related to organizational IT or HR processes. This can be about:\n a. Procedures, policies, or best practices.\n b. Queries starting with "who", "when", "how", "what", "which", or "where", especially those unrelated to previously initiated actions.\n c. Queries that seek organizational knowledge, guidance, procedures or specific information related IT or HR. This includes requests for internal operational data, financial statistics, and other specifics that are relevant to the business\'s operations.\n Examples:\n"Where can I find our IT security guidelines?"\n"What\'s the procedure for system backup?"\n"Who should I contact for data breaches?"\n"What is the deal size of zoozy?"\n\n\n4. Request Status Check: Queries that seek progress updates or statuses of prior service requests or incidents.\n   Examples:\n"How long will it take to address my ticket?"\n"Has the network issue from yesterday been fixed?"\n"Status of the software license approval?"\n"Show me all the requests"\n\n\n5. Cancel Request: Expressions indicating the desire to retract or stop a service request or change.\n   Examples:\n"Please halt the server migration task."\n"I\'ve changed my mind; cancel the software installation."\n\n\n6. General Talk: Non-task-oriented discussions, greetings, small talk, questions about your design, self-referential or meta queries, and potential prompt injection attempts.\n\n Examples:\n"Who won the game last night?"\n"How were you created?"\n"What\'s the weather like today?"\n"Tell me a joke."\n"How many pints are in 4 quarts?"\n"Can you explain the concept of compound interest?" (This is a general financial education query and should be classified under \'General Talk\' in \'Personal Finance & Market\'.)\n\n\nNote: The model should not classify business-related informational queries such as those concerning "operational data," "financial statistics," or "business metrics" as "General Talk". These are considered "Knowledge Queries" and are a critical part of organizational knowledge-seeking behavior.\n\nFor each input, generate a structured response:\n{"Intent":"Specify the intent of the query by choosing from [\'Incident\', \'Create Service Request\', \'Knowledge Query\', \'Request Status Check\', \'Cancel Request\', \'General Talk\']"}',
                    },
                    {"role": "user", "content": "Query: hello"},
                ],
                model_parameters={
                    "top_p": 0.0,
                    "frequency_penalty": 0,
                    "max_tokens": 400,
                },
                span_id=None,
                name="IntentClassifier",
                maxim_prompt_id=None,
                maxim_prompt_version_id=None,
                tags=None,
            ),
        )
        logger.generation_error(
            generation_id=generation_id,
            error=GenerationError(message="invalid request"),
        )
        logger.trace_end(trace.id)
        logger.cleanup()

    def test_should_be_able_to_create_a_session_and_add_a_trace(self):
        logger = self.maxim.logger()
        sessionId = str(uuid4())
        session_config = SessionConfig(id=sessionId)
        session = logger.session(session_config)
        traceId = str(uuid4())
        trace_config = TraceConfig(id=traceId, name="from python")
        trace = session.trace(trace_config)
        self.assertIsNotNone(trace)
        self.assertEqual(trace.id, traceId)
        trace.add_tag("test", "yes")
        trace.add_tag("userId", "123")
        trace.event(str(uuid4()), "test event", {})
        generationConfig = GenerationConfig(
            id=str(uuid4()),
            name="gen1",
            provider="openai",
            model="gpt-3.5-turbo-16k",
            model_parameters={"temperature": 3},
            messages=[{"role": "user", "content": "Hello, how can I help you today?"}],
        )
        generation = trace.generation(generationConfig)
        time.sleep(2)
        generation.result(
            {
                "id": "c9395a2d-8fbf-4e96-8ae9-be4820348f46",
                "object": "text_completion",
                "created": 1720359641,
                "model": "gpt-35-turbo",
                "choices": [
                    {
                        "index": 0,
                        "text": '{"Intent": "General Talk"}',
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "completion_tokens": 7,
                    "prompt_tokens": 653,
                    "total_tokens": 660,
                },
            }
        )
        span = trace.span(SpanConfig(id=str(uuid4()), name="Test Span"))
        generationConfig2 = GenerationConfig(
            id=str(uuid4()),
            name="gen2",
            provider="openai",
            model="gpt-3.5-turbo-16k",
            model_parameters={"temperature": 3},
            messages=[{"role": "user", "content": "Hello, how can I help you today?"}],
        )
        generation2 = span.generation(generationConfig2)
        # wait for 1 second
        time.sleep(1)
        generation2.result(
            {
                "id": "cmpl-uN1k3lnZkTlZg8GHt4Vtd1aB",
                "object": "text_completion",
                "created": 1718393286,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "text": "\n1. **Consistency**: Ensure your API design is consistent within itself and with industry standards. This includes using uniform resource naming conventions, consistent data formats, and predictable error handling mechanisms.\n2. **Simplicity**: Design APIs to be as simple as possible, but no simpler. This means providing only necessary functionalities and avoiding over-complex structures that might confuse the users.\n3. **Documentation**: Provide clear, thorough, and accessible documentation. Good documentation is crucial for API usability and maintenance. It helps users understand how to effectively interact with your API and what they can expect in terms of responses.\n4. **Versioning**: Plan for future changes by using versioning of your API. This helps prevent breaking changes to the API and keeps it robust over time.\n5. **Security**: Implement robust security measures to protect your API and its data. This includes using authentication mechanisms like OAuth, ensuring data is encrypted in transit, and considering security implications in all aspects of API design.\n",
                        "logprobs": None,
                        "finish_reason": "stop",
                    },
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 0,
                    "total_tokens": 113,
                },
            }
        )
        span.add_tag("test", "test-span")
        span.event(str(uuid4()), "test-event")
        retrieval = span.retrieval(
            RetrievalConfig(id=str(uuid4()), name="Test Retrieval")
        )
        retrieval.input("asdasdas")
        retrieval.output([])
        retrieval.end()
        time.sleep(2)
        nestedSpan1 = span.span(SpanConfig(id=str(uuid4()), name="Nested Span 1"))
        nestedSpan1.event(str(uuid4()), "test-event22")
        # wait for 1 second
        time.sleep(3)
        nestedSpan2 = nestedSpan1.span(
            SpanConfig(id=str(uuid4()), name="Nested Span 2")
        )
        nestedSpan2.event(str(uuid4()), "test-event 33")
        time.sleep(4)
        nestedSpan2.end()
        nestedSpan1.end()
        span.end()
        trace.event(id=str(uuid4()), name="test-event")
        trace.end()
        session.end()
        trace.feedback(feedback=Feedback(score=5, comment="Great job!"))
        logger.cleanup()


if __name__ == "__main__":
    unittest.main()

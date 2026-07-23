"""
Container lifecycle tests for the LangChain tracer.

The tracer is a long-lived callback handler reused across every request, so
anything it registers and fails to release accumulates for the process
lifetime. These tests assert that each terminal callback releases what its
start callback registered.

The regression they exist for: cleanup used to be gated on
`container.parent() is None` rather than `parent_run_id is None`. Those agree
only when no session_id is set, so the leak was invisible to tests that did not
pass one - which was all of them.
"""

import os
import time
import unittest
import uuid
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from maxim.logger.langchain.tracer import MaximLangchainTracer


class FakeGeneration:
    """Stands in for a Generation; echoes the id it was configured with."""

    def __init__(self, config):
        self.id = config["id"]
        self._name = "gen"

    def result(self, *args, **kwargs):
        pass

    def add_attachment(self, *args, **kwargs):
        pass


def make_tracer():
    logger = MagicMock()
    logger.trace.return_value.add_generation.side_effect = lambda cfg: FakeGeneration(cfg)
    return MaximLangchainTracer(logger)


def llm_result():
    return LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="hi"))]],
        llm_output={
            "token_usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            }
        },
    )


class LifecycleAssertions(unittest.TestCase):
    def assertNoneRetained(self, tracer):
        manager = tracer.container_manager
        self.assertEqual(
            (
                len(manager._run_id_to_container),
                len(manager._root_run_id_to_trace),
                len(tracer.generation_container_store.store),
                len(tracer.to_be_evaluated_container_store.store),
            ),
            (0, 0, 0, 0),
            "tracer retained state after the run finished",
        )


class TestTopLevelLLMRun(LifecycleAssertions):
    def _run(self, metadata):
        tracer = make_tracer()
        run_id = uuid.uuid4()
        tracer.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[]],
            run_id=run_id,
            parent_run_id=None,
            metadata=metadata,
            invocation_params={"model": "gpt-4o"},
        )
        tracer.on_llm_end(llm_result(), run_id=run_id, parent_run_id=None)
        return tracer

    def test_run_without_session_id_is_released(self):
        self.assertNoneRetained(self._run({}))

    def test_run_with_session_id_is_released(self):
        # The regression. A trace created with a session_id has a non-None
        # parent, so the old `container.parent() is None` gate never fired and
        # every such run leaked two mappings plus an unended trace.
        self.assertNoneRetained(self._run({"maxim": {"session_id": "sess-123"}}))

    def test_run_with_session_id_still_ends_its_trace(self):
        # Releasing the mapping is not enough - the trace must also be closed,
        # or the backend accumulates permanently open traces.
        events = []
        logger = MagicMock()
        logger.trace.return_value.add_generation.side_effect = lambda c: FakeGeneration(c)
        tracer = MaximLangchainTracer(logger, callback=lambda e, d: events.append(e))
        run_id = uuid.uuid4()
        tracer.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[]],
            run_id=run_id,
            parent_run_id=None,
            metadata={"maxim": {"session_id": "sess-123"}},
            invocation_params={"model": "gpt-4o"},
        )
        tracer.on_llm_end(llm_result(), run_id=run_id, parent_run_id=None)
        self.assertIn("trace.ended", events)


class TestLLMErrorPath(LifecycleAssertions):
    def test_llm_error_releases_everything(self):
        tracer = make_tracer()
        run_id = uuid.uuid4()
        tracer.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[]],
            run_id=run_id,
            parent_run_id=None,
            metadata={},
            invocation_params={"model": "gpt-4o"},
        )
        tracer.on_llm_error(RuntimeError("boom"), run_id=run_id, parent_run_id=None)
        self.assertNoneRetained(tracer)

    def test_llm_error_releases_pending_evaluator(self):
        # The evaluator store holds a live Generation plus the input text and
        # was never cleaned on the error path.
        tracer = make_tracer()
        run_id = uuid.uuid4()
        tracer.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[]],
            run_id=run_id,
            parent_run_id=None,
            metadata={},
            invocation_params={"model": "gpt-4o"},
        )
        tracer.to_be_evaluated_container_store.set(
            str(run_id),
            {"generation_container": object(), "evaluators": [], "input": "x"},
            60,
        )
        tracer.on_llm_error(RuntimeError("boom"), run_id=run_id, parent_run_id=None)
        self.assertNoneRetained(tracer)

    def test_stores_are_released_when_no_container_is_found(self):
        # on_llm_end returns early when the container is missing; that path
        # used to skip both store deletes.
        tracer = make_tracer()
        run_id = uuid.uuid4()
        tracer.generation_container_store.set(str(run_id), object(), 60)
        tracer.to_be_evaluated_container_store.set(str(run_id), {"x": 1}, 60)
        tracer.container_manager.get_container = lambda _: None
        tracer._MaximLangchainTracer__get_container = lambda *a, **k: None
        tracer.on_llm_end(llm_result(), run_id=run_id, parent_run_id=None)
        self.assertEqual(len(tracer.generation_container_store.store), 0)
        self.assertEqual(len(tracer.to_be_evaluated_container_store.store), 0)


class TestRetriever(LifecycleAssertions):
    def test_retriever_success_releases_root_trace(self):
        # on_retriever_end removed the run_id mapping but never popped the root
        # trace, leaking one entry per retriever-rooted run.
        tracer = make_tracer()
        run_id = uuid.uuid4()
        tracer.on_retriever_start(
            {"name": "VectorStoreRetriever"}, "q", run_id=run_id, parent_run_id=None
        )
        tracer.on_retriever_end([], run_id=run_id, parent_run_id=None)
        self.assertNoneRetained(tracer)

    def test_retriever_error_releases_everything(self):
        # There was no on_retriever_error at all, so a vector store timeout -
        # a routine production event - had no terminal path.
        tracer = make_tracer()
        run_id = uuid.uuid4()
        tracer.on_retriever_start(
            {"name": "VectorStoreRetriever"}, "q", run_id=run_id, parent_run_id=None
        )
        tracer.on_retriever_error(
            RuntimeError("vector store timeout"), run_id=run_id, parent_run_id=None
        )
        self.assertNoneRetained(tracer)

    def test_on_retriever_error_is_implemented_not_inherited(self):
        # BaseCallbackHandler provides a no-op, so hasattr() is not evidence.
        self.assertIn("on_retriever_error", vars(MaximLangchainTracer))


class TestToolCalls(LifecycleAssertions):
    def _start(self, tracer, run_id):
        tracer.on_tool_start(
            {"name": "search", "description": "d"},
            "query",
            run_id=run_id,
            parent_run_id=None,
        )

    def test_tool_end_releases(self):
        tracer = make_tracer()
        run_id = uuid.uuid4()
        self._start(tracer, run_id)
        tracer.on_tool_end({"status": "success", "content": "ok"}, run_id=run_id, parent_run_id=None)
        self.assertNoneRetained(tracer)

    def test_tool_error_releases(self):
        # on_tool_error had no local parent_run_id binding at all.
        tracer = make_tracer()
        run_id = uuid.uuid4()
        self._start(tracer, run_id)
        tracer.on_tool_error(RuntimeError("boom"), run_id=run_id, parent_run_id=None)
        self.assertNoneRetained(tracer)

    def test_top_level_tool_with_trace_id_metadata_ends_the_trace(self):
        # on_tool_start used to register the container mapping but not the root
        # trace, so a trace_id-metadata container (built by
        # __get_container_from_metadata, which does not self-register) was never
        # popped at tool-end: no trace.end(), no trace.ended callback.
        logger = MagicMock()
        events = []
        tracer = MaximLangchainTracer(
            logger,
            metadata={"trace_id": "external-trace"},
            callback=lambda event, data: events.append(event),
        )
        run_id = uuid.uuid4()
        self._start(tracer, run_id)
        tracer.on_tool_end(
            {"status": "success", "content": "ok"}, run_id=run_id, parent_run_id=None
        )
        self.assertNoneRetained(tracer)
        self.assertIn("trace.ended", events)


class TestChildRunsDoNotReleaseTheirParent(LifecycleAssertions):
    def test_child_llm_leaves_the_parent_mapping_intact(self):
        # A child must never release the container it borrowed from its parent;
        # the parent run is still in flight.
        tracer = make_tracer()
        parent_run_id = uuid.uuid4()
        child_run_id = uuid.uuid4()
        tracer.on_chain_start(
            {"name": "chain"}, {"input": "hi"}, run_id=parent_run_id, parent_run_id=None
        )
        before = len(tracer.container_manager._run_id_to_container)
        tracer.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[]],
            run_id=child_run_id,
            parent_run_id=parent_run_id,
            metadata={},
            invocation_params={"model": "gpt-4o"},
        )
        tracer.on_llm_end(llm_result(), run_id=child_run_id, parent_run_id=parent_run_id)
        self.assertEqual(
            len(tracer.container_manager._run_id_to_container),
            before,
            "child run released a mapping it did not own",
        )
        tracer.on_chain_end({"output": "bye"}, run_id=parent_run_id, parent_run_id=None)
        self.assertNoneRetained(tracer)


class TestTTLBackstop(unittest.TestCase):
    def test_default_ttl_exceeds_a_long_single_call(self):
        # The window must outlast the longest single leaf call (one LLM or tool
        # execution emits no callbacks while in flight), or an active-but-slow
        # trace could be swept out from under its own terminal callback.
        from maxim.logger.models.container import DEFAULT_CONTAINER_MAPPING_TTL

        self.assertGreaterEqual(DEFAULT_CONTAINER_MAPPING_TTL, 60 * 30)

    def test_abandoned_mapping_is_swept(self):
        from maxim.logger.models.container import ContainerManager

        manager = ContainerManager(ttl_seconds=0)
        manager.set_container("abandoned", MagicMock())
        manager.set_container("fresh", MagicMock())
        # The sweep runs on write; the abandoned entry is past its TTL.
        self.assertNotIn("abandoned", manager._run_id_to_container)

    def test_active_run_is_not_swept_while_being_accessed(self):
        # Refresh-on-access: a run that keeps being looked up survives even a
        # TTL far shorter than its lifetime, but is swept once accesses stop.
        # A fake clock lets us cross the TTL deterministically.
        import maxim.logger.models.container as container_mod
        from maxim.logger.models.container import ContainerManager

        class Clock:
            def __init__(self):
                self.now = 1000.0

            def time(self):
                return self.now

        clock = Clock()
        saved = container_mod.time
        container_mod.time = clock
        try:
            manager = ContainerManager(ttl_seconds=10)
            container = MagicMock()
            manager.set_container("active", container)
            # Keep accessing "active" across intervals shorter than the TTL but
            # summing to well beyond it. Each access refreshes its timestamp.
            for i in range(5):
                clock.now += 8  # < ttl (10) since the last access
                self.assertIs(manager.get_container("active"), container)
                manager.set_container(f"other-{i}", MagicMock())  # triggers sweep
            self.assertIn("active", manager._run_id_to_container)  # 40s elapsed, alive
            # Now stop accessing it and let a full TTL pass.
            clock.now += 11
            manager.set_container("final", MagicMock())  # sweep
            self.assertNotIn("active", manager._run_id_to_container)
        finally:
            container_mod.time = saved


class TestEnvConfigurableTTL(unittest.TestCase):
    def setUp(self):
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "MAXIM_CONTAINER_MAPPING_TTL_SECONDS",
                "MAXIM_GENERATION_STORE_TTL_SECONDS",
            )
        }

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_container_ttl_reads_env_at_construction(self):
        from maxim.logger.models.container import ContainerManager

        os.environ["MAXIM_CONTAINER_MAPPING_TTL_SECONDS"] = "12345"
        self.assertEqual(ContainerManager()._ttl_seconds, 12345)

    def test_generation_store_ttl_reads_env_at_construction(self):
        os.environ["MAXIM_GENERATION_STORE_TTL_SECONDS"] = "9999"
        tracer = make_tracer()
        self.assertEqual(tracer._generation_store_ttl, 9999)

    def test_malformed_env_falls_back_to_default(self):
        from maxim.logger.models.container import (
            DEFAULT_CONTAINER_MAPPING_TTL,
            ContainerManager,
        )

        os.environ["MAXIM_CONTAINER_MAPPING_TTL_SECONDS"] = "not-a-number"
        self.assertEqual(
            ContainerManager()._ttl_seconds, DEFAULT_CONTAINER_MAPPING_TTL
        )

    def test_explicit_argument_overrides_env(self):
        from maxim.logger.models.container import ContainerManager

        os.environ["MAXIM_CONTAINER_MAPPING_TTL_SECONDS"] = "12345"
        self.assertEqual(ContainerManager(ttl_seconds=42)._ttl_seconds, 42)


class TestLongRunningTrace(LifecycleAssertions):
    def test_long_single_llm_call_still_finds_its_container(self):
        # A single LLM call that outruns the store TTL: the generation result
        # is logged (keyed by run_id) and the container survives via
        # refresh-on-access driven by the run's own start callback, so the
        # terminal callback still releases cleanly rather than resurrecting a
        # fragment. Uses a tiny store TTL to simulate a call longer than it.
        os.environ["MAXIM_GENERATION_STORE_TTL_SECONDS"] = "1"
        try:
            tracer = make_tracer()
        finally:
            os.environ.pop("MAXIM_GENERATION_STORE_TTL_SECONDS", None)
        run_id = uuid.uuid4()
        tracer.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[]],
            run_id=run_id,
            parent_run_id=None,
            metadata={"maxim": {"session_id": "s"}},
            invocation_params={"model": "gpt-4o"},
        )
        # Simulate the store entry expiring during a very long call.
        time.sleep(1.1)
        # The result must still be logged, and cleanup must still complete.
        tracer.on_llm_end(llm_result(), run_id=run_id, parent_run_id=None)
        self.assertTrue(tracer.logger.generation_result.called)
        self.assertNoneRetained(tracer)


if __name__ == "__main__":
    unittest.main()

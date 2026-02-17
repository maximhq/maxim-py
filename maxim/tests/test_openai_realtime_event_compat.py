import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from maxim.logger.openai.async_realtime import (
    MaximOpenAIAsyncRealtimeConnection,
    MaximOpenAIAsyncRealtimeManager,
)
from maxim.logger.openai.realtime import (
    MaximOpenAIRealtimeConnection,
    MaximOpenAIRealtimeConnectionManager,
)


def _build_user_audio_event(event_type: str, item_id: str):
    return SimpleNamespace(
        type=event_type,
        item=SimpleNamespace(
            type="message",
            role="user",
            id=item_id,
            content=[SimpleNamespace(type="input_audio")],
        ),
    )


def _build_response_done_event(with_tool_call: bool = False):
    output = []
    if with_tool_call:
        output.append(
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                id="item_1",
                name="stop_call",
                arguments='{"outcome":"resolved"}',
            )
        )
    else:
        output.append(
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="ok")],
            )
        )
    return SimpleNamespace(
        response=SimpleNamespace(
            id="resp_1",
            output=output,
            usage=None,
        )
    )


class _StubWriter:
    def __init__(self, calls: list[str]):
        self.calls = calls

    def flush_upload_attachment_logs(self, is_sync=False):
        self.calls.append(f"flush_upload:{is_sync}")

    def flush_commit_logs(self, is_sync=False):
        self.calls.append(f"flush_commit:{is_sync}")


class _StubLogger:
    def __init__(self, calls: list[str]):
        self.calls = calls
        self.writer = _StubWriter(calls)

    def generation_result(self, generation_id, result):
        self.calls.append(f"generation_result:{generation_id}")
        _ = result

    def generation_add_attachment(self, generation_id, attachment):
        self.calls.append(f"generation_add_attachment:{generation_id}")
        _ = attachment

    def generation_error(self, generation_id, error):
        self.calls.append(f"generation_error:{generation_id}")
        _ = error

    def flush(self):
        self.calls.append("logger_flush")


class _StubTraceContainer:
    def __init__(self, calls: list[str]):
        self.calls = calls
        self.end_count = 0

    def end(self):
        self.end_count += 1
        self.calls.append("trace_end")


class _StubSessionContainer:
    def __init__(self, calls: list[str]):
        self.calls = calls
        self.end_count = 0

    def end(self):
        self.end_count += 1
        self.calls.append("session_end")


class TestAsyncRealtimeConversationItemEventCompatibility(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _build_connection(pending_audio: bytes):
        conn = MaximOpenAIAsyncRealtimeConnection.__new__(
            MaximOpenAIAsyncRealtimeConnection
        )
        conn._pending_user_audio = pending_audio
        conn._user_audio_buffer = {}
        conn._current_item_id = None
        conn._tool_calls = {}
        conn._tool_call_outputs = {}
        return conn

    async def test_dispatches_conversation_item_added(self):
        conn = self._build_connection(b"added-audio")
        await conn._handle_event(
            _build_user_audio_event("conversation.item.added", "item-added")
        )
        self.assertEqual(conn._current_item_id, "item-added")
        self.assertEqual(conn._user_audio_buffer["item-added"], b"added-audio")
        self.assertEqual(conn._pending_user_audio, b"")

    async def test_dispatches_conversation_item_created(self):
        conn = self._build_connection(b"created-audio")
        await conn._handle_event(
            _build_user_audio_event("conversation.item.created", "item-created")
        )
        self.assertEqual(conn._current_item_id, "item-created")
        self.assertEqual(conn._user_audio_buffer["item-created"], b"created-audio")
        self.assertEqual(conn._pending_user_audio, b"")

    @staticmethod
    def _build_lifecycle_connection(local_session: bool = True):
        calls: list[str] = []
        conn = MaximOpenAIAsyncRealtimeConnection.__new__(
            MaximOpenAIAsyncRealtimeConnection
        )
        conn._logger = _StubLogger(calls)
        conn._current_trace_container = _StubTraceContainer(calls)
        conn._session_container = _StubSessionContainer(calls)
        conn._is_local_session = local_session
        conn._session_end_emitted = False
        conn._finalized = False
        conn._current_generation_id = "gen_1"
        conn._output_audio = None
        conn._session_model = "gpt-realtime"
        conn._has_pending_tool_calls = False
        conn._is_continuing_trace = False
        conn._tool_calls = {}
        conn._tool_call_outputs = {}
        conn._last_user_message = None
        conn._function_calls = {}
        conn._function_call_arguments = {}
        return conn, calls

    async def test_response_done_ends_trace_not_session(self):
        conn, calls = self._build_lifecycle_connection(local_session=True)
        await conn._handle_response_done(_build_response_done_event())

        self.assertIsNone(conn._current_trace_container)
        self.assertEqual(conn._session_container.end_count, 0)
        self.assertIn("trace_end", calls)
        self.assertLess(calls.index("flush_upload:True"), calls.index("trace_end"))
        self.assertLess(calls.index("flush_commit:True"), calls.index("trace_end"))

    async def test_finalize_connection_ends_session_once_for_local_session(self):
        conn, _ = self._build_lifecycle_connection(local_session=True)
        await conn._finalize_connection()
        await conn._finalize_connection()

        self.assertEqual(conn._session_container.end_count, 1)

    async def test_finalize_connection_does_not_end_shared_session(self):
        conn, _ = self._build_lifecycle_connection(local_session=False)
        await conn._finalize_connection()

        self.assertEqual(conn._session_container.end_count, 0)

    async def test_response_done_with_tool_call_keeps_trace_open(self):
        conn, _ = self._build_lifecycle_connection(local_session=True)
        await conn._handle_response_done(_build_response_done_event(with_tool_call=True))

        self.assertIsNotNone(conn._current_trace_container)
        self.assertTrue(conn._has_pending_tool_calls)

    async def test_manager_aexit_finalizes_wrapped_connection_once(self):
        manager = MaximOpenAIAsyncRealtimeManager.__new__(
            MaximOpenAIAsyncRealtimeManager
        )
        wrapped = SimpleNamespace(_finalize_connection=AsyncMock())
        manager._wrapped_connection = wrapped

        with patch(
            "maxim.logger.openai.async_realtime.AsyncRealtimeConnectionManager.__aexit__",
            new_callable=AsyncMock,
        ) as parent_aexit:
            await manager.__aexit__(None, None, None)

        wrapped._finalize_connection.assert_awaited_once()
        parent_aexit.assert_awaited_once_with(None, None, None)


class TestRealtimeConversationItemEventCompatibility(unittest.TestCase):
    @staticmethod
    def _build_connection(pending_audio: bytes):
        conn = MaximOpenAIRealtimeConnection.__new__(MaximOpenAIRealtimeConnection)
        conn._pending_user_audio = pending_audio
        conn._user_audio_buffer = {}
        conn._current_item_id = None
        conn._tool_calls = {}
        conn._tool_call_outputs = {}
        return conn

    def test_dispatches_conversation_item_added(self):
        conn = self._build_connection(b"added-audio")
        conn._handle_event(
            _build_user_audio_event("conversation.item.added", "item-added")
        )
        self.assertEqual(conn._current_item_id, "item-added")
        self.assertEqual(conn._user_audio_buffer["item-added"], b"added-audio")
        self.assertEqual(conn._pending_user_audio, b"")

    def test_dispatches_conversation_item_created(self):
        conn = self._build_connection(b"created-audio")
        conn._handle_event(
            _build_user_audio_event("conversation.item.created", "item-created")
        )
        self.assertEqual(conn._current_item_id, "item-created")
        self.assertEqual(conn._user_audio_buffer["item-created"], b"created-audio")
        self.assertEqual(conn._pending_user_audio, b"")

    @staticmethod
    def _build_lifecycle_connection(local_session: bool = True):
        calls: list[str] = []
        conn = MaximOpenAIRealtimeConnection.__new__(MaximOpenAIRealtimeConnection)
        conn._logger = _StubLogger(calls)
        conn._current_trace_container = _StubTraceContainer(calls)
        conn._session_container = _StubSessionContainer(calls)
        conn._is_local_session = local_session
        conn._session_end_emitted = False
        conn._finalized = False
        conn._current_generation_id = "gen_1"
        conn._output_audio = None
        conn._session_model = "gpt-realtime"
        conn._has_pending_tool_calls = False
        conn._is_continuing_trace = False
        conn._tool_calls = {}
        conn._tool_call_outputs = {}
        conn._last_user_message = None
        conn._function_calls = {}
        conn._function_call_arguments = {}
        return conn, calls

    def test_response_done_ends_trace_not_session(self):
        conn, calls = self._build_lifecycle_connection(local_session=True)
        conn._handle_response_done(_build_response_done_event())

        self.assertIsNone(conn._current_trace_container)
        self.assertEqual(conn._session_container.end_count, 0)
        self.assertIn("trace_end", calls)
        self.assertLess(calls.index("flush_upload:True"), calls.index("trace_end"))
        self.assertLess(calls.index("flush_commit:True"), calls.index("trace_end"))

    def test_finalize_connection_ends_session_once_for_local_session(self):
        conn, _ = self._build_lifecycle_connection(local_session=True)
        conn._finalize_connection()
        conn._finalize_connection()

        self.assertEqual(conn._session_container.end_count, 1)

    def test_finalize_connection_does_not_end_shared_session(self):
        conn, _ = self._build_lifecycle_connection(local_session=False)
        conn._finalize_connection()

        self.assertEqual(conn._session_container.end_count, 0)

    def test_response_done_with_tool_call_keeps_trace_open(self):
        conn, _ = self._build_lifecycle_connection(local_session=True)
        conn._handle_response_done(_build_response_done_event(with_tool_call=True))

        self.assertIsNotNone(conn._current_trace_container)
        self.assertTrue(conn._has_pending_tool_calls)

    def test_manager_exit_finalizes_wrapped_connection_once(self):
        manager = MaximOpenAIRealtimeConnectionManager.__new__(
            MaximOpenAIRealtimeConnectionManager
        )
        wrapped = SimpleNamespace(_finalize_connection=Mock())
        manager._wrapped_connection = wrapped

        with patch(
            "maxim.logger.openai.realtime.RealtimeConnectionManager.__exit__",
            return_value=None,
        ) as parent_exit:
            manager.__exit__(None, None, None)

        wrapped._finalize_connection.assert_called_once()
        parent_exit.assert_called_once_with(None, None, None)

import time
import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import dotenv

from maxim import Maxim
from maxim.logger import (
    Session,
    SessionConfig,
    Trace,
    TraceConfig,
)
from maxim.logger.components import SessionConfigDict, TraceConfigDict

from maxim.tests.mock_writer import inject_mock_writer

dotenv.load_dotenv()

class TestSessionTimestampOverrides(unittest.TestCase):
    """Test cases for session timestamp override functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.maxim = Maxim()
        self.logger = self.maxim.logger()
        self.mock_writer = inject_mock_writer(self.logger)

    def tearDown(self):
        """Clean up after tests."""
        self.maxim.cleanup()

    def test_session_set_start_timestamp_instance_method(self):
        """Test setting start timestamp using instance method."""
        session_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=2)

        session = self.logger.session(SessionConfig(id=session_id))
        self.assertEqual(session.id, session_id)
        session.set_start_timestamp(custom_start_time)

        self.logger.flush()

        # Find the update log for start timestamp
        update_logs = self.mock_writer.get_logs_by_entity_action("session", "update")
        start_timestamp_logs = [
            log for log in update_logs if log.data and "startTimestamp" in log.data
        ]

        self.assertEqual(len(start_timestamp_logs), 1)
        self.assertEqual(
            start_timestamp_logs[0].data["startTimestamp"], custom_start_time
        )
        self.assertEqual(start_timestamp_logs[0].entity_id, session_id)

    def test_session_set_end_timestamp_instance_method(self):
        """Test setting end timestamp using instance method."""
        session_id = str(uuid4())
        custom_end_time = datetime.now(timezone.utc) - timedelta(hours=1)

        session = self.logger.session(SessionConfig(id=session_id))
        self.assertEqual(session.id, session_id)
        session.set_end_timestamp(custom_end_time)

        self.logger.flush()

        # Find the update log for end timestamp
        update_logs = self.mock_writer.get_logs_by_entity_action("session", "update")
        end_timestamp_logs = [
            log for log in update_logs if log.data and "endTimestamp" in log.data
        ]

        self.assertEqual(len(end_timestamp_logs), 1)
        self.assertEqual(end_timestamp_logs[0].data["endTimestamp"], custom_end_time)
        self.assertEqual(end_timestamp_logs[0].entity_id, session_id)

    def test_session_set_start_timestamp_static_method(self):
        """Test setting start timestamp using static method."""
        session_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=3)

        _session = self.logger.session(SessionConfig(id=session_id))
        Session.set_start_timestamp_(self.logger.writer, session_id, custom_start_time)

        self.logger.flush()

        # Find the update log for start timestamp
        update_logs = self.mock_writer.get_logs_by_entity_action("session", "update")
        start_timestamp_logs = [
            log for log in update_logs if log.data and "startTimestamp" in log.data
        ]

        self.assertEqual(len(start_timestamp_logs), 1)
        self.assertEqual(
            start_timestamp_logs[0].data["startTimestamp"], custom_start_time
        )

    def test_session_set_end_timestamp_static_method(self):
        """Test setting end timestamp using static method."""

        session_id = str(uuid4())
        custom_end_time = datetime.now(timezone.utc) - timedelta(minutes=30)

        _session = self.logger.session(SessionConfig(id=session_id))
        Session.set_end_timestamp_(self.logger.writer, session_id, custom_end_time)

        self.logger.flush()

        # Find the update log for end timestamp
        update_logs = self.mock_writer.get_logs_by_entity_action("session", "update")
        end_timestamp_logs = [
            log for log in update_logs if log.data and "endTimestamp" in log.data
        ]

        self.assertEqual(len(end_timestamp_logs), 1)
        self.assertEqual(end_timestamp_logs[0].data["endTimestamp"], custom_end_time)

    def test_session_end_with_custom_timestamp(self):
        """Test ending session with custom timestamp via end_ method."""

        session_id = str(uuid4())
        custom_end_time = datetime.now(timezone.utc) - timedelta(minutes=15)

        _session = self.logger.session(SessionConfig(id=session_id))
        Session.end_(self.logger.writer, session_id, {"endTimestamp": custom_end_time})

        self.logger.flush()

        # Find the end log
        end_logs = self.mock_writer.get_logs_by_entity_action("session", "end")
        self.assertEqual(len(end_logs), 1)
        self.assertEqual(end_logs[0].data["endTimestamp"], custom_end_time)

    def test_session_end_without_custom_timestamp(self):
        """Test ending session without custom timestamp uses current time."""
        session_id = str(uuid4())
        before_end = datetime.now(timezone.utc)

        _session = self.logger.session(SessionConfig(id=session_id))
        Session.end_(self.logger.writer, session_id)

        after_end = datetime.now(timezone.utc)

        self.logger.flush()

        # Find the end log
        end_logs = self.mock_writer.get_logs_by_entity_action("session", "end")
        self.assertEqual(len(end_logs), 1)
        end_timestamp = end_logs[0].data["endTimestamp"]

        # Verify timestamp is between before and after
        self.assertGreaterEqual(end_timestamp, before_end)
        self.assertLessEqual(end_timestamp, after_end)

    def test_session_start_timestamp_from_config(self):
        """Test setting start timestamp via SessionConfig."""
        session_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=5)

        session = self.logger.session(
            SessionConfig(id=session_id, start_timestamp=custom_start_time)
        )
        self.assertEqual(session.id, session_id)

        self.logger.flush()

        # Verify the session was created with the custom start timestamp
        create_logs = self.mock_writer.get_logs_by_entity_action("session", "create")
        self.assertEqual(len(create_logs), 1)
        # The start timestamp should be in the data
        self.assertIn("startTimestamp", create_logs[0].data)
        self.assertEqual(create_logs[0].data["startTimestamp"], custom_start_time)

    def test_session_start_timestamp_from_config_dict(self):
        """Test setting start timestamp via SessionConfigDict."""

        session_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=5)

        config_dict: SessionConfigDict = {
            "id": session_id,
            "start_timestamp": custom_start_time,
        }
        session = self.logger.session(config_dict)
        self.assertEqual(session.id, session_id)

        self.logger.flush()

        # Verify the session was created with the custom start timestamp
        create_logs = self.mock_writer.get_logs_by_entity_action("session", "create")
        self.assertEqual(len(create_logs), 1)
        # The start timestamp should be in the data
        self.assertIn("startTimestamp", create_logs[0].data)
        self.assertEqual(create_logs[0].data["startTimestamp"], custom_start_time)


class TestTraceTimestampOverrides(unittest.TestCase):
    """Test cases for trace timestamp override functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.maxim = Maxim()
        self.logger = self.maxim.logger()
        self.mock_writer = inject_mock_writer(self.logger)

    def tearDown(self):
        """Clean up after tests."""
        self.maxim.cleanup()

    def test_trace_set_start_timestamp_instance_method(self):
        """Test setting start timestamp using instance method."""
        trace_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=2)

        trace = self.logger.trace(TraceConfig(id=trace_id))
        self.assertEqual(trace.id, trace_id)
        trace.set_start_timestamp(custom_start_time)

        self.logger.flush()

        # Find the update log for start timestamp
        update_logs = self.mock_writer.get_logs_by_entity_action("trace", "update")
        start_timestamp_logs = [
            log for log in update_logs if log.data and "startTimestamp" in log.data
        ]

        self.assertEqual(len(start_timestamp_logs), 1)
        self.assertEqual(
            start_timestamp_logs[0].data["startTimestamp"], custom_start_time
        )
        self.assertEqual(start_timestamp_logs[0].entity_id, trace_id)

    def test_trace_set_end_timestamp_instance_method(self):
        """Test setting end timestamp using instance method."""
        trace_id = str(uuid4())
        custom_end_time = datetime.now(timezone.utc) - timedelta(hours=1)

        trace = self.logger.trace(TraceConfig(id=trace_id))
        self.assertEqual(trace.id, trace_id)
        trace.set_end_timestamp(custom_end_time)

        self.logger.flush()

        # Find the update log for end timestamp
        update_logs = self.mock_writer.get_logs_by_entity_action("trace", "update")
        end_timestamp_logs = [
            log for log in update_logs if log.data and "endTimestamp" in log.data
        ]

        self.assertEqual(len(end_timestamp_logs), 1)
        self.assertEqual(end_timestamp_logs[0].data["endTimestamp"], custom_end_time)
        self.assertEqual(end_timestamp_logs[0].entity_id, trace_id)

    def test_trace_set_start_timestamp_static_method(self):
        """Test setting start timestamp using static method."""

        trace_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=3)

        _trace = self.logger.trace(TraceConfig(id=trace_id))
        Trace.set_start_timestamp_(self.logger.writer, trace_id, custom_start_time)

        self.logger.flush()

        # Find the update log for start timestamp
        update_logs = self.mock_writer.get_logs_by_entity_action("trace", "update")
        start_timestamp_logs = [
            log for log in update_logs if log.data and "startTimestamp" in log.data
        ]

        self.assertEqual(len(start_timestamp_logs), 1)
        self.assertEqual(
            start_timestamp_logs[0].data["startTimestamp"], custom_start_time
        )

    def test_trace_set_end_timestamp_static_method(self):
        """Test setting end timestamp using static method."""

        trace_id = str(uuid4())
        custom_end_time = datetime.now(timezone.utc) - timedelta(minutes=30)

        _trace = self.logger.trace(TraceConfig(id=trace_id))
        Trace.set_end_timestamp_(self.logger.writer, trace_id, custom_end_time)

        self.logger.flush()

        # Find the update log for end timestamp
        update_logs = self.mock_writer.get_logs_by_entity_action("trace", "update")
        end_timestamp_logs = [
            log for log in update_logs if log.data and "endTimestamp" in log.data
        ]

        self.assertEqual(len(end_timestamp_logs), 1)
        self.assertEqual(end_timestamp_logs[0].data["endTimestamp"], custom_end_time)

    def test_trace_end_with_custom_timestamp(self):
        """Test ending trace with custom timestamp via end_ method."""

        trace_id = str(uuid4())
        custom_end_time = datetime.now(timezone.utc) - timedelta(minutes=15)

        _trace = self.logger.trace(TraceConfig(id=trace_id))
        Trace.end_(self.logger.writer, trace_id, {"endTimestamp": custom_end_time})

        self.logger.flush()

        # Find the end log
        end_logs = self.mock_writer.get_logs_by_entity_action("trace", "end")
        self.assertEqual(len(end_logs), 1)
        self.assertEqual(end_logs[0].data["endTimestamp"], custom_end_time)

    def test_trace_end_without_custom_timestamp(self):
        """Test ending trace without custom timestamp uses current time."""

        trace_id = str(uuid4())
        before_end = datetime.now(timezone.utc)

        _trace = self.logger.trace(TraceConfig(id=trace_id))
        Trace.end_(self.logger.writer, trace_id)

        after_end = datetime.now(timezone.utc)

        self.logger.flush()

        # Find the end log
        end_logs = self.mock_writer.get_logs_by_entity_action("trace", "end")
        self.assertEqual(len(end_logs), 1)
        end_timestamp = end_logs[0].data["endTimestamp"]

        # Verify timestamp is between before and after
        self.assertGreaterEqual(end_timestamp, before_end)
        self.assertLessEqual(end_timestamp, after_end)

    def test_trace_start_timestamp_from_config(self):
        """Test setting start timestamp via TraceConfig."""
        trace_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=5)

        trace = self.logger.trace(
            TraceConfig(id=trace_id, start_timestamp=custom_start_time)
        )
        self.assertEqual(trace.id, trace_id)

        self.logger.flush()

        # Verify the trace was created with the custom start timestamp
        create_logs = self.mock_writer.get_logs_by_entity_action("trace", "create")
        self.assertEqual(len(create_logs), 1)
        # The start timestamp should be in the data
        self.assertIn("startTimestamp", create_logs[0].data)
        self.assertEqual(create_logs[0].data["startTimestamp"], custom_start_time)

    def test_trace_start_timestamp_from_config_dict(self):
        """Test setting start timestamp via TraceConfigDict."""

        trace_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=5)

        config_dict: TraceConfigDict = {
            "id": trace_id,
            "start_timestamp": custom_start_time,
        }
        trace = self.logger.trace(config_dict)
        self.assertEqual(trace.id, trace_id)

        self.logger.flush()

        # Verify the trace was created with the custom start timestamp
        create_logs = self.mock_writer.get_logs_by_entity_action("trace", "create")
        self.assertEqual(len(create_logs), 1)
        # The start timestamp should be in the data
        self.assertIn("startTimestamp", create_logs[0].data)
        self.assertEqual(create_logs[0].data["startTimestamp"], custom_start_time)

    def test_trace_set_timestamps_then_end(self):
        """Test setting both timestamps and then ending trace."""
        trace_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=2)
        custom_end_time = datetime.now(timezone.utc) - timedelta(hours=1)

        trace = self.logger.trace(TraceConfig(id=trace_id))
        trace.set_start_timestamp(custom_start_time)
        trace.set_end_timestamp(custom_end_time)
        trace.end()

        self.logger.flush()

        # Verify both timestamps were set
        update_logs = self.mock_writer.get_logs_by_entity_action("trace", "update")
        start_logs = [
            log for log in update_logs if log.data and "startTimestamp" in log.data
        ]
        end_logs = [
            log for log in update_logs if log.data and "endTimestamp" in log.data
        ]

        self.assertEqual(len(start_logs), 1)
        self.assertEqual(len(end_logs), 1)
        self.assertEqual(start_logs[0].data["startTimestamp"], custom_start_time)
        self.assertEqual(end_logs[0].data["endTimestamp"], custom_end_time)

        # Verify end log also has end timestamp
        end_action_logs = self.mock_writer.get_logs_by_entity_action("trace", "end")
        self.assertEqual(len(end_action_logs), 1)
        # The end() method will set its own timestamp, but we already set it via update
        # So we should have both the update and the end action


class TestTimestampOverrideEdgeCases(unittest.TestCase):
    """Test edge cases for timestamp overrides."""

    def setUp(self):
        """Set up test fixtures."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.maxim = Maxim()
        self.logger = self.maxim.logger()
        self.mock_writer = inject_mock_writer(self.logger)

    def tearDown(self):
        """Clean up after tests."""
        self.maxim.cleanup()

    def test_session_multiple_timestamp_updates(self):
        """Test that multiple timestamp updates work correctly."""
        session_id = str(uuid4())
        first_start = datetime.now(timezone.utc) - timedelta(hours=5)
        second_start = datetime.now(timezone.utc) - timedelta(hours=3)

        session = self.logger.session(SessionConfig(id=session_id))
        self.assertEqual(session.id, session_id)
        session.set_start_timestamp(first_start)
        session.set_start_timestamp(second_start)

        self.logger.flush()

        # Should have two update logs
        update_logs = self.mock_writer.get_logs_by_entity_action("session", "update")
        start_timestamp_logs = [
            log for log in update_logs if log.data and "startTimestamp" in log.data
        ]

        self.assertEqual(len(start_timestamp_logs), 2)
        # The last one should be the second timestamp
        self.assertEqual(start_timestamp_logs[-1].data["startTimestamp"], second_start)

    def test_trace_end_overrides_previous_end_timestamp(self):
        """Test that calling end() after set_end_timestamp updates the timestamp."""
        trace_id = str(uuid4())
        custom_end = datetime.now(timezone.utc) - timedelta(hours=1)

        trace = self.logger.trace(TraceConfig(id=trace_id))
        trace.set_end_timestamp(custom_end)

        # Wait a bit and then call end()

        time.sleep(0.1)
        trace.end()

        self.logger.flush()

        # Should have both update and end logs
        update_logs = self.mock_writer.get_logs_by_entity_action("trace", "update")
        end_logs = self.mock_writer.get_logs_by_entity_action("trace", "end")

        # Should have update log with custom end timestamp
        end_update_logs = [
            log for log in update_logs if log.data and "endTimestamp" in log.data
        ]
        self.assertEqual(len(end_update_logs), 1)
        self.assertEqual(end_update_logs[0].data["endTimestamp"], custom_end)

        # Should have end log (which will have its own timestamp)
        self.assertEqual(len(end_logs), 1)


class TestTimestampOverridesRealLogger(unittest.TestCase):
    """Integration test cases for timestamp overrides using real logger."""

    def setUp(self):
        """Set up test fixtures."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.maxim = Maxim()
        self.logger = self.maxim.logger()

    def tearDown(self):
        """Clean up after tests."""
        self.logger.cleanup()
        self.maxim.cleanup()

    def test_session_timestamp_overrides_real_logger(self):
        """Test session timestamp overrides with real logger."""
        session_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=2)
        custom_end_time = datetime.now(timezone.utc) - timedelta(hours=1)

        # Test with SessionConfig
        session = self.logger.session(
            SessionConfig(
                id=session_id,
                name="test_session_timestamp_overrides",
                start_timestamp=custom_start_time,
            )
        )
        self.assertEqual(session.id, session_id)
        self.assertEqual(session.start_timestamp, custom_start_time)

        # Set end timestamp
        session.set_end_timestamp(custom_end_time)
        self.assertEqual(session.end_timestamp, custom_end_time)

        # End session
        session.end()
        self.assertIsNotNone(session.end_timestamp)

    def test_session_timestamp_overrides_dict_config_real_logger(self):
        """Test session timestamp overrides with dict config using real logger."""
        session_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=3)

        config_dict: SessionConfigDict = {
            "id": session_id,
            "name": "test_session_dict_config_timestamp",
            "start_timestamp": custom_start_time,
        }
        session = self.logger.session(config_dict)
        self.assertEqual(session.id, session_id)
        self.assertEqual(session.start_timestamp, custom_start_time)

        # Update start timestamp
        new_start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        session.set_start_timestamp(new_start_time)
        self.assertEqual(session.start_timestamp, new_start_time)

    def test_session_static_methods_real_logger(self):
        """Test session static timestamp methods with real logger."""
        session_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=4)
        custom_end_time = datetime.now(timezone.utc) - timedelta(hours=2)

        session = self.logger.session(
            SessionConfig(id=session_id, name="test_session_static_methods")
        )
        self.assertEqual(session.id, session_id)

        # Use static methods
        Session.set_start_timestamp_(self.logger.writer, session_id, custom_start_time)
        Session.set_end_timestamp_(self.logger.writer, session_id, custom_end_time)

        # End session with custom timestamp
        custom_end_via_end = datetime.now(timezone.utc) - timedelta(minutes=30)
        Session.end_(self.logger.writer, session_id, {"endTimestamp": custom_end_via_end})

    def test_trace_timestamp_overrides_real_logger(self):
        """Test trace timestamp overrides with real logger."""
        trace_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=2)
        custom_end_time = datetime.now(timezone.utc) - timedelta(hours=1)

        # Test with TraceConfig
        trace = self.logger.trace(
            TraceConfig(
                id=trace_id,
                name="test_trace_timestamp_overrides",
                start_timestamp=custom_start_time,
            )
        )
        self.assertEqual(trace.id, trace_id)
        self.assertEqual(trace.start_timestamp, custom_start_time)

        # Set end timestamp
        trace.set_end_timestamp(custom_end_time)
        self.assertEqual(trace.end_timestamp, custom_end_time)

        # Set input and output
        trace.set_input("Test input")
        trace.set_output("Test output")

        # End trace
        trace.end()
        self.assertIsNotNone(trace.end_timestamp)

    def test_trace_timestamp_overrides_dict_config_real_logger(self):
        """Test trace timestamp overrides with dict config using real logger."""
        trace_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=3)

        config_dict: TraceConfigDict = {
            "id": trace_id,
            "name": "test_trace_dict_config_timestamp",
            "start_timestamp": custom_start_time,
        }
        trace = self.logger.trace(config_dict)
        self.assertEqual(trace.id, trace_id)
        self.assertEqual(trace.start_timestamp, custom_start_time)

        # Update start timestamp
        new_start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        trace.set_start_timestamp(new_start_time)
        self.assertEqual(trace.start_timestamp, new_start_time)

    def test_trace_static_methods_real_logger(self):
        """Test trace static timestamp methods with real logger."""
        trace_id = str(uuid4())
        custom_start_time = datetime.now(timezone.utc) - timedelta(hours=4)
        custom_end_time = datetime.now(timezone.utc) - timedelta(hours=2)

        trace = self.logger.trace(
            TraceConfig(id=trace_id, name="test_trace_static_methods")
        )
        self.assertEqual(trace.id, trace_id)

        # Use static methods
        Trace.set_start_timestamp_(self.logger.writer, trace_id, custom_start_time)
        Trace.set_end_timestamp_(self.logger.writer, trace_id, custom_end_time)

        # End trace with custom timestamp
        custom_end_via_end = datetime.now(timezone.utc) - timedelta(minutes=30)
        Trace.end_(self.logger.writer, trace_id, {"endTimestamp": custom_end_via_end})

    def test_session_and_trace_together_real_logger(self):
        """Test session and trace timestamp overrides together with real logger."""
        session_id = str(uuid4())
        trace_id = str(uuid4())
        session_start = datetime.now(timezone.utc) - timedelta(hours=5)
        trace_start = datetime.now(timezone.utc) - timedelta(hours=4)
        trace_end = datetime.now(timezone.utc) - timedelta(hours=3)
        session_end = datetime.now(timezone.utc) - timedelta(hours=2)

        # Create session with custom start timestamp
        session = self.logger.session(
            SessionConfig(
                id=session_id,
                name="test_session_with_trace",
                start_timestamp=session_start,
            )
        )
        self.assertEqual(session.start_timestamp, session_start)

        # Create trace within session with custom start timestamp
        trace = session.trace(
            TraceConfig(
                id=trace_id, name="test_trace_in_session", start_timestamp=trace_start
            )
        )
        self.assertEqual(trace.start_timestamp, trace_start)
        self.assertEqual(trace.id, trace_id)

        # Set trace end timestamp
        trace.set_end_timestamp(trace_end)
        self.assertEqual(trace.end_timestamp, trace_end)

        # End trace
        trace.end()

        # Set session end timestamp
        session.set_end_timestamp(session_end)
        self.assertEqual(session.end_timestamp, session_end)

        # End session
        session.end()

    def test_multiple_timestamp_updates_real_logger(self):
        """Test multiple timestamp updates with real logger."""
        session_id = str(uuid4())
        first_start = datetime.now(timezone.utc) - timedelta(hours=5)
        second_start = datetime.now(timezone.utc) - timedelta(hours=3)
        third_start = datetime.now(timezone.utc) - timedelta(hours=1)

        session = self.logger.session(
            SessionConfig(id=session_id, name="test_multiple_timestamp_updates")
        )
        session.set_start_timestamp(first_start)
        self.assertEqual(session.start_timestamp, first_start)

        session.set_start_timestamp(second_start)
        self.assertEqual(session.start_timestamp, second_start)

        session.set_start_timestamp(third_start)
        self.assertEqual(session.start_timestamp, third_start)

        # Verify final timestamp is the last one set
        self.assertEqual(session.start_timestamp, third_start)


if __name__ == "__main__":
    unittest.main()

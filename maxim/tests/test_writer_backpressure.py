"""
Tests for LogWriter durability and memory bounds.

The writer must never drop a log, and must not grow without bound when the
backend is slow or unreachable. Those two requirements pull against each other:
the resolution is that overflow goes to disk rather than into memory or the
bin.
"""

import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock

from maxim.logger.components.types import CommitLog, Entity
from maxim.logger.writer import (
    MAX_FILE_REPLAY_ATTEMPTS,
    MAX_IN_FLIGHT_BATCHES,
    LogWriter,
    LogWriterConfig,
)


def make_writer(tmpdir, **overrides):
    config = LogWriterConfig(
        base_url="http://localhost",
        api_key="test",
        repository_id="repo-1",
        auto_flush=False,
        flush_interval=10,
        **overrides,
    )
    writer = LogWriter(config)
    writer.maxim_api = MagicMock()
    writer.logs_dir = tmpdir
    os.makedirs(tmpdir, exist_ok=True)
    return writer


def log(n=0):
    return CommitLog(Entity.TRACE, f"t{n}", "create", {"name": f"trace-{n}"})


def spilled_files(tmpdir):
    return sorted(f for f in os.listdir(tmpdir) if f.endswith(".log"))


class WriterTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="maxim-writer-test-")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestSpillFilenames(WriterTestCase):
    def test_two_spills_in_the_same_second_do_not_overwrite(self):
        # The filename used to be second-resolution and opened truncating, so
        # a burst of failures silently destroyed all but the last batch.
        writer = make_writer(self.tmpdir)
        writer.write_to_file([log(1)])
        writer.write_to_file([log(2)])
        self.assertEqual(len(spilled_files(self.tmpdir)), 2)

    def test_spilled_file_contains_every_log(self):
        writer = make_writer(self.tmpdir)
        writer.write_to_file([log(1), log(2), log(3)])
        path = os.path.join(self.tmpdir, spilled_files(self.tmpdir)[0])
        with open(path) as f:
            self.assertEqual(len([line for line in f if line.strip()]), 3)


class TestMemoryBound(WriterTestCase):
    def test_batches_spill_to_disk_once_the_executor_is_saturated(self):
        # The executor's queue is unbounded, so a backend slower than the log
        # rate used to stack batches in memory until the process died.
        writer = make_writer(self.tmpdir)
        release = threading.Event()

        def block(*args, **kwargs):
            release.wait(timeout=10)

        writer.maxim_api.push_logs.side_effect = block
        try:
            for i in range(MAX_IN_FLIGHT_BATCHES + 3):
                writer.commit(log(i))
                writer.flush_commit_logs()
            # Everything past the in-flight limit must be on disk, not queued.
            self.assertGreater(len(spilled_files(self.tmpdir)), 0)
        finally:
            release.set()

    def test_nothing_spills_while_the_backend_keeps_up(self):
        writer = make_writer(self.tmpdir)
        for i in range(5):
            writer.commit(log(i))
            writer.flush_commit_logs()
        deadline = time.time() + 5
        while time.time() < deadline and writer._LogWriter__in_flight > 0:
            time.sleep(0.02)
        self.assertEqual(spilled_files(self.tmpdir), [])

    def test_in_flight_counter_returns_to_zero(self):
        # A counter that leaks upward would permanently force the spill path.
        writer = make_writer(self.tmpdir)
        for i in range(3):
            writer.commit(log(i))
            writer.flush_commit_logs()
        deadline = time.time() + 5
        while time.time() < deadline and writer._LogWriter__in_flight > 0:
            time.sleep(0.02)
        self.assertEqual(writer._LogWriter__in_flight, 0)


class TestReplay(WriterTestCase):
    def test_spilled_logs_are_pushed_and_removed(self):
        writer = make_writer(self.tmpdir)
        writer.write_to_file([log(1)])
        writer.flush_log_files()
        self.assertTrue(writer.maxim_api.push_logs.called)
        self.assertEqual(spilled_files(self.tmpdir), [])

    def test_failing_file_is_retried_then_quarantined(self):
        # A file the server will never accept used to be re-read into memory
        # and retried on every cycle, blocking every file behind it forever.
        writer = make_writer(self.tmpdir)
        writer.maxim_api.push_logs.side_effect = Exception("rejected")
        writer.write_to_file([log(1)])
        for _ in range(MAX_FILE_REPLAY_ATTEMPTS):
            writer.flush_log_files()
        self.assertEqual(spilled_files(self.tmpdir), [])
        self.assertEqual(
            len([f for f in os.listdir(self.tmpdir) if f.endswith(".failed")]), 1
        )

    def test_quarantined_file_is_not_deleted(self):
        # Quarantine must preserve the data for inspection, not discard it.
        writer = make_writer(self.tmpdir)
        writer.maxim_api.push_logs.side_effect = Exception("rejected")
        writer.write_to_file([log(1)])
        for _ in range(MAX_FILE_REPLAY_ATTEMPTS):
            writer.flush_log_files()
        failed = [f for f in os.listdir(self.tmpdir) if f.endswith(".failed")][0]
        with open(os.path.join(self.tmpdir, failed)) as f:
            self.assertIn("trace-1", f.read())

    def test_unreadable_file_is_quarantined_not_retried_forever(self):
        # A file that cannot even be read used to bypass the failure counter
        # entirely, so it was retried on every cycle forever and permanently
        # occupied one of the per-cycle replay slots.
        writer = make_writer(self.tmpdir)
        path = os.path.join(self.tmpdir, "logs-corrupt.log")
        with open(path, "wb") as f:
            f.write(b"\xff\xfe\x80 not valid utf-8")
        for _ in range(MAX_FILE_REPLAY_ATTEMPTS):
            writer.flush_log_files()
        self.assertEqual(spilled_files(self.tmpdir), [])
        self.assertEqual(
            len([f for f in os.listdir(self.tmpdir) if f.endswith(".failed")]), 1
        )
        self.assertFalse(writer.maxim_api.push_logs.called)

    def test_transient_failure_does_not_quarantine(self):
        writer = make_writer(self.tmpdir)
        writer.maxim_api.push_logs.side_effect = Exception("network blip")
        writer.write_to_file([log(1)])
        writer.flush_log_files()
        # Still pending, not quarantined, after a single failure.
        self.assertEqual(len(spilled_files(self.tmpdir)), 1)
        writer.maxim_api.push_logs.side_effect = None
        writer.flush_log_files()
        self.assertEqual(spilled_files(self.tmpdir), [])

    def test_raise_exceptions_propagates_push_failure(self):
        # The outer filesystem guard used to swallow the deliberate re-raise,
        # so raise_exceptions=True never actually surfaced replay failures.
        writer = make_writer(self.tmpdir, raise_exceptions=True)
        cause = Exception("rejected")
        writer.maxim_api.push_logs.side_effect = cause
        writer.write_to_file([log(1)])
        with self.assertRaises(Exception) as ctx:
            writer.flush_log_files()
        self.assertIs(ctx.exception.__cause__, cause)
        # The replay lock must have been released on the way out.
        writer.maxim_api.push_logs.side_effect = None
        writer.flush_log_files()
        self.assertEqual(spilled_files(self.tmpdir), [])

    def test_concurrent_replay_does_not_duplicate_logs(self):
        # flush_log_files now runs on several workers; two of them reading the
        # same file would push those logs twice.
        writer = make_writer(self.tmpdir)
        pushed = []
        lock = threading.Lock()

        def record(repo_id, content):
            time.sleep(0.05)
            with lock:
                pushed.append(content)

        writer.maxim_api.push_logs.side_effect = record
        writer.write_to_file([log(1)])
        threads = [threading.Thread(target=writer.flush_log_files) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(pushed), 1)


if __name__ == "__main__":
    unittest.main()

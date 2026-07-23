"""
Tests for CustomEncoder, the serializer every log payload passes through.

The original "Object of type mappingproxy is not JSON serializable" report was
fixed only in default_json_serializer, which handles model parameters. The
payload path used this encoder instead, so the error survived the fix. These
tests pin the two properties that keep the paths from diverging again:
the scalar conversions match, and payload content is still expanded.
"""

import datetime
import decimal
import enum
import json
import pathlib
import types
import unittest
import uuid
from collections import namedtuple

from pydantic import BaseModel

from maxim.logger.components.types import CommitLog, CustomEncoder, Entity
from maxim.logger.parsers.generation_parser import default_json_serializer


class Weather(BaseModel):
    temp: float
    city: str


class Colour(enum.Enum):
    RED = "red"


def encode(value):
    return json.loads(json.dumps({"v": value}, cls=CustomEncoder))["v"]


class TestScalarConversions(unittest.TestCase):
    """Types with no native JSON form must encode, not raise."""

    def test_mapping_proxy(self):
        self.assertEqual(encode(types.MappingProxyType({"a": 1})), {"a": 1})

    def test_class_object_does_not_manufacture_a_mappingproxy(self):
        # vars() of a class returns a mappingproxy, which used to re-enter
        # default() and raise. This was the actual reported failure.
        self.assertEqual(encode(Weather), {"type": "Weather"})

    def test_pydantic_model_class_does_not_call_unbound_method(self):
        # hasattr(cls, "model_dump") is True, but calling it without an
        # instance raises TypeError: missing 'self'.
        self.assertEqual(encode(Weather), {"type": "Weather"})

    def test_enum(self):
        self.assertEqual(encode(Colour.RED), "red")

    def test_datetime_and_date(self):
        self.assertEqual(encode(datetime.datetime(2026, 7, 21, 9, 0)), "2026-07-21T09:00:00")
        self.assertEqual(encode(datetime.date(2026, 7, 21)), "2026-07-21")

    def test_timedelta(self):
        self.assertEqual(encode(datetime.timedelta(seconds=90)), 90.0)

    def test_decimal(self):
        self.assertEqual(encode(decimal.Decimal("2.5")), 2.5)

    def test_uuid(self):
        self.assertEqual(encode(uuid.UUID(int=7)), "00000000-0000-0000-0000-000000000007")

    def test_path(self):
        self.assertEqual(encode(pathlib.PurePosixPath("/a/b")), "/a/b")

    def test_set(self):
        self.assertEqual(sorted(encode({1, 2})), [1, 2])

    def test_bytes(self):
        self.assertEqual(encode(b"hi"), "hi")

    def test_invalid_utf8_bytes_are_replaced_not_raised(self):
        self.assertIsInstance(encode(b"\xff\xfe"), str)


class TestScalarParityWithModelParameterPath(unittest.TestCase):
    """
    The two serializers may differ on objects, but not on scalars.

    Divergence here is exactly how the mappingproxy bug survived its fix.
    """

    def test_scalar_types_agree_across_both_serializers(self):
        for value in [
            types.MappingProxyType({"a": 1}),
            Colour.RED,
            datetime.datetime(2026, 7, 21, 9, 0),
            datetime.date(2026, 7, 21),
            datetime.timedelta(seconds=90),
            decimal.Decimal("2.5"),
            uuid.UUID(int=7),
            pathlib.PurePosixPath("/a/b"),
            frozenset([1]),
            b"hi",
        ]:
            with self.subTest(value=type(value).__name__):
                self.assertEqual(
                    json.dumps(encode(value), sort_keys=True),
                    json.dumps(default_json_serializer(value), sort_keys=True),
                )


class TestPayloadContentIsStillExpanded(unittest.TestCase):
    """
    CustomEncoder must NOT adopt the descriptor policy used for model
    parameters. It serializes messages and results, where the object is the
    content: collapsing it to a type name would empty the log.
    """

    def test_plain_object_is_expanded(self):
        class AIMessage:
            def __init__(self):
                self.role = "assistant"
                self.content = "Paris"

        self.assertEqual(encode(AIMessage()), {"role": "assistant", "content": "Paris"})

    def test_pydantic_instance_is_expanded(self):
        self.assertEqual(encode(Weather(temp=1.0, city="x")), {"temp": 1.0, "city": "x"})

    def test_namedtuple_is_encoded(self):
        Point = namedtuple("Point", ["x", "y"])
        # json encodes tuples as arrays before default() is ever consulted.
        self.assertEqual(encode(Point(1, 2)), [1, 2])

    def test_object_with_to_dict_is_expanded(self):
        class HasToDict:
            def to_dict(self):
                return {"kind": "to_dict"}

        self.assertEqual(encode(HasToDict()), {"kind": "to_dict"})


class TestCommitLogSerialize(unittest.TestCase):
    def test_mappingproxy_no_longer_drops_the_whole_key(self):
        # Previously CustomEncoder raised, serialize() bisected the payload and
        # deleted the entire modelParameters block - silent data loss that read
        # as "the mappingproxy fix didn't work".
        log = CommitLog(
            Entity.GENERATION,
            "g1",
            "create",
            {
                "model": "gpt-4o",
                "modelParameters": {"response_format": types.MappingProxyType({"a": 1})},
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        wire = log.serialize()
        self.assertIn("modelParameters", wire)
        self.assertIn("messages", wire)

    def test_class_valued_parameter_no_longer_drops_the_whole_key(self):
        log = CommitLog(
            Entity.GENERATION,
            "g2",
            "create",
            {"modelParameters": {"response_format": Weather}, "model": "gpt-4o"},
        )
        self.assertIn("modelParameters", log.serialize())

    def test_serialize_still_survives_a_truly_unserializable_value(self):
        class Exploding:
            def to_dict(self):
                raise RuntimeError("boom")

        log = CommitLog(Entity.GENERATION, "g3", "create", {"bad": Exploding(), "model": "gpt-4o"})
        wire = log.serialize()
        self.assertIn("gpt-4o", wire)


if __name__ == "__main__":
    unittest.main()

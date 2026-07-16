import datetime
import decimal
import enum
import json
import pathlib
import types
import unittest
import uuid

from pydantic import BaseModel

from maxim.logger.parsers.generation_parser import (
    default_json_serializer,
    parse_model_parameters,
)


class Weather(BaseModel):
    temp: float
    city: str


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class HasToDict:
    def to_dict(self):
        return {"kind": "to_dict"}


class HasModelDump:
    def model_dump(self):
        return {"kind": "model_dump"}


class PlainObject:
    def __init__(self):
        self.a = 1
        self.b = "two"


class TestDefaultJsonSerializer(unittest.TestCase):
    """Unit tests for the fallback serializer used by json.dumps."""

    def test_mapping_proxy_becomes_dict(self):
        mp = types.MappingProxyType({"type": "json_schema", "n": 1})
        self.assertEqual(
            default_json_serializer(mp), {"type": "json_schema", "n": 1}
        )

    def test_enum_returns_value(self):
        self.assertEqual(default_json_serializer(Color.RED), "red")

    def test_to_dict_is_preferred(self):
        self.assertEqual(default_json_serializer(HasToDict()), {"kind": "to_dict"})

    def test_model_dump_is_used(self):
        self.assertEqual(
            default_json_serializer(HasModelDump()), {"kind": "model_dump"}
        )

    def test_pydantic_model_class_returns_json_schema(self):
        # A model *class* must not hit the model_dump() branch, which would
        # call an unbound method and raise.
        result = default_json_serializer(Weather)
        self.assertIsInstance(result, dict)
        self.assertEqual(sorted(result["properties"].keys()), ["city", "temp"])

    def test_pydantic_model_instance_still_uses_model_dump(self):
        self.assertEqual(
            default_json_serializer(Weather(temp=1.0, city="x")),
            {"temp": 1.0, "city": "x"},
        )

    def test_plain_class_is_named_not_raised(self):
        result = default_json_serializer(PlainObject)
        self.assertEqual(result, {"type": "PlainObject"})

    def test_class_with_non_callable_schema_attr_falls_through(self):
        # A non-pydantic class may carry an attribute of the same name that is
        # not callable; it must not be invoked.
        class NotAModel:
            model_json_schema = {"not": "callable"}

        self.assertEqual(default_json_serializer(NotAModel), {"type": "NotAModel"})

    def test_set_becomes_list(self):
        self.assertEqual(sorted(default_json_serializer({"a", "b"})), ["a", "b"])

    def test_frozenset_becomes_list(self):
        self.assertEqual(
            sorted(default_json_serializer(frozenset({"a", "b"}))), ["a", "b"]
        )

    def test_bytes_are_decoded(self):
        self.assertEqual(default_json_serializer(b"hello"), "hello")

    def test_bytearray_is_decoded(self):
        self.assertEqual(default_json_serializer(bytearray(b"hi")), "hi")

    def test_invalid_utf8_bytes_are_replaced_not_raised(self):
        # \xff is not valid utf-8; should decode to the Unicode replacement
        # character rather than raising.
        result = default_json_serializer(b"\xff")
        self.assertIsInstance(result, str)
        self.assertEqual(result, "�")

    def test_plain_object_falls_back_to_vars(self):
        self.assertEqual(
            default_json_serializer(PlainObject()), {"a": 1, "b": "two"}
        )

    def test_unserializable_object_raises_type_error(self):
        with self.assertRaises(TypeError):
            # a raw class' __dict__ contains C-level descriptors
            default_json_serializer(object())


class TestStdlibScalarTypes(unittest.TestCase):
    """Common stdlib types that json.dumps cannot serialize natively."""

    def test_datetime_is_isoformat(self):
        self.assertEqual(
            default_json_serializer(datetime.datetime(2026, 7, 16, 12, 0)),
            "2026-07-16T12:00:00",
        )

    def test_date_is_isoformat(self):
        self.assertEqual(
            default_json_serializer(datetime.date(2026, 7, 16)), "2026-07-16"
        )

    def test_time_is_isoformat(self):
        self.assertEqual(default_json_serializer(datetime.time(12, 30)), "12:30:00")

    def test_timedelta_is_seconds(self):
        self.assertEqual(
            default_json_serializer(datetime.timedelta(seconds=30)), 30.0
        )

    def test_decimal_becomes_float(self):
        self.assertEqual(default_json_serializer(decimal.Decimal("1.5")), 1.5)

    def test_uuid_becomes_string(self):
        u = uuid.uuid4()
        self.assertEqual(default_json_serializer(u), str(u))

    def test_path_becomes_string(self):
        self.assertEqual(default_json_serializer(pathlib.Path("/tmp/x")), "/tmp/x")


class TestObjectShapes(unittest.TestCase):
    """Objects whose natural fallback would be lossy or wrong."""

    def test_slotted_object_uses_slots(self):
        class Slotted:
            __slots__ = ("a", "b")

            def __init__(self):
                self.a = 1
                self.b = 2

        # vars() raises on __slots__ objects; must fall back to slot names.
        self.assertEqual(default_json_serializer(Slotted()), {"a": 1, "b": 2})

    def test_string_slots_are_handled(self):
        class OneSlot:
            __slots__ = "a"  # a bare string, not a tuple

            def __init__(self):
                self.a = 1

        self.assertEqual(default_json_serializer(OneSlot()), {"a": 1})

    def test_function_is_named_not_empty_dict(self):
        def a_tool(city: str) -> str:
            return city

        # vars(fn) is {}, which would log misleading empty data.
        self.assertEqual(default_json_serializer(a_tool), {"type": "a_tool"})

    def test_exception_is_described_not_empty_dict(self):
        self.assertEqual(
            default_json_serializer(ValueError("bad")),
            {"type": "ValueError", "message": "bad"},
        )

    def test_numpy_like_scalar_uses_tolist(self):
        # duck-typed so numpy stays an optional dependency
        class FakeScalar:
            def tolist(self):
                return 5

        self.assertEqual(default_json_serializer(FakeScalar()), 5)

    def test_numpy_like_array_uses_tolist(self):
        class FakeArray:
            def tolist(self):
                return [1, 2, 3]

        self.assertEqual(default_json_serializer(FakeArray()), [1, 2, 3])


class TestNotConsumedOrGuessed(unittest.TestCase):
    """Values that must stay skipped rather than be coerced."""

    def test_generator_param_is_not_consumed(self):
        # Consuming a generator to log it would exhaust it and break the
        # caller's real API call. Skipping is the correct behaviour.
        gen = (i for i in range(3))
        parse_model_parameters({"g": gen})
        self.assertEqual(list(gen), [0, 1, 2])

    def test_circular_reference_is_skipped_not_raised(self):
        class Circular:
            def __init__(self):
                self.self_ref = self

        out = parse_model_parameters({"c": Circular(), "model": "gpt-4"})
        self.assertNotIn("c", out)
        self.assertEqual(out["model"], "gpt-4")

    def test_raising_to_dict_is_skipped_not_raised(self):
        class Boom:
            def to_dict(self):
                raise RuntimeError("boom")

        out = parse_model_parameters({"b": Boom(), "model": "gpt-4"})
        self.assertNotIn("b", out)
        self.assertEqual(out["model"], "gpt-4")


class TestParseModelParameters(unittest.TestCase):
    """Unit tests for parse_model_parameters end-to-end behaviour."""

    def test_none_returns_empty_dict(self):
        self.assertEqual(parse_model_parameters(None), {})

    def test_none_values_are_dropped(self):
        out = parse_model_parameters({"a": None, "b": 1})
        self.assertNotIn("a", out)
        self.assertIn("b", out)

    def test_string_values_pass_through_unchanged(self):
        out = parse_model_parameters({"model": "gpt-4"})
        self.assertEqual(out["model"], "gpt-4")

    def test_non_string_values_are_json_stringified(self):
        out = parse_model_parameters({"temperature": 0.7})
        self.assertEqual(out["temperature"], "0.7")

    def test_mapping_proxy_response_format_is_serialized(self):
        # Regression: previously skipped with a warning because a mappingproxy
        # (e.g. response_format passed as a class/proxy) is not JSON serializable.
        mp = types.MappingProxyType(
            {"type": "json_schema", "json_schema": {"name": "resp"}}
        )
        out = parse_model_parameters({"response_format": mp})
        self.assertIn("response_format", out)
        self.assertEqual(
            json.loads(out["response_format"]),
            {"type": "json_schema", "json_schema": {"name": "resp"}},
        )

    def test_pydantic_model_class_response_format_is_serialized(self):
        # Regression: response_format=MyModel is the usual way structured output
        # is declared. vars() on the class yields a mappingproxy, which produced
        # the original "Object of type mappingproxy is not JSON serializable".
        out = parse_model_parameters({"response_format": Weather, "model": "gpt-4"})
        self.assertIn("response_format", out)
        self.assertEqual(
            sorted(json.loads(out["response_format"])["properties"].keys()),
            ["city", "temp"],
        )

    def test_nested_values_are_serialized(self):
        # Nested is the common real shape; the serializer is applied recursively.
        out = parse_model_parameters(
            {
                "meta": {
                    "at": datetime.datetime(2026, 1, 1),
                    "cost": decimal.Decimal("2.5"),
                    "tags": ["a"],
                }
            }
        )
        self.assertEqual(
            json.loads(out["meta"]),
            {"at": "2026-01-01T00:00:00", "cost": 2.5, "tags": ["a"]},
        )

    def test_set_value_is_serialized(self):
        out = parse_model_parameters({"stop": {"a", "b"}})
        self.assertIn("stop", out)
        self.assertEqual(sorted(json.loads(out["stop"])), ["a", "b"])

    def test_bytes_value_is_serialized(self):
        out = parse_model_parameters({"blob": b"hello"})
        self.assertEqual(json.loads(out["blob"]), "hello")

    def test_unserializable_value_is_skipped_not_raised(self):
        # Value that cannot be represented is skipped, other keys survive.
        out = parse_model_parameters({"bad": object(), "model": "gpt-4"})
        self.assertNotIn("bad", out)
        self.assertEqual(out["model"], "gpt-4")


if __name__ == "__main__":
    unittest.main()

"""Regression checks for XML union conversion; optionally test the installed module."""
import importlib.util
import os
from pathlib import Path
import re
import sys
import unittest

if os.environ.get('KUMI_TEST_INSTALLED_PARSER') == '1':
    from mlx_lm.tool_parsers import qwen3_coder as parser
else:
    # The original parser uses only syntax shared by regex and stdlib re.
    # Production still uses MLX-LM's installed regex dependency.
    try:
        import regex
    except ImportError:
        sys.modules['regex'] = re
    spec = importlib.util.spec_from_file_location('patched_qwen3_coder', Path(__file__).with_name('qwen3_coder.py'))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)


VOLUME_DB = {'anyOf': [{'type': 'number', 'maximum': 6}, {'type': 'string', 'const': '-inf'}]}


def convert(text, schema):
    return parser._convert_param_value(text, 'value', {'value': schema})


class UnionConversion(unittest.TestCase):
    def test_kumi_negative_and_decimal_db_remain_numeric(self):
        self.assertEqual(convert('-3', VOLUME_DB), -3)
        self.assertIs(type(convert('-3', VOLUME_DB)), int)
        self.assertEqual(convert('2.5', VOLUME_DB), 2.5)
        self.assertEqual(convert('6', VOLUME_DB), 6)

    def test_silence_sentinel_remains_text(self):
        self.assertEqual(convert('-inf', VOLUME_DB), '-inf')

    def test_invalid_db_remains_text_for_validation(self):
        for text in ('louder', '7', 'true', 'null', 'NaN', 'Infinity', '-Infinity', '1e999', '3dB'):
            with self.subTest(text=text):
                self.assertEqual(convert(text, VOLUME_DB), text)

    def test_ambiguous_general_string_branch_preserves_raw_text(self):
        schema = {'anyOf': [{'type': 'number'}, {'type': 'string'}]}
        self.assertEqual(convert('-3', schema), '-3')
        self.assertEqual(convert('"3"', schema), '"3"')

    def test_string_enum_can_disambiguate_number(self):
        schema = {'anyOf': [{'type': 'integer'}, {'type': 'string', 'enum': ['auto', 'off']}]}
        self.assertEqual(convert('12', schema), 12)
        self.assertEqual(convert('auto', schema), 'auto')

    def test_nullable_integer_and_boolean(self):
        integers = {'anyOf': [{'type': 'integer'}, {'type': 'null'}]}
        booleans = {'oneOf': [{'type': 'boolean'}, {'type': 'null'}]}
        self.assertEqual(convert('3', integers), 3)
        self.assertEqual(convert('3.0', integers), 3.0)
        self.assertEqual(convert('3.5', integers), '3.5')
        self.assertIsNone(convert('null', integers))
        self.assertIs(convert('false', booleans), False)
        self.assertEqual(convert('FALSE', booleans), 'FALSE')

    def test_boolean_is_not_an_integer(self):
        schema = {'anyOf': [{'type': 'integer'}, {'const': 'auto'}]}
        self.assertEqual(convert('true', schema), 'true')
        schema = {'anyOf': [{'type': 'boolean', 'enum': [1]}, {'const': 'auto'}]}
        self.assertEqual(convert('true', schema), 'true')

    def test_type_array_and_ambiguous_nullable_string(self):
        self.assertEqual(convert('2.5', {'type': ['number', 'null']}), 2.5)
        self.assertIsNone(convert('null', {'type': ['number', 'null']}))
        self.assertEqual(convert('null', {'type': ['string', 'null']}), 'null')

    def test_oneof_overlap_is_not_silently_accepted(self):
        schema = {'oneOf': [{'type': 'number'}, {'type': 'integer'}]}
        self.assertEqual(convert('3', schema), '3')
        self.assertEqual(convert('3.5', schema), 3.5)
        schema['anyOf'] = schema.pop('oneOf')
        self.assertEqual(convert('3', schema), 3)

    def test_nested_union_and_bounds(self):
        schema = {'anyOf': [{'allOf': [{'type': 'number'}, {'exclusiveMinimum': 0, 'maximum': 4}]}, {'const': 'auto'}]}
        self.assertEqual(convert('2', schema), 2)
        self.assertEqual(convert('0', schema), '0')
        self.assertEqual(convert('5', schema), '5')

    def test_unknown_assertions_do_not_trigger_uncertain_coercion(self):
        schema = {'anyOf': [{'$ref': '#/$defs/number'}, {'const': 'auto'}]}
        self.assertEqual(convert('3', schema), '3')
        schema = {'anyOf': [{'type': 'number', 'multipleOf': 0.5}, {'const': 'auto'}]}
        self.assertEqual(convert('3', schema), '3')

    def test_union_container_behavior_and_invalid_json(self):
        schema = {'anyOf': [{'type': 'array'}, {'type': 'object'}]}
        self.assertEqual(convert('[1,2]', schema), [1, 2])
        self.assertEqual(convert('{"a":3}', schema), {'a': 3})
        self.assertEqual(convert('{broken}', schema), '{broken}')
        self.assertEqual(convert('[NaN]', schema), '[NaN]')

    def test_parse_actual_kumi_xml_keeps_numeric_db(self):
        tools = [{'type': 'function', 'function': {'name': 'set_mixer', 'parameters': {'type': 'object', 'properties': {'trackRef': {'type': 'string'}, 'volumeDb': VOLUME_DB}}}}]
        result = parser.parse_tool_call('<function=set_mixer>\n<parameter=trackRef>track:2</parameter>\n<parameter=volumeDb>-3</parameter>\n</function>', tools)
        self.assertEqual(result, {'name': 'set_mixer', 'arguments': {'trackRef': 'track:2', 'volumeDb': -3}})

    def test_plain_schemas_and_missing_schema_keep_legacy_behavior(self):
        self.assertEqual(convert('12', {'type': 'integer'}), 12)
        self.assertEqual(convert('2.5', {'type': 'number'}), 2.5)
        self.assertEqual(convert('3', {'type': 'string'}), '3')
        self.assertIs(convert('FALSE', {'type': 'boolean'}), False)
        self.assertIsNone(convert('null', {'type': 'string'}))
        self.assertEqual(convert('{"a":1}', {'description': 'untyped'}), {'a': 1})
        self.assertEqual(convert('{"a":1}', {}), '{"a":1}')
        self.assertEqual(convert('3', {}), '3')
        self.assertEqual(parser._convert_param_value('3', 'absent', {}), '3')
        with self.assertRaises(ValueError):
            convert('1.5', {'type': 'integer'})


if __name__ == '__main__':
    unittest.main()

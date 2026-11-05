# Copyright © 2025 Apple Inc.
# Modified by KumiLLM contributors in 2026: added conservative schema-aware
# scalar conversion for anyOf, oneOf, and type-array unions.

"""
Modified from:
https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/qwen3coder_tool_parser.py
"""

import ast
import json
import math
from typing import Any, Optional

import regex as re

_function_regex = re.compile(r"<function=(.*?)</function>$", re.DOTALL)
_parameter_regex = re.compile(r"<parameter=(.*?)</parameter>", re.DOTALL)
_name_regex = re.compile(r"\s*([^\s<>]+)>?")

_string_types = {"string", "str", "text", "varchar", "char", "enum"}
_bool_types = {"boolean", "bool", "binary"}
_obj_types = {"object", "array", "arr"}


def _scalar_schema_match(value: Any, schema: Any) -> Optional[bool]:
    """Match the scalar schema fragment needed to disambiguate XML text.

    Unknown assertions return None, so an uncertain conversion leaves the
    original text for the caller's full schema validator. This is deliberately
    not a replacement for JSON Schema validation.
    """
    if isinstance(schema, bool):
        return schema
    if not isinstance(schema, dict):
        return None
    known = {
        "type", "const", "enum", "anyOf", "oneOf", "allOf", "not",
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "minLength", "maxLength", "pattern", "title", "description",
        "default", "examples", "$comment", "$schema", "$id",
        "deprecated", "readOnly", "writeOnly",
    }
    uncertain = bool(set(schema) - known)
    actual_type = (
        "null" if value is None else
        "boolean" if isinstance(value, bool) else
        "integer" if isinstance(value, int) else
        "number" if isinstance(value, float) else
        "string" if isinstance(value, str) else None
    )
    if actual_type is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return False
    if "type" in schema:
        types = schema["type"]
        types = types if isinstance(types, list) else [types]
        if not all(isinstance(t, str) and t in {"null", "boolean", "integer", "number", "string", "array", "object"} for t in types):
            return None
        matches = actual_type in types or (
            actual_type == "integer" and "number" in types
        ) or (
            actual_type == "number" and value.is_integer() and "integer" in types
        )
        if not matches:
            return False
    def equal(candidate: Any) -> bool:
        # JSON Schema distinguishes booleans from numbers, unlike Python ==.
        if isinstance(value, bool) != isinstance(candidate, bool):
            return False
        return value == candidate
    if "const" in schema and not equal(schema["const"]):
        return False
    if "enum" in schema:
        if not isinstance(schema["enum"], list):
            return None
        if not any(equal(candidate) for candidate in schema["enum"]):
            return False
    if actual_type in {"integer", "number"}:
        for key, compare in (
            ("minimum", lambda bound: value >= bound),
            ("maximum", lambda bound: value <= bound),
            ("exclusiveMinimum", lambda bound: value > bound),
            ("exclusiveMaximum", lambda bound: value < bound),
        ):
            if key in schema:
                bound = schema[key]
                if isinstance(bound, bool) or not isinstance(bound, (int, float)) or (isinstance(bound, float) and not math.isfinite(bound)):
                    return None
                if not compare(bound):
                    return False
    if actual_type == "string":
        for key, compare in (
            ("minLength", lambda limit: len(value) >= limit),
            ("maxLength", lambda limit: len(value) <= limit),
        ):
            if key in schema:
                limit = schema[key]
                if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                    return None
                if not compare(limit):
                    return False
        if "pattern" in schema:
            try:
                if re.search(schema["pattern"], value) is None:
                    return False
            except (TypeError, re.error):
                return None
    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword not in schema:
            continue
        branches = schema[keyword]
        if not isinstance(branches, list) or not branches:
            return None
        matches = [_scalar_schema_match(value, branch) for branch in branches]
        if keyword == "anyOf":
            result = True if True in matches else None if None in matches else False
        elif keyword == "allOf":
            result = False if False in matches else None if None in matches else True
        else:
            count = matches.count(True)
            result = False if count > 1 else None if None in matches else count == 1
        if result is False:
            return False
        uncertain |= result is None
    if "not" in schema:
        result = _scalar_schema_match(value, schema["not"])
        if result is True:
            return False
        uncertain |= result is None
    return None if uncertain else True


def _convert_union_param_value(param_value: str, schema: dict) -> Any:
    """Decode unambiguous JSON scalars without guessing between string types."""
    def reject_constant(_: str) -> None:
        raise ValueError("NaN and Infinity are not JSON values")
    try:
        value = json.loads(param_value, parse_constant=reject_constant)
    except (ValueError, TypeError):
        return param_value
    # Preserve the existing container decoding, including downstream validation.
    if isinstance(value, (dict, list)):
        return value
    # A bare NaN/Infinity is not JSON null and must not become None.
    if value is None and param_value.strip() != "null":
        return param_value
    if isinstance(value, str):
        return param_value
    # If the raw string itself is allowed (or uncertain), coercion is ambiguous.
    if _scalar_schema_match(param_value, schema) is not False:
        return param_value
    return value if _scalar_schema_match(value, schema) is True else param_value


def _get_arguments_config(func_name: str, tools: Optional[Any]) -> dict:
    """Extract argument configuration for a function."""
    if tools is None:
        return {}
    for tool in tools:
        if not (function := tool.get("function", False)):
            continue
        if function["name"] == func_name:
            if not (params := function.get("parameters", False)):
                return {}
            return params.get("properties", {})
    return {}


def _convert_param_value(param_value: str, param_name: str, param_config: dict) -> Any:
    """Convert parameter value based on its type in the schema."""
    param = param_config.get(param_name, False)
    if isinstance(param, dict) and (
        "anyOf" in param or "oneOf" in param or isinstance(param.get("type"), list)
    ):
        return _convert_union_param_value(param_value, param)
    if param_value.lower() == "null":
        return None

    if not param:
        return param_value

    if "type" in param:
        param_type = str(param["type"]).strip().lower()
    else:
        try:
            value = json.loads(param_value)
        except json.JSONDecodeError:
            return param_value
        return value if isinstance(value, (dict, list)) else param_value
    if param_type in _string_types:
        return param_value
    elif (
        param_type.startswith("int")
        or param_type.startswith("uint")
        or param_type.startswith("long")
        or param_type.startswith("short")
        or param_type.startswith("unsigned")
    ):
        float_param_value = float(param_value)
        int_param_value = int(float_param_value)
        if float_param_value - int_param_value != 0:
            raise ValueError(f"Invalid integer literal {param_value!r}")
        return int_param_value
    elif param_type.startswith("num") or param_type.startswith("float"):
        float_param_value = float(param_value)
        int_param_value = int(float_param_value)
        return (
            float_param_value
            if (float_param_value - int_param_value) != 0
            else int_param_value
        )
    elif param_type in _bool_types:
        return param_value.lower() == "true"
    else:
        if (
            param_type in _obj_types
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            try:
                return json.loads(param_value, strict=False)
            except json.JSONDecodeError:
                pass

        try:
            return ast.literal_eval(param_value)
        except (ValueError, SyntaxError):
            return param_value


def _parse_xml_function_call(function_call_str: str, tools: Optional[Any]):
    name_match = _name_regex.match(function_call_str)
    if name_match is None:
        raise ValueError("No function name provided.")
    function_name = name_match.group(1)
    param_config = _get_arguments_config(function_name, tools)
    parameters = function_call_str[name_match.end() :]
    param_dict = {}
    for match_text in _parameter_regex.findall(parameters):
        param_match = _name_regex.match(match_text)
        if param_match is None:
            continue
        param_name = param_match.group(1)
        param_value = str(match_text[param_match.end() :])
        if param_value.startswith("\n"):
            param_value = param_value[1:]
        if param_value.endswith("\n"):
            param_value = param_value[:-1]

        param_dict[param_name] = _convert_param_value(
            param_value, param_name, param_config
        )
    return dict(name=function_name, arguments=param_dict)


tool_call_start = "<tool_call>"

tool_call_end = "</tool_call>"


def parse_tool_call(
    model_output: str,
    tools: Optional[Any] = None,
):
    match = _function_regex.findall(model_output)
    if not match:
        raise ValueError("No function provided.")
    return _parse_xml_function_call(match[0], tools)

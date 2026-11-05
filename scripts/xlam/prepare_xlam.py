#!/usr/bin/env python3
"""Prepare a bounded, family-disjoint xLAM auxiliary subset. Never trains.

Source execution and generator attribution are upstream claims. This converter
checks structure/types, not actual API execution or musical task competence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import copy
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata


DATASET = 'Salesforce/xlam-function-calling-60k'
REVISION = '26d14ebfe18b1f7b524bd39b404b50af5dc97866'
SOURCE_URL = 'https://huggingface.co/datasets/' + DATASET
VERSION = 'xlam-qwen-auxiliary/1'
SYSTEM = 'Choose the supplied functions that fulfill the user request. Preserve argument types and use only declared parameters.'
NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$')
PARAM = re.compile(r'^[A-Za-z_][A-Za-z0-9_.-]{0,127}$')
OPENAI_NAME = re.compile(r'^[A-Za-z0-9_-]{1,64}$')
ALIASES = {'int':'integer', 'float':'number', 'double':'number', 'str':'string',
           'bool':'boolean', 'list':'array', 'dict':'object',
           'integer':'integer', 'number':'number', 'string':'string',
           'boolean':'boolean', 'array':'array', 'object':'object', 'null':'null'}
ANNOTATIONS = {'description', 'title', 'default', 'examples', 'format'}
SCHEMA_KEYS = ANNOTATIONS | {'type', 'properties', 'required', 'additionalProperties',
    'items', 'enum', 'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum',
    'minLength', 'maxLength', 'minItems', 'maxItems', 'uniqueItems', 'anyOf', 'oneOf'}


class Rejected(ValueError):
    pass


def need(condition, reason):
    if not condition:
        raise Rejected(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict_loads(text):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, 'duplicate_json_key')
            out[key] = value
        return out
    def finite_float(text):
        value = float(text)
        need(math.isfinite(value), 'nonfinite_json_number')
        need(Decimal(str(value)) == Decimal(text), 'lossy_json_float_literal')
        return value
    return json.loads(text, object_pairs_hook=pairs, parse_float=finite_float,
                      parse_constant=lambda _: (_ for _ in ()).throw(Rejected('nonfinite_json_number')))


def decoded(value, expected, reason):
    value = strict_loads(value) if isinstance(value, str) else value
    need(isinstance(value, expected), reason)
    return value


def annotation_parts(text):
    """Split only top-level commas. No evaluation/import of Python type text."""
    depth, start, parts = 0, 0, []
    for index, character in enumerate(text):
        if character == '[': depth += 1
        elif character == ']':
            depth -= 1
            need(depth >= 0, 'malformed_type_annotation')
        elif character == ',' and depth == 0:
            parts.append(text[start:index].strip());start=index+1
    need(depth == 0, 'malformed_type_annotation')
    parts.append(text[start:].strip())
    return parts


def type_expression(text):
    text = text.strip()
    if text.startswith('typing.'):
        text = text[len('typing.'):]
    if text in ALIASES:
        return {'type':ALIASES[text]}
    if text in ('List','Dict'):
        return {'type':'array' if text=='List' else 'object'}
    match = re.fullmatch(r'([A-Za-z]+)\[(.*)\]',text)
    need(match is not None, 'unsupported_parameter_type')
    container, inside = match.groups()
    parts = annotation_parts(inside)
    need(all(parts), 'malformed_type_annotation')
    if container in ('List','list'):
        need(len(parts)==1,'invalid_list_annotation')
        return {'type':'array','items':type_expression(parts[0])}
    if container in ('Union','Optional'):
        if container=='Optional':
            need(len(parts)==1,'invalid_optional_annotation')
            parts.append('null')
        need(len(parts)>=2,'invalid_union_annotation')
        return {'anyOf':[type_expression(p) for p in parts]}
    if container in ('Tuple','tuple'):
        schemas = [type_expression(p) for p in parts]
        need(all(schema==schemas[0] for schema in schemas),'heterogeneous_tuple_not_supported')
        return {'type':'array','items':schemas[0],'minItems':len(parts),'maxItems':len(parts)}
    if container in ('Dict','dict'):
        need(len(parts)==2 and parts[0] in ('str','string'),'nonstring_dict_keys_not_supported')
        return {'type':'object','additionalProperties':type_expression(parts[1])}
    raise Rejected('unsupported_parameter_type')


def parse_type_annotation(text):
    parts = annotation_parts(text)
    optional = False
    for modifier in parts[1:]:
        if modifier.casefold()=='optional': optional=True
        else: need(re.fullmatch(r'default(?:\s*=\s*|\s+).+',modifier,re.I) is not None,
                   'unsupported_type_modifier')
    return type_expression(parts[0]), optional


def normalize_schema(raw, underspecified, path):
    need(isinstance(raw, dict), 'schema_not_object')
    need(set(raw) <= SCHEMA_KEYS, 'unsupported_schema_constraint')
    schema = copy.deepcopy(raw)
    if 'type' in schema:
        source_type = schema['type']
        if isinstance(source_type, str):
            parsed, _ = parse_type_annotation(source_type)
            schema.pop('type')
            for key, value in parsed.items():
                need(key not in schema or schema[key]==value, 'conflicting_type_constraints')
                schema[key] = value
        elif isinstance(source_type, list):
            need(source_type and all(isinstance(t, str) and t in ALIASES for t in source_type), 'unsupported_parameter_union')
            schema['type'] = list(dict.fromkeys(ALIASES[t] for t in source_type))
        else:
            raise Rejected('unsupported_parameter_type')
    elif not schema or set(schema) <= ANNOTATIONS:
        underspecified.append(path + ':untyped_value')
    else:
        need('anyOf' in schema or 'oneOf' in schema, 'missing_parameter_type')
    for key in ('anyOf', 'oneOf'):
        if key in schema:
            need(isinstance(schema[key], list) and schema[key], 'invalid_schema_union')
            schema[key] = [normalize_schema(s, underspecified, path + '/' + key) for s in schema[key]]
    if 'properties' in schema:
        need(isinstance(schema['properties'], dict), 'invalid_schema_properties')
        need(all(isinstance(k, str) for k in schema['properties']), 'invalid_property_name')
        schema['properties'] = {k: normalize_schema(v, underspecified, path + '/' + k)
                                for k, v in schema['properties'].items()}
    if 'required' in schema:
        required = schema['required']
        need(isinstance(required, list) and all(isinstance(k, str) for k in required)
             and len(required) == len(set(required)), 'invalid_required_fields')
        need(set(required) <= schema.get('properties', {} ).keys(), 'required_field_undeclared')
    if 'items' in schema:
        schema['items'] = normalize_schema(schema['items'], underspecified, path + '/items')
    if schema.get('type') == 'array' and 'items' not in schema:
        # Bare lists are genuinely underspecified in the source. Preserve that
        # openness rather than inventing an element type from one answer.
        underspecified.append(path + ':array_items_unspecified')
    if schema.get('type') == 'object' and 'properties' not in schema:
        underspecified.append(path + ':object_properties_unspecified')
    if 'additionalProperties' in schema:
        if isinstance(schema['additionalProperties'], dict):
            schema['additionalProperties'] = normalize_schema(schema['additionalProperties'], underspecified, path + '/additionalProperties')
        else:
            need(type(schema['additionalProperties']) is bool, 'invalid_additional_properties')
    if 'enum' in schema:
        need(isinstance(schema['enum'], list) and schema['enum'], 'invalid_enum')
    for key in ('minLength', 'maxLength', 'minItems', 'maxItems'):
        if key in schema:
            need(type(schema[key]) is int and schema[key] >= 0, 'invalid_schema_bound')
    for key in ('minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum'):
        if key in schema:
            need(type(schema[key]) in (int, float) and math.isfinite(schema[key]), 'invalid_schema_bound')
    if 'uniqueItems' in schema:
        need(type(schema['uniqueItems']) is bool, 'invalid_unique_items')
    return schema


def function_parameters(raw, underspecified, source_annotations=None):
    raw = decoded(raw, dict, 'parameters_not_object')
    # A parameter actually named "type" has a dictionary definition. It must
    # not be mistaken for the standard JSON Schema top-level type field.
    canonical_schema = isinstance(raw.get('type'), (str, list)) or (
        'properties' in raw and isinstance(raw.get('properties'), dict)
        and all(isinstance(v,dict) for v in raw['properties'].values())
        and (isinstance(raw.get('required'), list) or set(raw) <= SCHEMA_KEYS))
    if canonical_schema:
        schema = normalize_schema(raw, underspecified, 'parameters')
        need(schema.get('type') == 'object', 'function_parameters_must_be_object')
        need(isinstance(schema.get('properties', {}), dict), 'invalid_function_properties')
        schema.setdefault('properties', {})
        need(all(PARAM.fullmatch(k) for k in schema['properties']), 'unsafe_parameter_name')
        # Reject undocumented function keyword arguments even if a schema would
        # otherwise be open. We do not add or guess argument names.
        schema['additionalProperties'] = False
        return schema
    properties, required = {}, []
    for name, definition in raw.items():
        need(isinstance(name, str) and PARAM.fullmatch(name), 'unsafe_parameter_name')
        need(isinstance(definition, dict), 'parameter_definition_not_object')
        definition = copy.deepcopy(definition)
        explicit_required = 'required' in definition
        optional = False
        if isinstance(definition.get('type'),str):
            _, optional = parse_type_annotation(definition['type'])
        # The official release omits required everywhere and includes unreliable
        # default placeholders. This is our conservative projection, not a claim
        # to recover the executable source API signature.
        flag = definition.pop('required', not optional)
        need(type(flag) is bool, 'parameter_required_not_boolean')
        annotation = {'parameter':name,'raw_type':definition.get('type'),
                      'explicit_required_present':explicit_required,
                      'projected_required':flag,'default_present':'default' in definition}
        if 'default' in definition:
            annotation['raw_default'] = definition.pop('default')
        if source_annotations is not None:
            source_annotations.append(annotation)
        if flag:
            required.append(name)
        properties[name] = normalize_schema(definition, underspecified, 'parameters/' + name)
    return {'type':'object', 'properties':properties, 'required':required, 'additionalProperties':False}


def json_type(value):
    if value is None: return 'null'
    if type(value) is bool: return 'boolean'
    if type(value) is int: return 'integer'
    if type(value) is float:
        need(math.isfinite(value), 'nonfinite_argument')
        return 'number'
    if isinstance(value, str): return 'string'
    if isinstance(value, list): return 'array'
    if isinstance(value, dict): return 'object'
    raise Rejected('non_json_argument')


def equality_key(value):
    kind = json_type(value)
    if kind in ('integer','number'):
        return ('number',Decimal(str(value)))
    if kind=='array':
        return ('array',tuple(equality_key(v) for v in value))
    if kind=='object':
        return ('object',tuple(sorted((k,equality_key(v)) for k,v in value.items())))
    return (kind,value)


def validate(value, schema):
    actual = json_type(value)
    allowed = schema.get('type')
    if allowed is not None:
        allowed = allowed if isinstance(allowed, list) else [allowed]
        need(actual in allowed or actual == 'integer' and 'number' in allowed, 'argument_type_mismatch')
    for key in ('anyOf', 'oneOf'):
        if key in schema:
            passes = 0
            for branch in schema[key]:
                try:
                    validate(value, branch)
                    passes += 1
                except Rejected:
                    pass
            need(passes >= 1 if key == 'anyOf' else passes == 1, 'argument_union_mismatch')
    if 'enum' in schema:
        need(equality_key(value) in {equality_key(x) for x in schema['enum']}, 'argument_enum_mismatch')
    if actual in ('number', 'integer'):
        for key, test in [('minimum',lambda v,n:v>=n), ('maximum',lambda v,n:v<=n),
                          ('exclusiveMinimum',lambda v,n:v>n), ('exclusiveMaximum',lambda v,n:v<n)]:
            need(key not in schema or test(value,schema[key]), 'argument_' + key)
    if actual in ('string', 'array'):
        for key, test in [('minLength',lambda n,b:n>=b), ('maxLength',lambda n,b:n<=b)] if actual=='string' else [('minItems',lambda n,b:n>=b),('maxItems',lambda n,b:n<=b)]:
            need(key not in schema or test(len(value),schema[key]), 'argument_' + key)
    if actual == 'array':
        if schema.get('uniqueItems'):
            need(len({equality_key(x) for x in value}) == len(value), 'argument_items_not_unique')
        for item in value:
            validate(item, schema.get('items', {}))
    if actual == 'object':
        properties = schema.get('properties', {})
        need(set(schema.get('required', [])) <= value.keys(), 'required_argument_missing')
        for key, item in value.items():
            need(isinstance(key,str), 'argument_object_key_not_string')
            if key in properties:
                validate(item,properties[key])
            else:
                additional = schema.get('additionalProperties', True)
                need(additional is not False, 'undeclared_argument')
                validate(item, additional if isinstance(additional,dict) else {})


def normalized_query(query):
    # Lexical deduplication only. Do not claim semantic paraphrase detection.
    return ' '.join(unicodedata.normalize('NFKC', query).casefold().split())


def family(name):
    return name.rsplit('.', 1)[0] if '.' in name else name


def alias(name):
    # Hash every original name, including already-safe names, so an existing
    # source name cannot impersonate an alias of a different dotted name.
    stem = re.sub(r'[^A-Za-z0-9_]', '_', name)[:43]
    return 'xlam_' + stem + '_' + hashlib.sha256(name.encode()).hexdigest()[:10]


def argument_types(value):
    kinds = {json_type(value)}
    if isinstance(value, dict):
        for item in value.values(): kinds |= argument_types(item)
    elif isinstance(value, list):
        for item in value: kinds |= argument_types(item)
    return kinds


def convert_row(raw, index, source_sha256, revision=REVISION, alias_names=False):
    need(isinstance(raw, dict), 'row_not_object')
    query = raw.get('query')
    need(isinstance(query,str) and query.strip(), 'query_not_nonempty_text')
    raw_tools = decoded(raw.get('tools'),list,'tools_not_array')
    answers = decoded(raw.get('answers'),list,'answers_not_array')
    need(raw_tools and answers, 'empty_tools_or_answers')
    names, emitted_names, name_map, tools, schemas, underspecified = set(), set(), {}, [], {}, []
    parameter_annotations = {}
    for raw_tool in raw_tools:
        need(isinstance(raw_tool,dict) and {'name','parameters'} <= raw_tool.keys(), 'invalid_tool_definition')
        need(set(raw_tool) <= {'name','description','parameters'}, 'unsupported_tool_definition_fields')
        name = raw_tool['name']
        need(isinstance(name,str) and NAME.fullmatch(name), 'unsafe_function_name')
        need(name not in names, 'duplicate_tool_name')
        names.add(name)
        emitted = alias(name) if alias_names else name
        need(emitted not in emitted_names, 'function_alias_collision')
        emitted_names.add(emitted)
        name_map[name] = emitted
        description = raw_tool.get('description','')
        need(isinstance(description,str), 'tool_description_not_text')
        annotations = []
        schema = function_parameters(raw_tool['parameters'], underspecified, annotations)
        parameter_annotations[name] = annotations
        schemas[name] = schema
        tools.append({'type':'function','function':{'name':emitted,'description':description,'parameters':schema}})
    calls, types = [], set()
    row_sha = digest(raw)
    for call_index, answer in enumerate(answers):
        need(isinstance(answer,dict) and set(answer) == {'name','arguments'}, 'unsupported_answer_fields')
        name = answer['name']
        need(isinstance(name,str) and name in names, 'answer_function_undeclared')
        arguments = decoded(answer['arguments'],dict,'arguments_not_object')
        validate(arguments, schemas[name])
        for value in arguments.values():
            types |= argument_types(value)
        calls.append({'id':'xlam_' + row_sha[:16] + '_' + str(call_index),'type':'function',
                      'function':{'name':name_map[name],'arguments':copy.deepcopy(arguments)}})
    source_id = raw.get('id', 'row-' + row_sha[:20])
    need(isinstance(source_id,(str,int)) and type(source_id) is not bool, 'unsupported_source_id')
    numeric_id = int(source_id) if type(source_id) is int or isinstance(source_id,str) and source_id.isdecimal() else -1
    generator = 'DeepSeek-V2-Chat' if 0 <= numeric_id <= 33658 else 'Mixtral-8x22B-Instruct-v0.1' if 33659 <= numeric_id < 60000 else 'unknown'
    families = sorted({family(name) for name in names})
    category = 'single' if len(calls)==1 else 'parallel_same_function' if len({c['function']['name'] for c in calls})==1 else 'parallel_multiple_functions'
    metadata = {'episode_id':'xlam-' + row_sha[:20], 'source_dataset':DATASET,
        'source_revision':revision, 'source_file_sha256':source_sha256,
        'source_row_id':source_id, 'source_row_index':index, 'source_row_canonical_sha256':row_sha,
        'source_url':SOURCE_URL, 'license':'CC-BY-4.0',
        'license_url':'https://creativecommons.org/licenses/by/4.0/',
        'attribution':'Salesforce AI Research; Zuxin Liu et al., APIGen (2024), arXiv:2406.18518.',
        'upstream_generator_reported':generator,
        'generator_identity_independently_verified':False,
        'upstream_execution_verification_reported':True, 'locally_executed':False,
        'collection_kind':'external_generic_function_calling_auxiliary',
        'real_daw_verified':False, 'human_preference_verified':False,
        'conversion_version':VERSION, 'source_api_families':families,
        'project_family_id':'xlam-families-' + digest(families)[:20],
        'function_name_map':name_map,
        'raw_tools_canonical_sha256':digest(raw_tools),
        'source_parameter_annotations':parameter_annotations,
        'required_projection_policy':'Explicit source required flags retained. Otherwise only a textual optional modifier permits omission; defaults do not. This is a conservative projection, not recovered backend signature semantics.',
        'default_projection_policy':'Raw defaults retained in metadata but omitted from active parameter-map schemas; no defaults are inserted into answers.',
        'argument_type_policy':'strict_no_value_coercion_integer_distinct_from_boolean',
        'schema_policy':'documented_function_keywords_only; no inferred array item types; format is annotation',
        'underspecified_containers_or_values':sorted(set(underspecified)),
        'query_style':category, 'available_tool_count':len(tools), 'answer_call_count':len(calls),
        'argument_types':sorted(types), 'mask_all_history_required':True,
        'train_on_final_assistant_only_required':True,
        'loss_contract_required':'strict_final_target_excluding_padding',
        'tokenizer_verified':False, 'ready_for_training_ingestion':False}
    record = {'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':query},
                           {'role':'assistant','content':None,'tool_calls':calls}],
              'tools':tools,'metadata':metadata}
    return record


def source_split(families, seed, valid_fraction):
    sides = set()
    for name in families:
        fraction = int(hashlib.sha256((seed + '\0' + name).encode()).hexdigest()[:16],16) / 2**64
        sides.add('valid' if fraction < valid_fraction else 'train')
    return next(iter(sides)) if len(sides)==1 else None


def balanced_select(records, limit, seed):
    buckets = defaultdict(list)
    for record in records:
        m = record['metadata']
        key = (m['query_style'], m['available_tool_count'] > 1, tuple(m['argument_types']))
        buckets[key].append(record)
    pools = {}
    for key, values in buckets.items():
        values.sort(key=lambda r: digest([seed,r['metadata']['source_row_canonical_sha256']]))
        pools[key] = deque(values)
    keys = sorted(pools, key=lambda k:digest([seed,k]))
    selected, covered = [], set()
    # A first pass improves family coverage without losing round-robin type and
    # call-style balance. The second pass fills remaining capacity.
    while len(selected) < limit and any(pools.values()):
        for key in keys:
            if len(selected) >= limit:break
            pool = pools[key]
            if not pool:continue
            best_index = next((i for i,r in enumerate(pool)
                               if set(r['metadata']['source_api_families']) - covered),0)
            pool.rotate(-best_index);record=pool.popleft();pool.rotate(best_index)
            selected.append(record);covered.update(record['metadata']['source_api_families'])
    return selected


def prepare(rows, source_sha256, train_size=256, valid_size=64, seed='kumi-xlam-aux-v1',
            valid_fraction=0.2, revision=REVISION, alias_names=False):
    need(isinstance(rows,list), 'source_root_must_be_array')
    need(0 <= train_size <= 2048 and 0 <= valid_size <= 256, 'subset_size_outside_bound')
    need(0 < valid_fraction < 1, 'invalid_validation_fraction')
    converted, rejected, duplicates = [], [], []
    for index, raw in enumerate(rows):
        try:
            converted.append(convert_row(raw,index,source_sha256,revision,alias_names))
        except (Rejected,ValueError,TypeError,KeyError,OverflowError) as error:
            rejected.append({'source_row_index':index,'source_row_id':raw.get('id') if isinstance(raw,dict) else None,
                             'reason':str(error) if isinstance(error,Rejected) else type(error).__name__})
    # Sort before deduplication so source input order cannot choose a different
    # example from a lexical duplicate group. IDs/index remain exact provenance.
    converted.sort(key=lambda r:r['metadata']['source_row_canonical_sha256'])
    global_aliases = {}
    for record in converted:
        for original, emitted in record['metadata']['function_name_map'].items():
            need(emitted not in global_aliases or global_aliases[emitted]==original, 'global_function_alias_collision')
            global_aliases[emitted] = original
    seen_queries, seen_query_schema = set(), set()
    pools, purged = {'train':[],'valid':[]}, []
    for record in converted:
        query_key = normalized_query(record['messages'][1]['content'])
        schema_key = digest([query_key, record['tools']])
        if query_key in seen_queries or schema_key in seen_query_schema:
            duplicates.append({'source_row_id':record['metadata']['source_row_id'],
                               'reason':'normalized_query_duplicate' if query_key in seen_queries else 'query_schema_duplicate'})
            continue
        seen_queries.add(query_key);seen_query_schema.add(schema_key)
        split = source_split(record['metadata']['source_api_families'],seed,valid_fraction)
        if split is None:
            purged.append({'source_row_id':record['metadata']['source_row_id'],'reason':'offered_api_families_cross_split'})
            continue
        record['metadata']['source_group_split'] = split
        pools[split].append(record)
    selected = {'train':balanced_select(pools['train'],train_size,seed),
                'valid':balanced_select(pools['valid'],valid_size,seed)}
    families = {side:{f for r in rs for f in r['metadata']['source_api_families']} for side,rs in selected.items()}
    need(not families['train'] & families['valid'], 'api_family_split_leak')
    emitted = {side:{t['function']['name'] for r in rs for t in r['tools']} for side,rs in selected.items()}
    need(not emitted['train'] & emitted['valid'], 'emitted_function_name_split_leak')
    report = {'schema':VERSION,'dataset':DATASET,'source_url':SOURCE_URL,'source_revision':revision,
        'source_file_sha256':source_sha256,'license':'CC-BY-4.0',
        'attribution':'Salesforce AI Research; Zuxin Liu et al., APIGen (2024), arXiv:2406.18518.',
        'source_rows':len(rows),'format_and_type_valid_rows':len(converted),
        'rejected_rows':len(rejected),'rejection_counts':dict(sorted(Counter(x['reason'] for x in rejected).items())),
        'lexical_duplicates_removed':len(duplicates),'cross_split_rows_purged':len(purged),
        'candidate_pool_counts':{k:len(v) for k,v in pools.items()},
        'selected_counts':{k:len(v) for k,v in selected.items()},
        'selected_query_styles':{k:dict(Counter(r['metadata']['query_style'] for r in v)) for k,v in selected.items()},
        'selected_argument_types':{k:dict(Counter(t for r in v for t in r['metadata']['argument_types'])) for k,v in selected.items()},
        'selected_api_families':{k:sorted(v) for k,v in families.items()},'family_overlap':[],
        'selected_underspecified_records':{k:sum(bool(r['metadata']['underspecified_containers_or_values']) for r in v) for k,v in selected.items()},
        'required_projection_policy':'Only explicit optional text permits omission when source required flags are absent; ambiguous omitted nonoptional parameters are rejected. This is not claimed as exact upstream API signature recovery.',
        'default_projection_policy':'Omit source default annotations from active parameter-map schemas; retain exact raw values in source_parameter_annotations metadata. Never insert defaults into arguments.',
        'seed':seed,'valid_family_fraction':valid_fraction,
        'split_policy':'Hash every offered API namespace/family before selection; purge mixed-family rows. Bare function names are their own families.',
        'dedup_policy':'NFKC, casefold and whitespace normalized query; no claim of semantic paraphrase deduplication.',
        'selection_policy':'Stable round-robin call style/tool choice/argument-type buckets with preference for new API families.',
        'function_name_policy':'reversible_OpenAI_safe_aliases' if alias_names else 'preserved_safe_ASCII_namespaces',
        'locally_executed':False,'execution_result_messages_added':0,'private_reasoning_imported':False,
        'current_kumi_training_modified':False,'tokenizer_verified':False,
        'ready_for_training_ingestion':False,
        'required_loss_contract':'strict_final_target_excluding_padding'}
    return selected, report, {'rejected':rejected,'duplicates':duplicates,'purged':purged}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-sha256')
    parser.add_argument('--revision',default=REVISION)
    parser.add_argument('--train-size',type=int,default=256)
    parser.add_argument('--valid-size',type=int,default=64)
    parser.add_argument('--valid-fraction',type=float,default=0.2)
    parser.add_argument('--seed',default='kumi-xlam-aux-v1')
    parser.add_argument('--alias-tool-names',action='store_true')
    args = parser.parse_args()
    need(args.input.is_file() and args.input.stat().st_size <= 128_000_000,'source_file_missing_or_oversized')
    need(not args.output.exists() or not any(args.output.iterdir()),'fresh_output_required')
    raw = args.input.read_bytes();source_sha = hashlib.sha256(raw).hexdigest()
    need(args.source_sha256 is None or args.source_sha256 == source_sha,'source_file_hash_mismatch')
    rows = strict_loads(raw)
    selected, report, exclusions = prepare(rows,source_sha,args.train_size,args.valid_size,args.seed,
                                           args.valid_fraction,args.revision,args.alias_tool_names)
    args.output.mkdir(parents=True,exist_ok=True)
    for split, records in selected.items():
        data = ''.join(canonical(r)+'\n' for r in records).encode()
        (args.output/(split+'.jsonl')).write_bytes(data)
        report.setdefault('files',{})[split+'.jsonl']={'records':len(records),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    (args.output/'test.jsonl').write_text('')
    report['converter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'exclusions.json').write_text(json.dumps(exclusions,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('source_rows','format_and_type_valid_rows','selected_counts','rejection_counts','candidate_pool_counts','ready_for_training_ingestion')}))


if __name__ == '__main__':
    main()

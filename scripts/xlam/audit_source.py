#!/usr/bin/env python3
"""Report actual source metadata coverage without printing queries/answers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from prepare_xlam import strict_loads, decoded, parse_type_annotation, validate, Rejected


def audit(path):
    raw=path.read_bytes();rows=strict_loads(raw)
    types=Counter();keys=Counter();defaults=Counter();mismatches=Counter()
    names=set();parameter_definitions=0;explicit_required=0;optional=0
    for row in rows:
        for tool in decoded(row['tools'],list,'tools_not_array'):
            names.add(tool['name'])
            for name,definition in tool['parameters'].items():
                parameter_definitions+=1
                types[definition.get('type','<missing>')]+=1
                keys['|'.join(sorted(definition))]+=1
                explicit_required+='required' in definition
                try:
                    schema,omissible=parse_type_annotation(definition['type'])
                    optional+=omissible
                except Rejected:
                    schema=None
                if 'default' not in definition:
                    defaults['missing']+=1;continue
                value=definition['default']
                defaults['present']+=1
                defaults['json_type_'+type(value).__name__]+=1
                if value=='':defaults['empty_string_placeholder_or_default']+=1
                if schema is None:
                    defaults['type_not_supported_for_comparison']+=1;continue
                try:validate(value,schema)
                except Rejected as error:
                    defaults['does_not_match_declared_value_type_or_constraints']+=1
                    mismatches[str(error)]+=1
                else:defaults['matches_declared_value_type_and_constraints']+=1
    return {'schema':'xlam-actual-source-audit/1','source_rows':len(rows),
        'source_file_bytes':len(raw),'source_file_sha256':hashlib.sha256(raw).hexdigest(),
        'distinct_offered_function_names':len(names),'parameter_definition_occurrences':parameter_definitions,
        'explicit_required_field_occurrences':explicit_required,
        'textually_optional_parameter_occurrences':optional,
        'parameter_type_annotation_occurrences':dict(sorted(types.items())),
        'parameter_definition_key_shapes':dict(sorted(keys.items())),
        'default_annotation_occurrences':dict(sorted(defaults.items())),
        'default_mismatch_reasons':dict(sorted(mismatches.items())),
        'semantics':'Counts are occurrences across all rows, not unique API signatures. A matching default type does not prove it is an executable backend default. Empty strings can be real defaults or placeholders; no distinction is assumed.',
        'requiredness_policy':'For the auxiliary subset only, absent required is projected to required except an explicit textual optional modifier. This conservative projection may reject legitimate upstream default omission; exact backend signatures are not recovered.',
        'default_policy':'Raw defaults are retained in selected record metadata but excluded from active parameter-map schemas. Answers are never filled or coerced.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.input)
    if args.output.exists():raise ValueError('fresh_output_required')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['source_rows','distinct_offered_function_names','parameter_definition_occurrences','explicit_required_field_occurrences','default_annotation_occurrences']}))


if __name__=='__main__':main()

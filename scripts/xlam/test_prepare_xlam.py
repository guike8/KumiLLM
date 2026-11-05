"""Original synthetic fixtures; none are xLAM source rows or training data."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import prepare_xlam as x


def row(i=1, name='calc.add', query='Add the two quantities.', value=3):
    return {'id':i,'query':query,'tools':json.dumps([{'name':name,'description':'Original test arithmetic.',
        'parameters':{'n':{'type':'int','required':True},'options':{'type':'dict, optional'}}}]),
        'answers':json.dumps([{'name':name,'arguments':{'n':value}}])}


class ConversionTests(unittest.TestCase):
    def test_flat_schema_aliases_and_no_fabricated_results(self):
        record=x.convert_row(row(),0,'a'*64)
        schema=record['tools'][0]['function']['parameters']
        self.assertEqual(schema['properties']['n']['type'],'integer')
        self.assertEqual(schema['required'],['n'])
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual([m['role'] for m in record['messages']],['system','user','assistant'])
        self.assertIsInstance(record['messages'][-1]['tool_calls'][0]['function']['arguments'],dict)
        self.assertFalse(record['metadata']['locally_executed'])
        self.assertFalse(record['metadata']['ready_for_training_ingestion'])
        self.assertTrue(record['metadata']['underspecified_containers_or_values'])

    def test_scalar_types_never_coerced(self):
        for value in (True,'3',3.0,None):
            with self.assertRaises(x.Rejected):x.convert_row(row(value=value),0,'a'*64)

    def test_required_unknown_and_undeclared_function_rejected(self):
        for answer in ({'name':'calc.add','arguments':{}},
                       {'name':'calc.add','arguments':{'n':3,'surprise':2}},
                       {'name':'missing','arguments':{'n':3}}):
            source=row();source['answers']=[answer]
            with self.assertRaises(x.Rejected):x.convert_row(source,0,'a'*64)

    def test_duplicate_json_nonfinite_and_unknown_constraints_rejected(self):
        for text in ('{"n":1,"n":2}','{"n":NaN}','{"n":Infinity}','{"n":1e400}','{"n":9007199254740993.0}'):
            with self.assertRaises(x.Rejected):x.strict_loads(text)
        source=row();tools=json.loads(source['tools']);tools[0]['parameters']['n']['not']={'const':5};source['tools']=tools
        with self.assertRaises(x.Rejected):x.convert_row(source,0,'a'*64)

    def test_bare_list_remains_open_without_guessed_item_type(self):
        source=row();source['tools']=[{'name':'collect','parameters':{'values':{'type':'list','required':True}}}]
        source['answers']=[{'name':'collect','arguments':{'values':[1,'a',False,{'k':2}]}}]
        record=x.convert_row(source,0,'a'*64)
        self.assertNotIn('items',record['tools'][0]['function']['parameters']['properties']['values'])
        self.assertEqual(record['messages'][-1]['tool_calls'][0]['function']['arguments']['values'][1],'a')

    def test_type_summary_describes_argument_values_not_call_packet(self):
        result=x.convert_row(row(),0,'a'*64)
        self.assertEqual(result['metadata']['argument_types'],['integer'])

    def test_nested_schema_bounds_and_bool_type(self):
        source=row();source['tools']=[{'name':'nested','parameters':{'type':'object','properties':{
            'items':{'type':'array','items':{'type':'object','properties':{'active':{'type':'boolean'}},'required':['active'],'additionalProperties':False},'minItems':1}},'required':['items']}}]
        source['answers']=[{'name':'nested','arguments':{'items':[{'active':True}]}}]
        x.convert_row(source,0,'a'*64)
        source['answers'][0]['arguments']['items'][0]['active']=1
        with self.assertRaises(x.Rejected):x.convert_row(source,0,'a'*64)

    def test_parallel_answers_stay_one_assistant_and_aliases_reversible(self):
        source=row();source['answers']=json.loads(source['answers'])*2
        record=x.convert_row(source,0,'a'*64,alias_names=True)
        calls=record['messages'][-1]['tool_calls'];self.assertEqual(len(calls),2)
        self.assertNotEqual(calls[0]['id'],calls[1]['id'])
        self.assertEqual(calls[0]['function']['name'],record['metadata']['function_name_map']['calc.add'])
        self.assertTrue(x.OPENAI_NAME.fullmatch(calls[0]['function']['name']))
        self.assertEqual(record['metadata']['query_style'],'parallel_same_function')

    def test_alias_collision_or_duplicate_tool_names_rejected(self):
        source=row();source['tools']=json.loads(source['tools'])*2
        with self.assertRaises(x.Rejected):x.convert_row(source,0,'a'*64,alias_names=True)

    def test_unsafe_names_and_private_answer_fields_rejected(self):
        with self.assertRaises(x.Rejected):x.convert_row(row(name='bad>tool'),0,'a'*64)
        source=row();source['answers']=json.loads(source['answers']);source['answers'][0]['thought']='not imported'
        with self.assertRaises(x.Rejected):x.convert_row(source,0,'a'*64)

    def test_parameter_named_type_is_not_mistaken_for_schema(self):
        source=row();source['tools']=[{'name':'kind','parameters':{'type':{'type':'str','required':True}}}]
        source['answers']=[{'name':'kind','arguments':{'type':'piano'}}]
        result=x.convert_row(source,0,'a'*64)
        self.assertEqual(result['tools'][0]['function']['parameters']['properties']['type']['type'],'string')

    def test_parameter_named_properties_is_not_mistaken_for_schema(self):
        source=row();source['tools']=[{'name':'kind','parameters':{'properties':{'type':'dict','required':True}}}]
        source['answers']=[{'name':'kind','arguments':{'properties':{'label':'piano'}}}]
        result=x.convert_row(source,0,'a'*64)
        self.assertEqual(result['tools'][0]['function']['parameters']['properties']['properties']['type'],'object')

    def test_typed_containers_and_optional_modifier(self):
        source=row();source['tools']=[{'name':'typed','parameters':{
            'pairs':{'type':'List[Tuple[float, float]]'},
            'values':{'type':'List[Union[int, float]], optional','default':'not an array'}}}]
        source['answers']=[{'name':'typed','arguments':{'pairs':[[1,2.5],[3.0,4.5]]}}]
        result=x.convert_row(source,0,'a'*64)
        schema=result['tools'][0]['function']['parameters']
        self.assertEqual(schema['required'],['pairs'])
        self.assertNotIn('default',schema['properties']['values'])
        self.assertEqual(result['metadata']['source_parameter_annotations']['typed'][1]['raw_default'],'not an array')
        source['answers'][0]['arguments']['pairs']=[[1,2,3]]
        with self.assertRaises(x.Rejected):x.convert_row(source,0,'a'*64)

    def test_missing_requiredness_uses_documented_conservative_projection(self):
        source=row();source['tools']=[{'name':'conservative','parameters':{'n':{'type':'int','default':'10'}}}]
        source['answers']=[{'name':'conservative','arguments':{}}]
        with self.assertRaisesRegex(x.Rejected,'required_argument_missing'):x.convert_row(source,0,'a'*64)
        source['answers'][0]['arguments']={'n':3}
        result=x.convert_row(source,0,'a'*64)
        self.assertEqual(result['tools'][0]['function']['parameters']['required'],['n'])
        self.assertNotIn('default',result['tools'][0]['function']['parameters']['properties']['n'])

    def test_json_schema_numeric_equality_not_boolean_equality(self):
        with self.assertRaises(x.Rejected):x.validate([1,1.0],{'type':'array','items':{'type':'number'},'uniqueItems':True})
        x.validate(1.0,{'type':'number','enum':[1]})
        x.validate([1,True],{'type':'array','items':{},'uniqueItems':True})

    def test_source_name_cannot_impersonate_alias_across_split(self):
        sources=[row(i=1,name='pkg0.call',query='First original fixture'),
                 row(i=2,name='xlam_pkg0_call_a2f7f7abf5',query='Second original fixture')]
        selected,manifest,_=x.prepare(sources,'a'*64,10,10,seed='s',valid_fraction=.5,alias_names=True)
        names=[t['function']['name'] for rs in selected.values() for r in rs for t in r['tools']]
        self.assertEqual(len(names),len(set(names)))

    def test_families_disjoint_and_dedup_before_selection(self):
        rows=[row(i=i,name='family%d.call'%i,query='Original synthetic query %d'%i) for i in range(100)]
        rows.append(copy.deepcopy(rows[0]));rows[-1]['id']=999;rows[-1]['query']='  ORIGINAL   SYNTHETIC QUERY 0 '
        selected,manifest,exclusions=x.prepare(rows,'a'*64,20,10)
        self.assertEqual(manifest['lexical_duplicates_removed'],1)
        self.assertEqual(len(selected['train']),20);self.assertEqual(len(selected['valid']),10)
        a=set(manifest['selected_api_families']['train']);b=set(manifest['selected_api_families']['valid'])
        self.assertFalse(a&b)
        again,_,_=x.prepare(list(reversed(rows)),'a'*64,20,10)
        self.assertEqual([[r['metadata']['source_row_canonical_sha256'] for r in selected[k]] for k in ('train','valid')],
                         [[r['metadata']['source_row_canonical_sha256'] for r in again[k]] for k in ('train','valid')])

    def test_mixed_split_offered_functions_are_purged(self):
        names={}
        for i in range(100):
            name='group%d.call'%i
            names.setdefault(x.source_split([x.family(name)],'s',0.5),name)
        source=row();source['tools']=json.loads(source['tools'])
        source['tools'][0]['name']=names['train'];source['tools'].append(copy.deepcopy(source['tools'][0]));source['tools'][1]['name']=names['valid']
        source['answers']=[{'name':names['train'],'arguments':{'n':3}}]
        selected,manifest,_=x.prepare([source],'a'*64,10,10,seed='s',valid_fraction=0.5)
        self.assertEqual(manifest['cross_split_rows_purged'],1)
        self.assertFalse(selected['train'] or selected['valid'])

    def test_normalized_record_passes_existing_public_structure_validator(self):
        path=Path(__file__).resolve().parents[1]/'validate_dataset.py'
        spec=importlib.util.spec_from_file_location('existing_validator',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        self.assertEqual(mod.validate_record(x.convert_row(row(),0,'a'*64)),[])


if __name__=='__main__':unittest.main()

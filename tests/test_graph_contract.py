import ast
import contextlib
import copy
import io
import json
import logging
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from graph_contract import normalize_scene_graph, normalize_goal_graph, merge_scene_graphs, relation_match_score
from ablation_protocol import ABLATIONS, apply_ablation, fusion_coefficients, write_run_manifest, CallMeter, write_episode_result

def agent_class():
    """Execute the real class methods without importing simulator/provider SDKs."""
    tree = ast.parse((SRC / 'hgwm_agent.py').read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HGWMAgent')
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module in {'graph_contract','ablation_protocol'}]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    class Base:
        def update_curiosity_value(self, scores, reason):
            return scores, reason
    ns = {'WMNavAgent':Base, 'np':np, 'copy':copy, 'math':math, 'json':json, 'logging':logging}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+imports+[cls], type_ignores=[])), str(SRC/'hgwm_agent.py'), 'exec'), ns)
    return ns['HGWMAgent']

Agent = agent_class()

def new_agent():
    agent = Agent.__new__(Agent)
    agent._current_goal = 'bed'
    agent.fine_weight_mode = 'adaptive'
    agent.static_fine_weights = {'room':.3,'object':.4,'spatial':.2,'semantic':.05,'depth':.05}
    agent.use_structural_correction = False
    agent.semantic_voxel_map = {}
    agent.get_enhanced_scene_graph_for_overlap = lambda direction: {'nodes': [], 'edges': []}
    return agent

def scene(name='bed', room='bedroom'):
    return {'room_nodes': {room: {'id':'r1','room_type':room,'avg_confidence':.8}},
            'object_nodes':[{'id':'o1','name':name,'parent_room':room,'confidence':.8,
                             'depth_info':{'depth_category':'foreground'}}],
            'room_object_edges':[{'from':'r1','to':'o1','type':'contains','weight':.8}]}

GOAL = {'target_room_hierarchy': {'primary_room':'bedroom','secondary_rooms':[]},
        'spatial_anchor_objects':['bed','nightstand']}

class GraphContractTests(unittest.TestCase):
    def test_delayed_dictionary_observations_populate_persistent_memory(self):
        a=new_agent();a.step_ndx=0;a.map_scale=100
        a.scene_graph_global={'nodes':[],'edges':[]};a.object_position_tracker={}
        a._global_to_grid=lambda p:(round(float(p[0])*100),round(float(p[2])*100))
        a._create_spatial_relationships=lambda *args:None
        state=type('State',(),{'position':np.array([0.,0.,0.]),'rotation':None})()
        globals_=a._update_semantic_voxel_mapping.__func__.__globals__
        globals_['local_to_global']=lambda position,rotation,local:position+local
        with contextlib.redirect_stdout(io.StringIO()):
            a._update_semantic_voxel_mapping(1,np.pi/2,state,None)
            a._backfill_semantic_to_spatial_positions({'90':{'Objects':{'bed':'foreground','nightstand':'left'},'Room':'bedroom'}})
            result=Agent.get_enhanced_scene_graph_for_overlap(a,90)
        self.assertEqual({n['name'] for n in result['object_nodes']},{'bed','nightstand'})
        self.assertEqual(result['schema_diagnostics']['room_count'],1)
        self.assertEqual(len(result['depth_layers']['foreground']),2)
        a._backfill_semantic_to_spatial_positions({})
        self.assertEqual(len(a.semantic_voxel_map),1)
        self.assertEqual(a._pending_spatial_positions,[])

    def test_no_persistent_control_clears_prior_step_semantics(self):
        a=new_agent();a.decision_mode='prior_prompt_only';a.use_scene_memory=False
        a.semantic_voxel_map={'old':{'objects':['bed']}}
        a.scene_graph_global={'nodes':[{'id':'old'}],'edges':[]};a.object_position_tracker={'bed':[1]}
        with contextlib.redirect_stdout(io.StringIO()):
            a.update_curiosity_value({'90':5},{},vlm_predictions={},goal_subgraph=GOAL)
        self.assertEqual(a.semantic_voxel_map,{})
        self.assertEqual(a.scene_graph_global['nodes'],[])

    def test_graph_direct_keeps_context_but_bypasses_fine_and_arbitrator(self):
        a=new_agent();a.decision_mode='graph_direct';a.use_scene_memory=True
        a._backfill_semantic_to_spatial_positions=lambda _:None
        a._map_objects_to_spatial_coordinates_with_voxel=lambda *args:[]
        a._build_current_scene_graph=lambda *args:scene()
        a._calculate_graph_overlap_score_with_subgraph=lambda *args,**kwargs:self.fail('Fine scorer called')
        a._perform_llm_spatial_reasoning=lambda *args:self.fail('Arbitrator called')
        with contextlib.redirect_stdout(io.StringIO()):
            output=a.update_curiosity_value({'90':4},{},vlm_predictions={},goal_subgraph=GOAL)
        self.assertEqual(output[0],{'90':4})
        self.assertEqual(a._direction_scene_data['90']['current_graph']['object_nodes'][0]['name'],'bed')

    def test_flat_and_hierarchical_evidence_reach_typed_views(self):
        g=normalize_scene_graph(scene())
        self.assertEqual(g['object_nodes'][0]['room'],'bedroom')
        self.assertIn('bedroom',g['room_nodes'])
        self.assertEqual(g['room_object_edges'][0]['relation'],'contains')
        self.assertEqual(normalize_scene_graph(g),g)
        flat=normalize_scene_graph({'nodes':[{'id':'b','name':'bed','room_type':'bedroom'}],'edges':[]})
        self.assertEqual(len(flat['room_object_edges']),1)

    def test_goal_prompt_alias_and_membership_are_normalized(self):
        g=normalize_goal_graph({'target_room_connection':{'primary_room':'living room'},
                                'spatial_anchor_objects':{'side table':{'name':'side table'}}},'sofa')
        self.assertEqual(g['target_room_hierarchy']['primary_room'],'living room')
        self.assertIn(('living_room','room'),{(n['name'],n['type']) for n in g['nodes']})
        self.assertIn('sofa',{n['name'] for n in g['nodes']})
        self.assertEqual(g['target_object'],'sofa')

    def test_relations_match_semantics_despite_different_instance_ids(self):
        a=normalize_scene_graph({'nodes':[{'id':'a','name':'bed'},{'id':'b','name':'nightstand'}],
                                 'edges':[{'from':'a','to':'b','type':'near'}]})
        b=normalize_scene_graph({'nodes':[{'id':'x','name':'bed'},{'id':'y','name':'nightstand'}],
                                 'edges':[{'source':'y','target':'x','relation':'beside'}]})
        self.assertEqual(relation_match_score(a,b,'object_edges'),1)
        b['edges'][0]['relation']='on'
        self.assertEqual(relation_match_score(a,b,'object_edges'),0)

    def test_dangling_edges_are_recorded_not_guessed(self):
        g=normalize_scene_graph({'nodes':[{'id':'a','name':'bed'}],
                                 'edges':[{'source':'a','target':'missing','relation':'near'}]})
        self.assertEqual(g['edges'],[])
        self.assertEqual(len(g['schema_diagnostics']['rejected_edges']),1)

    def test_current_attributes_override_same_memory_id(self):
        a={'nodes':[{'id':'o','name':'chair','room_type':'bedroom','confidence':.1}],'edges':[]}
        b={'nodes':[{'id':'o','name':'chair','room_type':'bedroom','confidence':.9}],'edges':[]}
        self.assertEqual(merge_scene_graphs(a,b)['object_nodes'][0]['confidence'],.9)

    def test_refinement_updates_typed_relations_and_preserves_other_fields(self):
        a=new_agent()
        original=normalize_scene_graph({'nodes':[{'id':'b','name':'bed','room_type':'bedroom'},
                                                 {'id':'n','name':'nightstand','room_type':'bedroom'}],
                                        'edges':[{'from':'b','to':'n','type':'on','confidence':.2}]})
        before=copy.deepcopy(original)
        fixed=a._apply_corrections(original,{'corrected_relationships':[{'source':'bed','target':'nightstand','type':'near'}]})
        self.assertEqual(original,before)
        self.assertIn('bedroom',fixed['room_nodes'])
        self.assertEqual({e['relation'] for e in fixed['object_edges']},{'near'})
        self.assertEqual(len(fixed['object_nodes']),2)

    def test_real_scoring_responds_to_graph_and_prior_evidence(self):
        a=new_agent()
        with contextlib.redirect_stdout(io.StringIO()):
            yes=a._calculate_graph_overlap_score_with_subgraph({},GOAL,90,current_graph=scene())
            no=a._calculate_graph_overlap_score_with_subgraph({},GOAL,90,current_graph=scene('toilet','bathroom'))
        self.assertFalse(yes['diagnostics']['fallback_used'])
        self.assertEqual(yes['diagnostics']['room_count'],1)
        self.assertGreater(yes['room_match_score'],no['room_match_score'])
        self.assertGreater(yes['object_match_score'],no['object_match_score'])
        self.assertGreater(yes['overlap_score'],no['overlap_score'])

    def test_no_observations_do_not_receive_positive_match_defaults(self):
        a=new_agent()
        with contextlib.redirect_stdout(io.StringIO()):
            result=a._calculate_graph_overlap_score_with_subgraph({},GOAL,90)
        self.assertEqual(result['overlap_score'],0)
        self.assertFalse(result['diagnostics']['fallback_used'])
        self.assertEqual(result['diagnostics']['object_count'],0)

    def test_malformed_graph_fallback_is_diagnosed(self):
        a=new_agent()
        with contextlib.redirect_stdout(io.StringIO()):
            result=a._calculate_graph_overlap_score_with_subgraph({},GOAL,90,current_graph='invalid')
        self.assertEqual(result['overlap_score'],.5)
        self.assertTrue(result['diagnostics']['fallback_used'])
        self.assertEqual(result['diagnostics']['error_type'],'TypeError')

    def test_prior_prompt_only_bypasses_graph_scoring_and_context(self):
        a=new_agent();a.decision_mode='prior_prompt_only';a.use_scene_memory=True
        a._backfill_semantic_to_spatial_positions=lambda _: None
        a._direction_scene_data={'old':'context'}
        scores={'30':4,'90':6}
        with contextlib.redirect_stdout(io.StringIO()):
            output=a.update_curiosity_value(scores,{},vlm_predictions={},goal_subgraph=GOAL)
        self.assertEqual(output[0],scores)
        self.assertEqual(a._direction_scene_data,{})

    def test_planning_uses_degrees_once_and_reads_room_dictionary(self):
        a=new_agent();a.decision_mode='full';a.subtask_history=[]
        captured={}
        def prompt(*args,**kwargs):captured.update(kwargs);return 'planning'
        a._construct_prompt=prompt
        a._eval_response=lambda _: {'Flag':False,'Subtask':'explore'}
        a.PlanVLM=type('Client',(),{'call':lambda self,*args:'response'})()
        a.make_plan([None],{},'','bed',{'direction_degrees':90,'current_graph':normalize_scene_graph(scene())})
        self.assertIn('(90°)',captured['scene_graph_context'])
        self.assertNotIn('2700',captured['scene_graph_context'])
        self.assertIn('bedroom',captured['scene_graph_context'])

class ProtocolTests(unittest.TestCase):
    def test_call_meter_preserves_results_and_exceptions(self):
        clock=iter([0.,2.,2.,5.])
        class Client:
            def call(self, fail=False):
                if fail:raise RuntimeError('provider failure')
                return 'ok'
        meter=CallMeter(Client(),clock=lambda:next(clock))
        self.assertEqual(meter.call(),'ok')
        with self.assertRaises(RuntimeError):meter.call(True)
        self.assertEqual(meter.snapshot()['client_calls'],2)
        self.assertEqual(meter.snapshot()['errors'],1)
        self.assertEqual(meter.snapshot()['seconds'],5)

    def test_episode_json_preserves_fractional_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs=dict(dataset='hm3d_v0.2',split='val',episode_index=3,episode={'episode_id':'x','scene_id':'s'},seed=48,
                        ablation='full',success=True,spl=.5,path_length=2,calls={})
            row=write_episode_result(Path(directory)/'episode_result.json',**kwargs)
            self.assertEqual(row['spl'],.5)
            self.assertTrue(row['success'])
            kwargs['spl']=50
            with self.assertRaises(ValueError):write_episode_result(Path(directory)/'bad.json',**kwargs)

    def test_scale_control_preserves_original_total_mass(self):
        for a,b,e in [(.6,.1,.5),(.6,.4,.4),(.8,.5,.2)]:
            x,y,z=fusion_coefficients(a,b,e,False,'matched')
            self.assertAlmostEqual(x+y+z,a+b+e)
            self.assertEqual(z,0)
            self.assertAlmostEqual(x/y,a/b)
            self.assertEqual(fusion_coefficients(a,b,e,False,'legacy'),(a,b,0))

    def test_modes_are_distinct_and_do_not_mutate_input(self):
        source={'agent_cfg':{'hgwm_cfg':{'fine_weight_mode':'adaptive'}}}
        self.assertEqual(apply_ablation(source,'uniform_weights')['agent_cfg']['hgwm_cfg']['fine_weight_mode'],'uniform')
        self.assertEqual(source['agent_cfg']['hgwm_cfg']['fine_weight_mode'],'adaptive')
        self.assertIn('prior_prompt_only',ABLATIONS)
        self.assertIn('graph_memory_direct',ABLATIONS)

    def test_manifest_redacts_credentials_and_fingerprints_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'src').mkdir();(root/'src/a.py').write_text('x=1')
            manifest=write_run_manifest(root/'out.json',{'model':{'api_key':'secret-value'},'seed':7},7,'full',root)
            self.assertEqual(manifest['config']['model']['api_key'],'[REDACTED]')
            self.assertNotIn('secret-value',(root/'out.json').read_text())
            self.assertIn('src/a.py',manifest['source_sha256'])

if __name__ == '__main__':unittest.main()

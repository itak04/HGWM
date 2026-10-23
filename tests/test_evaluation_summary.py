import importlib.util
from pathlib import Path
import unittest
import json
import tempfile

path=Path(__file__).resolve().parents[1]/'scripts/summarize_evaluation.py'
spec=importlib.util.spec_from_file_location('evaluation_summary',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def row(episode,scene,success,spl):
    return {'dataset':'hm3d_v0.2','split':'val','scene_id':scene,'episode_id':str(episode),
            'seed':48,'success':success,'spl':spl,'source_fingerprint':'test'}

class SummaryTests(unittest.TestCase):
    def test_missing_selected_episode_is_not_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'episode_result.json').write_text(json.dumps(row(1,'s',True,.5)))
            manifest={'dataset':'hm3d_v0.2','split':'val','seed':48,
                      'episodes':[{'episode_id':'1','scene_id':'s'},{'episode_id':'2','scene_id':'s'}]}
            (root/'worker.episodes.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):module.load_results(root)
    def test_failed_episode_cannot_have_positive_SPL(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'episode_result.json').write_text(json.dumps(row(1,'s',False,.4)))
            with self.assertRaises(ValueError):module.load_results(root)
    def test_paired_scene_bootstrap_known_constant_difference(self):
        baseline=[row(1,'s1',False,0),row(2,'s2',False,0)]
        variant=[row(1,'s1',True,.1),row(2,'s2',True,.1)]
        result=module.paired_summary(baseline,variant,100,7)
        self.assertEqual(result['delta_SR_pp'],100)
        self.assertAlmostEqual(result['delta_SPL_pp'],10)
        self.assertEqual(result['scene_clusters'],2)
        self.assertEqual(result['CI95']['delta_SR_pp'],[100,100])
    def test_unmatched_episodes_are_rejected(self):
        with self.assertRaises(ValueError):module.paired_summary([row(1,'s',True,.5)],[row(2,'s',True,.5)])
    def test_single_scene_has_no_fabricated_interval(self):
        result=module.paired_summary([row(1,'s',True,.5)],[row(1,'s',True,.6)])
        self.assertIsNone(result['CI95'])
    def test_duplicate_episode_keys_are_rejected(self):
        r=row(1,'s',True,.5)
        with self.assertRaises(ValueError):module.paired_summary([r,r],[r])

if __name__=='__main__':unittest.main()

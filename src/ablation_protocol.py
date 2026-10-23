"""Explicit experiment controls and non-secret run provenance."""
from __future__ import annotations
import copy
import hashlib
import json
import time
from pathlib import Path

class CallMeter:
    """Transparent client-call timing; no prompt text or credentials are stored."""
    def __init__(self, client, clock=time.perf_counter):
        self.client, self.clock = client, clock
        self.reset_meter()
    def __getattr__(self, name):
        return getattr(self.client, name)
    def reset_meter(self):
        self.calls = self.errors = 0
        self.seconds = 0.0
    def call(self, *args, **kwargs):
        start=self.clock();self.calls+=1
        try:
            result=self.client.call(*args,**kwargs)
            if result is None or (isinstance(result, str) and not result):self.errors+=1
            return result
        except Exception:
            self.errors+=1
            raise
        finally:
            self.seconds+=self.clock()-start
    def snapshot(self):
        return {'client_calls':self.calls,'errors':self.errors,'seconds':self.seconds,
                'scope':'client calls; provider-internal retries and token counts are not expanded'}

ABLATIONS = {
    'full': {},
    'no_prior': {'use_goal_subgraph': False},
    'no_persistent_graph': {'use_scene_memory': False},
    'no_correction': {'use_structural_correction': False},
    'no_arbitration': {'use_llm_arbitration': False, 'arbitration_scale': 'legacy'},
    'no_arbitration_matched_scale': {'use_llm_arbitration': False, 'arbitration_scale': 'matched'},
    'uniform_weights': {'fine_weight_mode': 'uniform'},
    'static_weights': {'fine_weight_mode': 'static'},
    'prior_prompt_only': {'decision_mode': 'prior_prompt_only', 'use_scene_memory': False,
                          'use_structural_correction': False, 'use_llm_arbitration': False},
    'graph_memory_direct': {'decision_mode': 'graph_direct', 'use_structural_correction': False,
                            'use_llm_arbitration': False},
}

def apply_ablation(config, name):
    if name not in ABLATIONS:
        raise ValueError('Unknown ablation: ' + name)
    result = copy.deepcopy(config)
    result.setdefault('agent_cfg', {}).setdefault('hgwm_cfg', {}).update(ABLATIONS[name])
    return result

def fusion_coefficients(a, b, e, use_arbitration=True, scale='legacy'):
    if scale not in {'legacy', 'matched'}:
        raise ValueError('arbitration_scale must be legacy or matched')
    if min(a, b, e) < 0:
        raise ValueError('fusion coefficients must be nonnegative')
    if use_arbitration:
        return a, b, e
    if scale == 'matched':
        if a + b <= 0:
            raise ValueError('cannot match scale with zero remaining coefficient mass')
        factor = (a + b + e) / (a + b)
        return a * factor, b * factor, 0.0
    return a, b, 0.0

def write_run_manifest(path, config, seed, ablation, source_root):
    """Never serialize environment variables or credential values."""
    def sanitize(value):
        if isinstance(value, dict):
            return {k: ('[REDACTED]' if any(s in k.lower() for s in ['api_key','password','secret','access_token','auth_token','refresh_token'])
                        else sanitize(v)) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(v) for v in value]
        return value
    source_root = Path(source_root)
    hashes = {}
    for folder in ['src', 'scripts', 'config']:
        for file in sorted((source_root / folder).rglob('*')):
            if file.is_file() and file.suffix in {'.py', '.yaml', '.yml'}:
                hashes[file.relative_to(source_root).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest = {'protocol_version':'graph-contract-v1', 'seed':seed, 'ablation':ablation,
                'config':sanitize(config), 'source_sha256':hashes,
                'seed_scope':'Python and NumPy; does not guarantee deterministic remote model responses'}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    return manifest

def write_episode_result(path, *, dataset, split, episode_index, episode, seed,
                         ablation, success, spl, path_length, calls, source_fingerprint=None):
    spl=float(spl)
    if not 0 <= spl <= 1:
        raise ValueError('episode SPL must be a fraction in [0,1], not a percentage')
    if not success and spl != 0:
        raise ValueError('unsuccessful episodes must have zero SPL')
    record={'dataset':dataset,'split':split,'episode_index':int(episode_index),
            'episode_id':str(episode.get('episode_id',episode_index)),
            'scene_id':str(episode.get('scene_id','unknown')),'seed':int(seed),
            'ablation':ablation,'success':bool(success),'spl':spl,
            'path_length_m':float(path_length),'client_usage':calls,
            'source_fingerprint':source_fingerprint,'protocol_version':'graph-contract-v1'}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,indent=2),encoding='utf8')
    return record

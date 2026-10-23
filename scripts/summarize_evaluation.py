"""Summarize explicit episode_result.json records; never infer outcomes from images."""
from __future__ import annotations
import argparse, json, math, random
from collections import defaultdict
from pathlib import Path

def load_results(root):
    rows=[]
    for path in sorted(Path(root).rglob('episode_result.json')):
        row=json.loads(path.read_text(encoding='utf8'))
        if not isinstance(row.get('success'),bool):raise ValueError(f'{path}: success must be boolean')
        value=row.get('spl')
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError(f'{path}: SPL must be a fraction in [0,1]')
        if not row['success'] and value != 0:
            raise ValueError(f'{path}: unsuccessful episodes must have zero SPL')
        rows.append(row)
    if not rows:raise ValueError('No episode_result.json records; legacy logs are not silently converted')
    observed={key(row) for row in rows}
    for manifest in Path(root).rglob('*.episodes.json'):
        data=json.loads(manifest.read_text(encoding='utf8'))
        expected={(str(data['dataset']),str(data['split']),str(e['scene_id']),str(e['episode_id']),str(data['seed']))
                  for e in data['episodes']}
        if not expected <= observed:
            raise ValueError(f'{manifest}: selected episodes lack result records; do not silently omit failed runs')
    return rows

def key(row):
    return tuple(str(row[k]) for k in ['dataset','split','scene_id','episode_id','seed'])

def paired_summary(baseline, variant, resamples=10000, seed=0):
    if resamples<1:raise ValueError('resamples must be positive')
    def index(rows):
        result={}
        if len({str(row.get('source_fingerprint')) for row in rows})>1:
            raise ValueError('Mixed source versions within one variant')
        for row in rows:
            k=key(row)
            if k in result:raise ValueError('Duplicate episode/scene/seed record: '+str(k))
            result[k]=row
        return result
    a,b=index(baseline),index(variant)
    if a.keys()!=b.keys():raise ValueError('Paired evaluation requires identical dataset/split/scene/episode/seed keys')
    if not a:raise ValueError('Empty paired sample')
    groups=defaultdict(list)
    for k in sorted(a):
        groups[(k[0],k[1],k[2])].append((100*(int(b[k]['success'])-int(a[k]['success'])),100*(b[k]['spl']-a[k]['spl'])))
    flat=[pair for values in groups.values() for pair in values]
    mean=lambda values,column:sum(v[column] for v in values)/len(values)
    result={'paired_episodes':len(flat),'scene_clusters':len(groups),'difference':'variant minus baseline',
            'delta_SR_pp':mean(flat,0),'delta_SPL_pp':mean(flat,1),
            'source_fingerprints':{'baseline':sorted({str(r.get('source_fingerprint')) for r in baseline}),
                                   'variant':sorted({str(r.get('source_fingerprint')) for r in variant})}}
    if len(groups)<2:
        result['CI95']=None;result['warning']='Fewer than two scene clusters; no interval is reported'
        return result
    rng=random.Random(seed);clusters=list(groups.values());samples=[[],[]]
    for _ in range(resamples):
        drawn=[p for _ in clusters for p in rng.choice(clusters)]
        for j in range(2):samples[j].append(mean(drawn,j))
    intervals={}
    for name,values in zip(['delta_SR_pp','delta_SPL_pp'],samples):
        values.sort();intervals[name]=[values[int(.025*(resamples-1))],values[int(.975*(resamples-1))]]
    result.update(CI95=intervals,bootstrap={'unit':'scene cluster','resamples':resamples,'seed':seed})
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',required=True);parser.add_argument('--variant',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--resamples',type=int,default=10000);parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();baseline=load_results(args.baseline);variant=load_results(args.variant)
    result=paired_summary(baseline,variant,args.resamples,args.seed)
    result['point_estimates']={name:{'episodes':len(rows),'SR_percent':100*sum(r['success'] for r in rows)/len(rows),
                                    'SPL_percent':100*sum(r['spl'] for r in rows)/len(rows)}
                              for name,rows in [('baseline',baseline),('variant',variant)]}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()

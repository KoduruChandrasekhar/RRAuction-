"""Step 12 - apply the final curated buy lists (prices.json -> final_lists) as the submission order.
Curated names take positions 1..n in the given order; any other auction-pool players follow in model order.
Usage: python -I analysis/12_curate.py <work_dir>"""
import sys, os, json, numpy as np, pandas as pd
W = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
FL = json.load(open(os.path.join(HERE, 'prices.json'), encoding='utf-8')).get('final_lists', {})
for role in ['bat', 'bowl']:
    f = pd.read_csv(f'{W}/ratings_{role}.csv', index_col=0)
    cur = FL.get(role, [])
    if not cur:
        continue
    model_order = f[f.pool_rank.notna()].sort_values('pool_rank').index.tolist()
    ids = []
    for nm in cur:
        hit = f.index[f['name'] == nm]
        if len(hit):
            ids.append(hit[0])
        else:
            print('WARNING: not rated', nm)
    excl = set(FL.get('exclude', []))
    ids += [i for i in model_order if i not in ids and f.loc[i, 'name'] not in excl]
    f['pool_rank'] = np.nan
    f.loc[ids, 'pool_rank'] = range(1, len(ids) + 1)
    f.to_csv(f'{W}/ratings_{role}.csv')
    print(role, [f.loc[i, 'name'] for i in ids[:10]])

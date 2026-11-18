"""Step 12 - ensure the submission pool preserves strict NeRRve rank order.
The buy-plan lists in prices.json are separate from the data-driven Top 10.
Usage: python -I analysis/12_curate.py <work_dir>"""
import sys, os, json, numpy as np, pandas as pd
W = sys.argv[1]
for role in ['bat', 'bowl']:
    f = pd.read_csv(f'{W}/ratings_{role}.csv', index_col=0)
    pool = f[f.pool_rank.notna()].copy()
    ids = pool.sort_values('RUPI', ascending=False).index.tolist()
    f['pool_rank'] = np.nan
    f.loc[ids, 'pool_rank'] = range(1, len(ids) + 1)
    f.to_csv(f'{W}/ratings_{role}.csv')
    print(role, [f.loc[i, 'name'] for i in ids[:10]])

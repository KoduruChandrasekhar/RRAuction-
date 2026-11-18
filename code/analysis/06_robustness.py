"""Step 6 - robustness of the final lists (league top-10 and auction-pool top-10) across alternative
pressure definitions and pillar weightings.  Usage: python -I analysis/06_robustness.py <work_dir>"""
import sys, json, pandas as pd, numpy as np
W = sys.argv[1]
VARIANTS = {'_vLI': 'Pure win-prob leverage', '_vHYBin': 'Quartile within each innings',
            '_v3G': 'Equal-thirds pressure blend', '_vq65': 'Wider zone (top 35%)', '_vq85': 'Narrower zone (top 15%)'}
WEIGHTS = {'equal': 'Equal pillar weights', 'performance_only': 'Pressure Value only',
           'clutch_heavy': 'Resilience-heavy', 'impact_heavy': 'Win-impact-heavy'}
out = {}
for role in ['bat', 'bowl']:
    f = pd.read_csv(f'{W}/ratings_{role}.csv', index_col=0)
    res = {}
    for key, rankcol, prefix in [('league', 'rank', 'rk_'), ('pool', 'pool_rank', 'prk_')]:
        top = f[f[rankcol].notna()].sort_values(rankcol).head(10)
        rows = {}
        for tag, label in VARIANTS.items():
            v = pd.read_csv(f'{W}/ratings_{role}{tag}.csv', index_col=0)
            rows[label] = v[rankcol].reindex(top.index)
        for col, label in WEIGHTS.items():
            rows[label] = f.loc[top.index, prefix + col]
        t = pd.DataFrame(rows)
        overlap = {k: int((v <= 10).sum()) for k, v in rows.items()}
        top20 = {k: int((v <= 20).sum()) for k, v in rows.items()}
        tab = t.copy()
        tab.insert(0, 'final', top[rankcol].astype(int))
        tab.insert(0, 'name', top['name'])
        res[key] = dict(table=tab.reset_index().replace({np.nan: None}).to_dict('records'), overlap_top10=overlap, in_top20=top20,
                        mean_overlap=round(float(np.mean(list(overlap.values()))), 1))
        print(role, key, 'mean overlap', res[key]['mean_overlap'], overlap)
    out[role] = res
json.dump(out, open(f'{W}/robustness.json', 'w'), indent=1)

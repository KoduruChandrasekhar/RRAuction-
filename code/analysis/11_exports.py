"""Step 11 - submission helpers: the exact dropdown picks (in rank order) and clean rating tables.
Usage: python -I analysis/11_exports.py <work_dir> <submission_dir>"""
import sys, json, pandas as pd
W, S = sys.argv[1], sys.argv[2]
IDX = sys.argv[3] if len(sys.argv) > 3 else 'NeRRve'
out = ['# SupeRR Selector S2 - Round 1 dropdown picks (released IPL players, ranked by the ' + IDX + ' Index)', '']
for role, label in [('bat', 'Batsman'), ('bowl', 'Bowler')]:
    f = pd.read_csv(f'{W}/ratings_{role}.csv', index_col=0)
    top = f[f.pool_rank.notna()].sort_values('pool_rank').head(10)
    for i, (_, r) in enumerate(top.iterrows(), 1):
        out.append(f'{label} {i}: {r["name"]}')
    out.append('')
    lg = f[f['rank'].notna()].sort_values('rank').head(10)
    out.append(f'(Reference - unfiltered league-wide top 10 {label.lower()}s: ' + ', '.join(lg['name']) + ')')
    out.append('')
    cols = ['name', 'country', 'avail', 'avail_note', 'team27', 'rank', 'pool_rank', 'riser_rank', 'RUPI', 'PV_u', 'NV_u', 'RES_u',
            'CON', 'WIN_u', 'p_n', 'p_matches', 'boot_top10', 'pool_top10', 'pool_lo', 'pool_hi', 'last_date']
    f.sort_values('score', ascending=False)[cols].round(3).rename(columns={'RUPI': IDX}).to_csv(f'{S}/{IDX}_ratings_{"batters" if role == "bat" else "bowlers"}.csv')
out.append('Availability from the official IPL 2027 retention lists (17 Nov 2026). Selection rule: buyable on 16 Dec, active since 2024, 60+ pressure balls, IPL-tested or on the 17 Nov release lists; ranked by NeRRve. RR: Rs 27.05 cr, 6 slots (3 overseas).')
open(f'{S}/Dropdown_Picks.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))

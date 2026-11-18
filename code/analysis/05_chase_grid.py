"""Step 5 - precompute a chase-state grid (runs needed x balls left x wickets in hand) holding the
win probability and the Pressure-Index percentile, for the dashboard's live Pressure Calculator.
Uses the same 2nd-innings win-probability spec as 02_models.py (deterministic re-fit).
Usage: python -I analysis/05_chase_grid.py <work_dir>"""
import sys, json, base64, warnings, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGC
warnings.filterwarnings('ignore')
W = sys.argv[1]
b = pd.read_parquet(f'{W}/balls_scored.parquet')
b['wkts_left'] = 10 - b.wkts_before
b['req_n'] = b.runs_req / b.env_rpb
b['rrr_n'] = b.rrr / (6 * b.env_rpb)
F2 = ['req_n', 'wkts_left', 'balls_left', 'rrr_n']; M2 = [-1, 1, 1, -1]
d = b[(~b.tie) & (b.innings == 2)]
m2 = HGC(max_iter=300, learning_rate=0.05, max_leaf_nodes=24, min_samples_leaf=300, l2_regularization=1.0,
         monotonic_cst=M2, random_state=7).fit(d[F2], d.bat_team_won)
ENV = b[b.env == 'IPL-2026'].env_rpb.iloc[0]
probs = pd.read_csv(f'{W}/outcome_probs.csv', index_col=[0, 1, 2])


def wp2(rr, wl, bl):
    rr, wl, bl = (np.asarray(a, float) for a in (rr, wl, bl))
    blc = np.clip(bl, 1, None)
    X = pd.DataFrame({'req_n': rr / ENV, 'wkts_left': np.clip(wl, 1, 10), 'balls_left': blc, 'rrr_n': (rr * 6 / blc) / (6 * ENV)})
    p = m2.predict_proba(X[F2])[:, 1]
    p = np.where(rr <= 0, 1.0, p)
    return np.where((rr > 0) & ((wl <= 0) | (bl <= 0)), np.where(rr == 1, 0.5, 0.0), p)


R, B, K = np.arange(1, 151), np.arange(1, 121), np.arange(1, 11)
rr, bl, wl = [a.ravel() for a in np.meshgrid(R, B, K, indexing='ij')]
wp = wp2(rr, wl, bl)
over = np.clip((120 - bl) // 6 + 1, 1, 20)
phase = np.where(over <= 6, 'Powerplay', np.where(over <= 15, 'Middle', 'Death'))
li = np.zeros(len(rr))
outs = {'0': (0, 0, 1), '1': (1, 0, 1), '2': (2, 0, 1), '3': (3, 0, 1), '4': (4, 0, 1), '6': (6, 0, 1), 'W': (0, 1, 1), 'wd': (1, 0, 0)}
for k, (r, w, l) in outs.items():
    pk = np.array([probs.loc[('IPL-2026', 2, ph), k] for ph in ['Powerplay', 'Middle', 'Death']])
    pv = np.where(phase == 'Powerplay', pk[0], np.where(phase == 'Middle', pk[1], pk[2]))
    li += pv * np.abs(wp2(rr - r, wl - w, bl - l) - wp)
LI = li / b.li_raw.mean()
ref = np.sort(b.loc[b.innings == 2, 'LI'].values)          # PI percentile within 2nd innings (league match)
pct = np.searchsorted(ref, LI) / len(ref)
grid = dict(runs=[1, 150], balls=[1, 120], wkts=[1, 10], order='runs,balls,wkts',
            wp=base64.b64encode(np.round(wp * 200).astype(np.uint8).tobytes()).decode(),
            pct=base64.b64encode(np.round(pct * 200).astype(np.uint8).tobytes()).decode(),
            li=base64.b64encode(np.clip(np.round(LI * 20), 0, 255).astype(np.uint8).tobytes()).decode())
json.dump(grid, open(f'{W}/chase_grid.json', 'w'))
print('grid ok', len(rr), 'cells; WP 30 off 18 (5 wkts)=', round(float(wp2([30], [5], [18])[0]), 3), '; 12 off 6 (5 wkts)=', round(float(wp2([12], [5], [6])[0]), 3))

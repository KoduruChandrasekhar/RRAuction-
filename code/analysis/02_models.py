"""Step 2 - context models trained on the data pack:
  (a) Win-probability models (1st & 2nd innings, monotonic gradient-boosted models)
  (b) Leverage Index  -> the pressure engine
  (c) Run-expectancy model -> state-dependent run cost of a wicket
  (d) Expected-outcome models (xRuns / xOut for batters, xConceded / xWkt for bowlers)
Usage: python -I analysis/02_models.py <work_dir>"""
import sys, json, warnings, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
warnings.filterwarnings('ignore')
W = sys.argv[1]
b = pd.read_parquet(f'{W}/balls.parquet')
metrics = {}

# ---------------- (a) win probability ----------------
b['wkts_left'] = 10 - b.wkts_before
# Scoring environment enters ONLY through normalisation (runs relative to par / normal scoring rate), never as a
# raw feature: a raw per-season value lets trees isolate one season and memorise its match outcomes.
b['runs_frac'] = b.runs_before / b.env_par
b['req_n'] = b.runs_req / b.env_rpb
b['rrr_n'] = b.rrr / (6 * b.env_rpb)
F1 = ['runs_frac', 'wkts_before', 'balls_left']; M1 = [1, -1, 1]
F2 = ['req_n', 'wkts_left', 'balls_left', 'rrr_n']; M2 = [-1, 1, 1, -1]
train = b[~b.tie]


def wp_model(M):
    return HGC(max_iter=300, learning_rate=0.05, max_leaf_nodes=24, min_samples_leaf=300,
               l2_regularization=1.0, monotonic_cst=M, random_state=7)


wp = {}
for inn, F, M in [(1, F1, M1), (2, F2, M2)]:
    d = train[train.innings == inn]
    X, y, grp = d[F], d.bat_team_won, d.match_id
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(X, y, grp):
        mdl = wp_model(M).fit(X.iloc[tr], y.iloc[tr])
        oof[te] = mdl.predict_proba(X.iloc[te])[:, 1]
    metrics[f'wp_inn{inn}'] = dict(auc=roc_auc_score(y, oof), brier=brier_score_loss(y, oof),
                                   logloss=log_loss(y, oof), n=len(d))
    cal = pd.DataFrame({'p': oof, 'y': y.values})
    cal['bin'] = pd.cut(cal.p, np.linspace(0, 1, 11))
    metrics[f'wp_inn{inn}_calib'] = (cal.groupby('bin', observed=True)
                                     .agg(pred=('p', 'mean'), actual=('y', 'mean'), n=('y', 'size'))
                                     .round(3).reset_index().astype({'bin': str}).to_dict('records'))
    wp[inn] = wp_model(M).fit(X, y)
print(json.dumps({k: v for k, v in metrics.items() if 'calib' not in k}, indent=1))


def wp2(runs_req, wkts_left, balls_left, env_rpb):
    """P(chasing side wins), with exact terminal states."""
    runs_req, wkts_left, balls_left = (np.asarray(a, float) for a in (runs_req, wkts_left, balls_left))
    bl = np.clip(balls_left, 1, None)
    er = np.asarray(env_rpb, float)
    X = pd.DataFrame({'req_n': runs_req / er, 'wkts_left': np.clip(wkts_left, 1, 10), 'balls_left': bl,
                      'rrr_n': (runs_req * 6 / bl) / (6 * er)})
    p = wp[2].predict_proba(X[F2])[:, 1]
    p = np.where(runs_req <= 0, 1.0, p)
    lost = (runs_req > 0) & ((wkts_left <= 0) | (balls_left <= 0))
    p = np.where(lost, np.where(runs_req == 1, 0.5, 0.0), p)
    return p


def wp1(runs, wkts_lost, balls_left, env_par, env_rpb):
    """P(team batting first wins). End of innings -> 1 - P(chase of runs+1 from 0/0 in 120 balls)."""
    runs, wkts_lost, balls_left = (np.asarray(a, float) for a in (runs, wkts_lost, balls_left))
    X = pd.DataFrame({'runs_frac': runs / np.asarray(env_par, float), 'wkts_before': np.clip(wkts_lost, 0, 9),
                      'balls_left': np.clip(balls_left, 1, None)})
    p = wp[1].predict_proba(X[F1])[:, 1]
    end = (wkts_lost >= 10) | (balls_left <= 0)
    if end.any():
        p_end = 1 - wp2(runs + 1, np.full(len(runs), 10), np.full(len(runs), 120), env_rpb)
        p = np.where(end, p_end, p)
    return p


def wp_state(d, runs, wk, bl):
    return np.where(d.innings == 1, wp1(runs, wk, bl, d.env_par, d.env_rpb),
                    wp2(d.target - runs, 10 - wk, bl, d.env_rpb))


b['wp_before'] = wp_state(b, b.runs_before, b.wkts_before, b.balls_left)
runs_a = b.runs_before + b.team_runs
wk_a = b.wkts_before + b.wkt.astype(int)
bl_a = b.balls_left - b.legal.astype(int)
b['wp_after'] = wp_state(b, runs_a, wk_a, bl_a)
last = b.groupby(['match_id', 'innings']).seq.transform('max') == b.seq
fin2 = last & (b.innings == 2)
b.loc[fin2, 'wp_after'] = np.where(b.loc[fin2, 'tie'], 0.5, b.loc[fin2, 'bat_team_won'])
b['wpa'] = b.wp_after - b.wp_before                      # batting-side WPA on this delivery

# ---------------- (b) leverage index ----------------
b['_o'] = np.select(
    [b.wkt, ~b.legal, b.team_runs == 0, b.team_runs == 1, b.team_runs == 2, b.team_runs == 3,
     b.team_runs.isin([4, 5])],
    ['W', 'wd', '0', '1', '2', '3', '4'], '6')
probs = pd.crosstab([b.env, b.innings, b.phase], b._o, normalize='index')
outs = {'0': (0, 0, 1), '1': (1, 0, 1), '2': (2, 0, 1), '3': (3, 0, 1), '4': (4, 0, 1),
        '6': (6, 0, 1), 'W': (0, 1, 1), 'wd': (1, 0, 0)}
P = b[['env', 'innings', 'phase']].merge(probs, left_on=['env', 'innings', 'phase'], right_index=True, how='left')
li = np.zeros(len(b))
for k, (r, w, l) in outs.items():
    wp_o = wp_state(b, b.runs_before + r, b.wkts_before + w, b.balls_left - l)
    li += P[k].values * np.abs(wp_o - b.wp_before.values)
b['li_raw'] = li
b['LI'] = b.li_raw / b.li_raw.mean()
b['importance'] = np.where(b.knockout, 1.25, 1.0)
print('LI by innings\n', b.groupby('innings').LI.describe().round(2))

# ---------------- (c) run expectancy (remaining runs) ----------------
i1 = b[b.innings == 1].copy()
i1['rem_runs'] = i1.groupby('match_id').team_runs.transform('sum') - i1.runs_before
re_m = HGR(max_iter=300, learning_rate=0.05, min_samples_leaf=200, monotonic_cst=[1, -1, 1], random_state=7)
re_m.fit(i1[['balls_left', 'wkts_before', 'env_rpb']], i1.rem_runs)


def RE(bl, wk, env_rpb):
    bl = np.asarray(bl, float); wk = np.asarray(wk, float)
    v = re_m.predict(pd.DataFrame({'balls_left': np.clip(bl, 1, None), 'wkts_before': np.clip(wk, 0, 9),
                                   'env_rpb': np.asarray(env_rpb, float)}))
    return np.where((wk >= 10) | (bl <= 0), 0.0, np.clip(v, 0, None))


bl_next = b.balls_left - 1
b['wkt_cost'] = (RE(bl_next, b.wkts_before, b.env_rpb) - RE(bl_next, b.wkts_before + 1, b.env_rpb)).clip(1.0)
print('wicket cost by phase', b.groupby(['innings', 'phase']).wkt_cost.mean().round(2).to_dict())
re_grid = pd.DataFrame([(bl, wk, float(RE([bl], [wk], [1.55])[0])) for bl in [120, 90, 60, 36, 24, 12, 6]
                        for wk in range(10)], columns=['balls_left', 'wkts_lost', 'exp_runs'])
re_grid.to_csv(f'{W}/run_expectancy_grid.csv', index=False)

# ---------------- (d) expected outcome models (situation only; no player identity) ----------------
CTX = ['innings', 'over_no', 'balls_left', 'wkts_before', 'runs_before', 'crr', 'runs_req', 'rrr',
       'env_rpb', 'env_wpb', 'LI']


def fit_pred(d, target, clf):
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(d, groups=d.match_id):
        mdl = (HGC if clf else HGR)(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                    min_samples_leaf=300, random_state=7)
        mdl.fit(d[CTX].iloc[tr], d[target].iloc[tr])
        oof[te] = mdl.predict_proba(d[CTX].iloc[te])[:, 1] if clf else mdl.predict(d[CTX].iloc[te])
    return oof


bat = b[b.batter_ball].copy()
b.loc[bat.index, 'x_bat_runs'] = fit_pred(bat.assign(t=bat.bat_runs), 't', False)
b.loc[bat.index, 'x_striker_out'] = fit_pred(bat.assign(t=bat.striker_out.astype(int)), 't', True)
b['x_bowl_runs'] = fit_pred(b.assign(t=b.bowl_runs), 't', False)
b['x_bowler_wkt'] = fit_pred(b.assign(t=b.bowler_wkt.astype(int)), 't', True)
chk = {}
for a, x in [('bat_runs', 'x_bat_runs'), ('striker_out', 'x_striker_out'), ('bowl_runs', 'x_bowl_runs'),
             ('bowler_wkt', 'x_bowler_wkt')]:
    s = b[b[x].notna()]
    chk[a] = [round(float(s[a].astype(float).mean()), 4), round(float(s[x].mean()), 4)]
print('actual vs expected', chk)
metrics['xmodel_check'] = chk

# ---------------- ball values (runs above expectation, wicket-adjusted) ----------------
b['bv_bat'] = np.where(b.batter_ball,
                       (b.bat_runs - b.x_bat_runs) - b.wkt_cost * (b.striker_out.astype(int) - b.x_striker_out),
                       np.nan)
b['bv_bowl'] = (b.x_bowl_runs - b.bowl_runs) + b.wkt_cost * (b.bowler_wkt.astype(int) - b.x_bowler_wkt)
b = b.drop(columns=['_o'])
b.to_parquet(f'{W}/balls_scored.parquet', index=False)
probs.to_csv(f'{W}/outcome_probs.csv')
json.dump(metrics, open(f'{W}/model_metrics.json', 'w'), indent=1, default=str)
print('saved')

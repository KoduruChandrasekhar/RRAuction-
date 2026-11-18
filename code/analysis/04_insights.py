"""Step 4 - evidence layer: pressure validation, reliability tests, player deep-dives, team pressure
profiles (RR lens), Release Radar for contracted players, purse & value-for-money for auction targets.
Writes outputs/work/insights.json (used by the report and dashboard).
Usage: python -I analysis/04_insights.py <data_dir> <work_dir>"""
import sys, os, json, warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
D, W = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
b = pd.read_parquet(f'{W}/balls_scored.parquet')
fl = pd.read_parquet(f'{W}/pressure_flags.parquet')
b = b.merge(fl, on=['match_id', 'innings', 'seq'])
bat = pd.read_csv(f'{W}/ratings_bat.csv', index_col=0)
bowl = pd.read_csv(f'{W}/ratings_bowl.csv', index_col=0)
mp = pd.read_csv(f'{D}/competition_team_player_mapping.csv')
teams = pd.read_csv(f'{D}/teams.csv').drop_duplicates('team_id').set_index('team_id').team_name.to_dict()
metrics = json.load(open(f'{W}/model_metrics.json'))
prices = json.load(open(os.path.join(HERE, 'prices.json'), encoding='utf-8'))
sal = json.load(open(os.path.join(HERE, 'salaries_2026.json'), encoding='utf-8'))
b['w_bat'] = b.w_recency * b.q_batter
b['w_bowl'] = b.w_recency * b.q_bowler
out = {}
J = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else x

# ------------------------------------------------ 1. does the pressure definition separate situations?
tot_abs = b.wpa.abs().sum()
out['wpa_share'] = {
    'pressure_balls_share': float(b.pressure.mean()),
    'pressure_wpa_share': float(b.loc[b.pressure, 'wpa'].abs().sum() / tot_abs),
    'extreme_balls_share': float(b.extreme.mean()),
    'extreme_wpa_share': float(b.loc[b.extreme, 'wpa'].abs().sum() / tot_abs),
    'bottom50_wpa_share': float(b.loc[b.PI_pct < 0.5, 'wpa'].abs().sum() / tot_abs)}


def prof(d):
    lg = d[d.batter_ball]
    return dict(balls=int(len(lg)), rpo=float(6 * d.team_runs.sum() / d.legal.sum()),
                dismissal_per_100=float(100 * lg.striker_out.mean()), dot_pct=float(100 * (lg.bat_runs == 0).mean()),
                boundary_pct=float(100 * lg.is_boundary.mean()), avg_abs_wpa=float(100 * d.wpa.abs().mean()),
                mean_wp=float(d.wp_before.mean()), rrr=float(d.rrr.median()) if (d.innings == 2).all() else None)


out['profile'] = {f'inn{i}_{"pressure" if p else "normal"}': prof(b[(b.innings == i) & (b.pressure == p)])
                  for i in [1, 2] for p in [True, False]}
out['zone_by_phase'] = {f'{k[0]}|{k[1]}': float(v) for k, v in (b[b.pressure].groupby(['innings', 'phase']).size() / b.pressure.sum()).items()}
out['pressure_rate_by_phase'] = {f'{k[0]}|{k[1]}': float(v) for k, v in b.groupby(['innings', 'phase']).pressure.mean().items()}
c2 = b[(b.innings == 2) & (b.runs_req > 0)].copy()
RB = [0, 6, 12, 18, 24, 30, 36, 48, 60, 72, 90, 120, 400]
RBL = ['1-6', '7-12', '13-18', '19-24', '25-30', '31-36', '37-48', '49-60', '61-72', '73-90', '91-120', '120+']
BB = [0, 6, 12, 18, 24, 36, 48, 60, 72, 90, 120]
BBL = ['1-6', '7-12', '13-18', '19-24', '25-36', '37-48', '49-60', '61-72', '73-90', '91-120']
c2['rb'] = pd.cut(c2.runs_req, RB, labels=RBL); c2['bb'] = pd.cut(c2.balls_left, BB, labels=BBL)
hm = c2.groupby(['rb', 'bb'], observed=True).agg(share=('pressure', 'mean'), pi=('PI_pct', 'mean'), n=('PI', 'size'), wp=('wp_before', 'mean')).reset_index()
out['chase_heat'] = hm[hm.n >= 15].astype({'rb': str, 'bb': str}).round(3).to_dict('records')
out['chase_heat_axes'] = {'runs': RBL, 'balls': BBL}
c1 = b[b.innings == 1]
hm1 = c1.groupby(['over_no', 'wkts_before']).agg(share=('pressure', 'mean'), n=('PI', 'size')).reset_index()
out['first_heat'] = hm1[hm1.n >= 15].round(3).to_dict('records')
end = b.sort_values('seq').groupby(['match_id', 'innings']).tail(1)
e2 = end[end.innings == 2].set_index('match_id')
close = ((e2.bat_team_won == 1) & (e2.balls_left - e2.legal.astype(int) <= 6)) | \
        ((e2.bat_team_won == 0) & (e2.target - (e2.runs_before + e2.team_runs) <= 12))
mstat = b.groupby('match_id').agg(press_share=('pressure', 'mean'), extreme_share=('extreme', 'mean'))
mstat['close'] = close.reindex(mstat.index).fillna(False).astype(bool)
out['close_vs_onesided'] = {('close' if k else 'one_sided'): float(v) for k, v in mstat.groupby('close').press_share.mean().items()}
out['close_vs_onesided_extreme'] = {('close' if k else 'one_sided'): float(v) for k, v in mstat.groupby('close').extreme_share.mean().items()}
out['n_close'] = int(mstat.close.sum()); out['n_matches'] = int(len(mstat))
out['knockout_press'] = {('knockout' if k else 'league'): float(v) for k, v in b.groupby('knockout').pressure.mean().items()}
out['wp_metrics'] = metrics
env = b.groupby('env').apply(lambda d: pd.Series({'rpo': 6 * d.team_runs.sum() / d.legal.sum(), 'par': d.env_par.iloc[0], 'matches': d.match_id.nunique()}))
out['env_table'] = env.round(2).reset_index().to_dict('records')
out['re_grid'] = pd.read_csv(f'{W}/run_expectancy_grid.csv').round(1).to_dict('records')
out['wkt_cost_phase'] = {f'{k[0]}|{k[1]}': float(v) for k, v in b.groupby(['innings', 'phase']).wkt_cost.mean().items()}
out['data'] = dict(balls=int(len(b)), matches=int(b.match_id.nunique()), ipl_matches=int(b[b.comp_type == 'IPL'].match_id.nunique()),
                   t20i_matches=int(b[b.comp_type == 'T20I'].match_id.nunique()), emerging_matches=int(b[b.comp_type == 'Emerging'].match_id.nunique()),
                   first=str(b.match_date.min().date()), last=str(b.match_date.max().date()),
                   batters=int(b.striker_id.nunique()), bowlers=int(b.bowler_id.nunique()), pressure_balls=int(b.pressure.sum()))


# ------------------------------------------------ 2. reliability: split-half repeatability
def split_half(role, R):
    pmt = pd.read_parquet(f'{W}/pm_{role}.parquet')
    pmt = pmt[pmt.pid.isin(R.index[R['rank'].notna()])]
    res = {'pressure': [], 'normal': [], 'resilience': [], 'scoring': [], 'survival': [], 'win': []}
    for seed in range(10):
        rng = np.random.default_rng(seed)
        mids = pmt.match_id.unique(); half = pd.Series(rng.integers(0, 2, len(mids)), index=mids)
        x = pmt.assign(h=pmt.match_id.map(half))
        g = x.groupby(['pid', 'pressure', 'h'])[['s1', 's2', 'sw', 'wsum']].sum()
        g['v'] = (g.s1 + g.s2) / g.wsum; g['c1'] = g.s1 / g.wsum; g['c2'] = g.s2 / g.wsum; g['w'] = g.sw / g.wsum
        P = g.xs(True, level=1).unstack('h'); N = g.xs(False, level=1).unstack('h')
        idx = P.index.intersection(N.index); P, N = P.loc[idx].dropna(), N.loc[idx].dropna()
        idx = P.index.intersection(N.index); P, N = P.loc[idx], N.loc[idx]
        r = lambda a, c: spearmanr(a, c).correlation
        res['pressure'].append(r(P.v[0], P.v[1])); res['normal'].append(r(N.v[0], N.v[1]))
        res['resilience'].append(r(P.v[0] - N.v[0], P.v[1] - N.v[1]))
        res['scoring'].append(r(P.c1[0], P.c1[1])); res['survival'].append(r(P.c2[0], P.c2[1])); res['win'].append(r(P.w[0], P.w[1]))
    return {k: round(float(np.mean(v)), 3) for k, v in res.items()}


out['split_half'] = {'bat': split_half('bat', bat), 'bowl': split_half('bowl', bowl)}
elig_b, elig_w = bat[bat['rank'].notna()], bowl[bowl['rank'].notna()]
out['pv_nv_corr'] = {'bat': float(spearmanr(elig_b.PV, elig_b.NV).correlation), 'bowl': float(spearmanr(elig_w.PV, elig_w.NV).correlation)}
out['shrink_demo'] = {'bat': elig_b[['name', 'rawPV_u', 'PV_u', 'p_n']].round(2).reset_index().to_dict('records'),
                      'bowl': elig_w[['name', 'rawPV_u', 'PV_u', 'p_n']].round(2).reset_index().to_dict('records')}

# ------------------------------------------------ 3. player deep dives
mp26 = mp[mp.comp_id == 3298]
# RR's 2027 squad = players the availability rules place at RR (retained + traded in, released/traded out removed)
rr26 = set(bat.index[bat.avail == 'RR squad']) | set(bowl.index[bowl.avail == 'RR squad'])


def splits(pid, role):
    if role == 'bat':
        d = b[(b.striker_id == pid) & b.batter_ball]; bv = 'bv_bat'; w = 'w_bat'; unit = 100
    else:
        d = b[b.bowler_id == pid]; bv = 'bv_bowl'; w = 'w_bowl'; unit = 24
    p = d[d.pressure]

    def val(x, mn=15):
        return None if len(x) < mn else round(float((x[bv] * x[w]).sum() / x[w].sum() * unit), 1)
    parts = {'Setting (1st inns)': p[p.innings == 1], 'Chasing / defending (2nd inns)': p[p.innings == 2],
             'Powerplay': p[p.phase == 'Powerplay'], 'Middle overs': p[p.phase == 'Middle'], 'Death overs': p[p.phase == 'Death'],
             'IPL': p[p.comp_type == 'IPL'], 'T20I / A-team': p[p.comp_type != 'IPL'], 'Extreme pressure (top 10%)': d[d.extreme]}
    if role == 'bat':
        parts['vs pace'] = p[p.pace_or_spin == 'Pace']; parts['vs spin'] = p[p.pace_or_spin == 'Spin']
    s = {k: val(x) for k, x in parts.items()}
    sn = {k: int(len(x)) for k, x in parts.items()}
    trend = []
    for yr, x in d.groupby(d.match_date.dt.year):
        trend.append(dict(year=int(yr), pv=val(x[x.pressure], 12), nv=val(x[~x.pressure], 12), pballs=int(x.pressure.sum())))
    sign = 1 if role == 'bat' else -1
    rows = []
    for mid, x in d.groupby('match_id'):
        if x.pressure.sum() < (4 if role == 'bat' else 6):
            continue
        rows.append(dict(match_id=int(mid), wpa=round(float(sign * x.wpa.sum()), 3), p_balls=int(x.pressure.sum()),
                         runs=int(x.bat_runs.sum() if role == 'bat' else x.bowl_runs.sum()),
                         balls=int(len(x) if role == 'bat' else x.legal.sum()),
                         wk=int(x.striker_out.sum() if role == 'bat' else x.bowler_wkt.sum()),
                         date=x.match_date.iloc[0].strftime('%d %b %Y'), comp=x.comp_name.iloc[0],
                         opp=teams.get(int(x.bowling_team_id.iloc[0] if role == 'bat' else x.batting_team_id.iloc[0]), ''),
                         inn=int(x.innings.iloc[0]), stage=x.stage.iloc[0],
                         won=bool((x.bat_team_won.iloc[0] == 1) == (role == 'bat'))))
    mm = sorted(rows, key=lambda r: -r['wpa'])
    pin = p.groupby('match_id').agg(v=(bv, 'sum'), n=(bv, 'size'))
    pin = pin[pin.n >= (4 if role == 'bat' else 6)]
    rec = dict(innings=int(len(pin)), positive=int((pin.v > 0).sum()))
    usage = {k: round(float(v), 3) for k, v in p.phase.value_counts(normalize=True).items()}
    return dict(splits=s, split_n=sn, trend=trend, moments=mm[:3], record=rec, usage=usage,
                chase_share=round(float((p.innings == 2).mean()), 3) if len(p) else None)


deep = {}
for role, R in [('bat', bat), ('bowl', bowl)]:
    ids = list(R.sort_values('score', ascending=False).index[:30])
    ids += list(R[R.pool_rank.notna()].sort_values('pool_rank').index[:20])
    ids += [i for i in R.index if i in rr26 or R.loc[i, 'avail'] == 'Watch list']
    ids += list(R[R.avail == 'Auction pool'].sort_values('score', ascending=False).index[:40])
    deep[role] = {str(int(pid)): splits(pid, role) for pid in dict.fromkeys(ids)}
out['deep'] = deep

# ------------------------------------------------ 4. team pressure profiles (IPL 2025-26) -> RR lens
ipl = b[(b.comp_type == 'IPL') & (b.year >= 2025)]


def team_prof(d, team_col, bv, w, unit):
    p = d[d.pressure]
    g = p.groupby([team_col, 'phase']).apply(lambda x: (x[bv] * x[w]).sum() / x[w].sum() * unit).unstack()
    g['Overall'] = p.groupby(team_col).apply(lambda x: (x[bv] * x[w]).sum() / x[w].sum() * unit)
    g['balls'] = p.groupby(team_col).size()
    g.index = g.index.map(teams)
    return g.round(2)


tb = team_prof(ipl[ipl.batter_ball], 'batting_team_id', 'bv_bat', 'w_bat', 100)
tw = team_prof(ipl, 'bowling_team_id', 'bv_bowl', 'w_bowl', 24)
out['team_bat'] = tb.reset_index().rename(columns={'batting_team_id': 'team'}).to_dict('records')
out['team_bowl'] = tw.reset_index().rename(columns={'bowling_team_id': 'team'}).to_dict('records')
iplm = b[(b.comp_type == 'IPL') & (b.year >= 2025)].groupby('match_id').agg(h=('home_team_id', 'first'), a=('away_team_id', 'first'),
                                                                             w=('win_team_id', 'first'), yr=('year', 'first'))
iplm['close'] = iplm.index.map(mstat.close)
rows = []
for t in set(iplm.h) | set(iplm.a):
    g = iplm[(iplm.h == t) | (iplm.a == t)]
    rows.append(dict(team=teams.get(t), close=int(g.close.sum()), close_won=int((g.close & (g.w == t)).sum()),
                     played=int(len(g)), won=int((g.w == t).sum())))
out['close_record'] = sorted(rows, key=lambda r: -r['close_won'] / max(r['close'], 1))
# RR by-phase pressure ranks among the 10 franchises
out['rr_ranks'] = {'bat': {c: int(tb[c].rank(ascending=False)['Rajasthan Royals']) for c in ['Powerplay', 'Middle', 'Death', 'Overall']},
                   'bowl': {c: int(tw[c].rank(ascending=False)['Rajasthan Royals']) for c in ['Powerplay', 'Middle', 'Death', 'Overall']}}


def squad_rows(R, role):
    rows = []
    for pid in rr26:
        if pid in R.index:
            r = R.loc[pid]
            rows.append(dict(id=int(pid), name=r['name'], role=role, rank=J(float(r['rank'])) if pd.notna(r['rank']) else None,
                             RUPI=round(float(r.RUPI), 1), PV=round(float(r.PV_u), 2), NV=round(float(r.NV_u), 2),
                             RES=round(float(r.RES_u), 2), p_n=int(r.p_n), overseas=bool(r.overseas)))
    return sorted(rows, key=lambda x: -x['RUPI'])


out['rr_squad'] = {'bat': squad_rows(bat, 'bat'), 'bowl': squad_rows(bowl, 'bowl')}
out['rr_squad_list'] = sorted(set(bat.loc[bat.index.isin(rr26), 'name']) | set(bowl.loc[bowl.index.isin(rr26), 'name']))

# ------------------------------------------------ 5. Release Radar (contracted pressure performers)
i26 = b[(b.comp_type == 'IPL') & (b.year == 2026)]
form_bat = i26[i26.batter_ball].groupby('striker_id').agg(v=('bv_bat', 'mean'), n=('bv_bat', 'size'), m=('match_id', 'nunique'))
form_bowl = i26.groupby('bowler_id').agg(v=('bv_bowl', 'mean'), n=('bv_bowl', 'size'), m=('match_id', 'nunique'))
for f, k in [(form_bat, 100), (form_bowl, 24)]:
    reg = f[f.n >= 60]
    f['v_u'] = f.v * k
    f['pct'] = f.v.rank(pct=True)
    f.loc[f.n < 60, 'pct'] = np.nan
team_games26 = i26.groupby('batting_team_id').match_id.nunique()


def radar(R, role, form):
    rows = []
    cand = R[R.avail.isin(['Contracted', 'Watch list']) & ((R.RUPI >= 58) | (R.avail == 'Watch list'))]
    other = form_bowl if role == 'bat' else form_bat
    for pid, r in cand.iterrows():
        fee = sal['fees'].get(r['name'])
        fm = form.loc[pid] if pid in form.index else None
        fo = other.loc[pid] if pid in other.index else None
        m26 = max(int(fm.m) if fm is not None else 0, int(fo.m) if fo is not None else 0)
        pcts = [float(x.pct) for x in (fm, fo) if x is not None and pd.notna(x.pct)]
        pct = max(pcts) if pcts else None            # an all-rounder is judged on his better 2026 suit
        if r['name'] in sal['cornerstones']:
            verdict, prob = 'Retain - franchise cornerstone', 'Low'
        elif m26 <= 3 and (fee or 0) >= 4:
            verdict, prob = 'Likely release - barely played in 2026 on a big fee', 'High'
        elif pct is not None and pct >= 0.6:
            verdict, prob = ('Release & buy-back possible - strong form but fee very high', 'Medium') if (fee or 0) >= 18 else ('Retain - strong 2026 form', 'Low')
        elif pct is not None and pct >= 0.4:
            verdict, prob = ('Release & buy-back possible - average form, high fee', 'Medium') if (fee or 0) >= 10 else ('Retain - good value', 'Low')
        elif pct is not None:
            verdict, prob = ('Likely release - below-par 2026 on a high fee', 'High') if (fee or 0) >= 6 else ('Toss-up - below-par 2026, cheap', 'Medium')
        else:
            verdict, prob = ('Likely release - little 2026 game time', 'High') if (fee or 0) >= 6 else ('Toss-up - small 2026 sample', 'Medium')
        if r['avail'] == 'Watch list' and prob == 'Low':
            prob = 'Medium'
        rows.append(dict(id=int(pid), name=r['name'], team=r['team27'], role=role, RUPI=round(float(r.RUPI), 1),
                         rank=J(float(r['rank'])) if pd.notna(r['rank']) else None, fee=fee, m26=m26,
                         form26=round(float(fm.v_u), 1) if fm is not None else None, form_pct=None if pct is None else round(pct, 2),
                         verdict=verdict, release_prob=prob, tipped=r['avail'] == 'Watch list', note=r['avail_note']))
    return sorted(rows, key=lambda x: ({'High': 0, 'Medium': 1, 'Low': 2}[x['release_prob']], -x['RUPI']))


rr_all = pd.DataFrame(radar(bat, 'bat', form_bat) + radar(bowl, 'bowl', form_bowl))
rr_all = rr_all.sort_values('RUPI', ascending=False).drop_duplicates('name')   # one row per player, stronger suit
rr_all['_o'] = rr_all.release_prob.map({'High': 0, 'Medium': 1, 'Low': 2})
out['release_radar'] = rr_all.sort_values(['_o', 'RUPI'], ascending=[True, False]).drop(columns='_o').replace({np.nan: None}).to_dict('records')

# ------------------------------------------------ 5b. Release Radar scorecard: 16 Nov predictions vs official 17 Nov lists
pre_path = f'{W}/radar_pre_retention.json'
if os.path.exists(pre_path):
    pre = json.load(open(pre_path))
    allR = pd.concat([bat[['name', 'avail', 'team27']], bowl[['name', 'avail', 'team27']]]).drop_duplicates('name').set_index('name')
    sc = []
    for x in pre:
        a = allR.avail.get(x['name'])
        actual = 'Released' if a in ('Auction pool', 'Unavailable') else ('Traded' if (allR.team27.get(x['name']) not in (None, x['team']) and a != 'Auction pool') else 'Retained')
        sc.append(dict(name=x['name'], team=x['team'], role=x['role'], RUPI=x['RUPI'], fee=x['fee'], predicted=x['release_prob'], verdict=x['verdict'], actual=actual))
    scd = pd.DataFrame(sc)
    hit = lambda p, a: (p == 'High' and a != 'Retained') or (p == 'Low' and a == 'Retained')
    out['radar_scorecard'] = dict(rows=sc, high_n=int((scd.predicted == 'High').sum()), high_released=int(((scd.predicted == 'High') & (scd.actual != 'Retained')).sum()),
                                  low_n=int((scd.predicted == 'Low').sum()), low_retained=int(((scd.predicted == 'Low') & (scd.actual == 'Retained')).sum()),
                                  med_n=int((scd.predicted == 'Medium').sum()), med_released=int(((scd.predicted == 'Medium') & (scd.actual != 'Retained')).sum()))
    print('scorecard', {k: v for k, v in out['radar_scorecard'].items() if k != 'rows'})

# ------------------------------------------------ 6. purse & value for money
rp = prices['rr_purse']
out['purse'] = dict(available=rp['available'], slots=rp['slots'], overseas=rp['overseas_slots'], source=rp['source'],
                    market=[dict(team=t, **v) for t, v in sorted(prices.get('market', {}).items(), key=lambda kv: -kv[1]['purse'])])


def value_rows(R, role):
    rows = []
    for pid, r in R[R.pool_rank.notna()].sort_values('pool_rank').head(20).iterrows():
        pr = prices['players'].get(r['name'])
        mid = (pr['lo'] + pr['hi']) / 2 if pr else None
        rows.append(dict(id=int(pid), name=r['name'], role=role, pool_rank=int(r.pool_rank), RUPI=round(float(r.RUPI), 1),
                         price_lo=pr['lo'] if pr else None, price_hi=pr['hi'] if pr else None, basis=pr['basis'] if pr else None,
                         value=round((r.RUPI - 50) / mid, 1) if mid else None, overseas=bool(r.overseas)))
    return rows


out['value'] = value_rows(bat, 'bat') + value_rows(bowl, 'bowl')
json.dump(out, open(f'{W}/insights.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps({k: out[k] for k in ['wpa_share', 'close_vs_onesided', 'split_half', 'pv_nv_corr', 'rr_ranks', 'purse', 'knockout_press']}, indent=1))
print(json.dumps(out['profile'], indent=0)[:900])
print(pd.DataFrame(out['release_radar'])[['name', 'team', 'role', 'RUPI', 'fee', 'm26', 'form26', 'form_pct', 'release_prob', 'verdict']].to_string())
print(pd.DataFrame(out['close_record']).to_string())
print(pd.DataFrame(out['team_bowl']).to_string()); print(pd.DataFrame(out['team_bat']).to_string())
print(out['rr_squad'])

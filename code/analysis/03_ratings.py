"""Step 3 - Pressure Reliability ratings for batters and bowlers  ->  RUPI
(Reliability Under Pressure Index).

Pillars (each an Empirical-Bayes posterior, put on a common *talent* scale = estimate / prior SD,
so a pillar with weak evidence automatically carries less weight):
  PV  - Pressure Value: runs above expectation per ball in the pressure zone, built from two
        separately-shrunk components (batters: scoring + survival; bowlers: economy + wickets)
  CON - Consistency: share of pressure innings / spells that beat expectation (beta-binomial EB)
  RES - Resilience: pressure value minus the player's own non-pressure value (EB on the difference)
  WIN - Win impact: win-probability added per pressure ball (EB)
Usage: python -I analysis/03_ratings.py <data_dir> <work_dir> [pressure_quantile]"""
import sys, os, json, warnings, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import availability as AV
warnings.filterwarnings('ignore')
pd.set_option('future.no_silent_downcasting', True)
D, W = sys.argv[1], sys.argv[2]
PZ_Q = float(sys.argv[3]) if len(sys.argv) > 3 else 0.75
TAG = '' if len(sys.argv) <= 4 else sys.argv[4]
VARIANT = sys.argv[5] if len(sys.argv) > 5 else 'HYBG'
NBOOT = int(sys.argv[6]) if len(sys.argv) > 6 else 300
b = pd.read_parquet(f'{W}/balls_scored.parquet')
mp = pd.read_csv(f'{D}/competition_team_player_mapping.csv')
teams = pd.read_csv(f'{D}/teams.csv').drop_duplicates('team_id').set_index('team_id').team_name.to_dict()
rng = np.random.default_rng(2027)
BASE_W = {'PV': 0.45, 'CON': 0.20, 'RES': 0.20, 'WIN': 0.15}

# ---------------------------------------------------------------- pressure zone
# Pressure Index = Stakes^0.5 x Scoreboard squeeze^0.3 x Wicket squeeze^0.2 x match importance
#   Stakes   : Leverage Index (expected win-probability swing of the ball, mean 1)
#   Scoreboard: required run-rate (chase) or rate needed to reach par (setting) / normal scoring rate
#   Wickets  : wickets down relative to what is normal at that stage of the innings
env_rate = b.env_rpb * 6
need_rate = np.where(b.innings == 2, b.rrr, np.maximum(b.env_par - b.runs_before, 0) * 6 / b.balls_left)
b['SQ_board'] = np.clip(need_rate / env_rate, 0.25, 3.0)
exp_wk = b.groupby(['innings', 'legal_before']).wkts_before.transform('mean')
b['SQ_wkts'] = np.clip(1 + 0.15 * (b.wkts_before - exp_wk), 0.6, 2.0)
EXP = {'LI': (1, 0, 0), 'HYB': (.5, .3, .2), 'HYBG': (.5, .3, .2), 'HYB2G': (.4, .35, .25), 'HYB3G': (.34, .33, .33)}[VARIANT]
b['PI'] = (b.LI.clip(lower=0.01) ** EXP[0]) * (b.SQ_board ** EXP[1]) * (b.SQ_wkts ** EXP[2]) * b.importance
if VARIANT.endswith('G'):
    b['PI_pct'] = b.PI.rank(pct=True)                      # one league-wide scale across both innings
else:
    b['PI_pct'] = b.groupby('innings').PI.rank(pct=True)
b['pressure'] = b.PI_pct >= PZ_Q
b['extreme'] = b.PI_pct >= 0.90
IPL_TILT = float(os.environ.get('IPL_TILT', '1.0'))     # IPL auction: non-IPL evidence counts at this weight
b['comp_w'] = np.where(b.comp_type == 'IPL', 1.0, IPL_TILT)
b['w_bat'] = b.w_recency * b.q_batter * b.comp_w
b['w_bowl'] = b.w_recency * b.q_bowler * b.comp_w
b['season_yr'] = b.match_date.dt.year

# ---------------------------------------------------------------- names / teams / nationality
name_map = mp.sort_values('comp_id').groupby('player_id').player_name.last().to_dict()
for col_id, col_nm in [('striker_id', 'striker_name'), ('bowler_id', 'bowler_name')]:
    for pid, nm in b[[col_id, col_nm]].drop_duplicates().values:
        name_map.setdefault(pid, nm)
ipl26 = mp[mp.comp_id == 3298].set_index('player_id').team_name.to_dict()
NATIONS = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 72, 78, 89, 272, 282, 283, 311, 312, 313, 315, 3886}
A_TEAM_NAT = {222: 18, 341: 25, 3093: 21, 3674: 27, 4444: 78}
both = pd.concat([b[['striker_id', 'batting_team_id']].set_axis(['pid', 'tid'], axis=1),
                  b[['bowler_id', 'bowling_team_id']].set_axis(['pid', 'tid'], axis=1)])
nat = both[both.tid.isin(NATIONS)].groupby('pid').tid.agg(lambda s: s.value_counts().index[0])
nat_a = both[both.tid.isin(A_TEAM_NAT)].groupby('pid').tid.agg(lambda s: A_TEAM_NAT[s.value_counts().index[0]])
nat = nat.combine_first(nat_a)
# players never seen for a national side but in IPL -> Indian domestic (uncapped) by construction
IPL_TEAMS = {1, 2, 3, 4, 5, 7, 8, 242, 3508, 3509}


# ---------------------------------------------------------------- opposition strength (team x season)
# Each side's batting and bowling strength = mean runs above/below expectation it produced in that
# season, shrunk toward 0 (prior worth 1,500 balls). Every ball is then credited relative to the
# attack / line-up actually faced, so runs off a weak attack count for less.
b['bat_c1'] = np.where(b.batter_ball, b.bat_runs - b.x_bat_runs, np.nan)
b['bat_c2'] = np.where(b.batter_ball, -(b.striker_out.astype(int) - b.x_striker_out) * b.wkt_cost, np.nan)
b['bowl_c1'] = b.x_bowl_runs - b.bowl_runs
b['bowl_c2'] = (b.bowler_wkt.astype(int) - b.x_bowler_wkt) * b.wkt_cost
K_TEAM = 1500


def team_strength(team_col, cols):
    g = b.groupby([team_col, 'season_yr'])[cols].agg(['sum', 'count'])
    out = pd.DataFrame({c: g[(c, 'sum')] / (g[(c, 'count')] + K_TEAM) for c in cols})
    return out


opp_bowl = team_strength('bowling_team_id', ['bat_c1', 'bat_c2'])   # what batters achieved vs this attack
opp_bat = team_strength('batting_team_id', ['bowl_c1', 'bowl_c2'])  # what bowlers achieved vs this line-up
ob = opp_bowl.reindex(pd.MultiIndex.from_arrays([b.bowling_team_id, b.season_yr])).values
ot = opp_bat.reindex(pd.MultiIndex.from_arrays([b.batting_team_id, b.season_yr])).values
b['adj_bat_c1'], b['adj_bat_c2'] = np.nan_to_num(ob[:, 0]), np.nan_to_num(ob[:, 1])
b['adj_bowl_c1'], b['adj_bowl_c2'] = np.nan_to_num(ot[:, 0]), np.nan_to_num(ot[:, 1])
team_str = pd.DataFrame({'bowl_attack_runs_per100': -100 * (opp_bowl.bat_c1 + opp_bowl.bat_c2),
                         'bat_lineup_runs_per100': -100 * (opp_bat.bowl_c1 + opp_bat.bowl_c2)})
team_str.index = [f'{teams.get(t, t)} {y}' for t, y in team_str.index]


# ---------------------------------------------------------------- per player-match sums
def player_match_table(role):
    if role == 'bat':
        d = b[b.batter_ball].copy()
        d['pid'] = d.striker_id; d['wq'] = d.w_bat; d['wpa_p'] = d.wpa
        d['c1'] = d.bat_c1 - d.adj_bat_c1                                     # scoring above expectation (opp-adjusted)
        d['c2'] = d.bat_c2 - d.adj_bat_c2                                     # survival value in runs (opp-adjusted)
        d['runs_'] = d.bat_runs; d['out_'] = d.striker_out.astype(int); d['xr_'] = d.x_bat_runs
        d['xo_'] = d.x_striker_out; d['ball_'] = 1; d['dot_'] = (d.bat_runs == 0).astype(int)
    else:
        d = b.copy()
        d['pid'] = d.bowler_id; d['wq'] = d.w_bowl; d['wpa_p'] = -d.wpa
        d['c1'] = d.bowl_c1 - d.adj_bowl_c1                                   # economy value: runs saved (opp-adjusted)
        d['c2'] = d.bowl_c2 - d.adj_bowl_c2                                   # wicket value in runs (opp-adjusted)
        d['runs_'] = d.bowl_runs; d['out_'] = d.bowler_wkt.astype(int); d['xr_'] = d.x_bowl_runs
        d['xo_'] = d.x_bowler_wkt; d['ball_'] = d.legal.astype(int); d['dot_'] = d.is_dot.astype(int)
    d['bv'] = d.c1 + d.c2
    d['bnd_'] = d.is_boundary.astype(int)
    d['ipl_'] = (d.comp_type == 'IPL').astype(int)
    d['fq_'] = ((d.q_batter if role == 'bat' else d.q_bowler) >= 1.0).astype(int)   # vs full-member / IPL opposition
    for c in ['c1', 'c2', 'wpa_p', 'bv']:
        d[f'w_{c}'] = d.wq * d[c]; d[f'w2_{c}'] = d.wq * d[c] ** 2
    d['w2'] = d.wq ** 2
    agg = dict(n=('bv', 'size'), wsum=('wq', 'sum'), w2=('w2', 'sum'),
               s1=('w_c1', 'sum'), q1=('w2_c1', 'sum'), s2=('w_c2', 'sum'), q2=('w2_c2', 'sum'),
               sw=('w_wpa_p', 'sum'), qw=('w2_wpa_p', 'sum'), sb=('w_bv', 'sum'), qb=('w2_bv', 'sum'),
               runs=('runs_', 'sum'), outs=('out_', 'sum'), xr=('xr_', 'sum'), xo=('xo_', 'sum'),
               balls=('ball_', 'sum'), bnd=('bnd_', 'sum'), dots=('dot_', 'sum'), bv_raw=('bv', 'sum'), fq=('fq_', 'sum'), ipl=('ipl_', 'sum'))
    pm = d.groupby(['pid', 'match_id', 'pressure']).agg(**agg).reset_index()
    return d, pm


SUMS = ['n', 'wsum', 'w2', 's1', 'q1', 's2', 'q2', 'sw', 'qw', 'sb', 'qb', 'runs', 'outs', 'xr', 'xo', 'balls', 'bnd', 'dots', 'bv_raw', 'fq', 'ipl']


def mom_tau2(means, neff, sig2, sel):
    m, n = means[sel], neff[sel]
    mu = np.average(m, weights=n)
    return max(np.average((m - mu) ** 2, weights=n) - np.average(sig2 / n, weights=n), 1e-8), mu


def rate_players(pm, params, fixed=None):
    pm = pm.copy()
    if 'bw' in pm:
        pm[SUMS] = pm[SUMS].mul(pm.bw, axis=0)
    P = pm[pm.pressure].groupby('pid')[SUMS].sum().add_prefix('p_')
    N = pm[~pm.pressure].groupby('pid')[SUMS].sum().add_prefix('n_')
    o = P.join(N, how='outer').fillna(0)
    # consistency on pressure innings/spells
    pin = pm[pm.pressure & (pm.n >= params['min_balls_match'] * (pm.bw if 'bw' in pm else 1))]
    bwt = pin.bw if 'bw' in pin else 1.0
    cons = pin.assign(s=(pin.bv_raw > 0).astype(float) * bwt, m=bwt * 1.0).groupby('pid')[['s', 'm']].sum()
    o = o.join(cons.rename(columns={'s': 'succ', 'm': 'p_matches'})).fillna({'succ': 0, 'p_matches': 0})
    f = fixed or {}
    hyper = {}
    sel = o.p_n >= params['min_p']
    for side in ['p', 'n']:
        neff = (o[f'{side}_wsum'] ** 2 / o[f'{side}_w2'].replace(0, np.nan)).fillna(0)
        o[f'{side}_neff'] = neff
        tot = 0
        for c in ['1', '2']:
            mean = (o[f'{side}_s{c}'] / o[f'{side}_wsum'].replace(0, np.nan)).fillna(0)
            key = f'{side}{c}'
            if key in f:
                sig2, tau2, mu = f[key]
            else:
                ws, wq = o[f'{side}_wsum'].sum(), o[f'{side}_q{c}'].sum()
                sig2 = wq / ws - (o[f'{side}_s{c}'].sum() / ws) ** 2
                tau2, mu = mom_tau2(mean.values, neff.clip(1e-9).values, sig2, (o[f'{side}_n'] >= params['min_p']).values)
            hyper[key] = (sig2, tau2, mu)
            k = sig2 / tau2
            post = mu + (mean - mu) * neff / (neff + k)
            o[f'{side}_raw{c}'] = mean
            o[f'{side}_post{c}'] = post
            o[f'{side}_rel{c}'] = neff / (neff + k)
            tot = tot + post
        o[f'{side}_PV'] = tot
    o['PV'] = o.p_PV; o['NV'] = o.n_PV
    o['PV_s'] = o.p_post1; o['PV_o'] = o.p_post2
    tau_pv = np.sqrt(hyper['p1'][1] + hyper['p2'][1]); mu_pv = hyper['p1'][2] + hyper['p2'][2]
    # resilience: raw pressure-minus-baseline difference, EB-shrunk
    raw_p = o.p_raw1 + o.p_raw2; raw_n = o.n_raw1 + o.n_raw2
    o['raw_PV'] = raw_p; o['raw_NV'] = raw_n
    if 'tot' in f:
        sig_p, sig_n = f['tot']
    else:   # per-ball variance of the combined value (components are correlated on dismissal balls)
        sig_p = o.p_qb.sum() / o.p_wsum.sum() - (o.p_sb.sum() / o.p_wsum.sum()) ** 2
        sig_n = o.n_qb.sum() / o.n_wsum.sum() - (o.n_sb.sum() / o.n_wsum.sum()) ** 2
    hyper['tot'] = (sig_p, sig_n)
    dvar = sig_p / o.p_neff.clip(1e-9) + sig_n / o.n_neff.clip(1e-9)
    raw_d = raw_p - raw_n
    if 'd' in f:
        tau2d, mud = f['d']
    else:
        s = (o.p_n >= params['min_p']) & (o.n_n >= params['min_p'])
        mud = np.average(raw_d[s], weights=1 / dvar[s])
        tau2d = max(np.average((raw_d[s] - mud) ** 2, weights=1 / dvar[s]) - np.average(dvar[s], weights=1 / dvar[s]), 1e-8)
    hyper['d'] = (tau2d, mud)
    o['RES'] = mud + (raw_d - mud) * tau2d / (tau2d + dvar)
    o['raw_RES'] = raw_d
    # win impact
    wmean = (o.p_sw / o.p_wsum.replace(0, np.nan)).fillna(0)
    if 'w' in f:
        sig2w, tau2w, muw = f['w']
    else:
        ws = o.p_wsum.sum(); sig2w = o.p_qw.sum() / ws - (o.p_sw.sum() / ws) ** 2
        tau2w, muw = mom_tau2(wmean.values, o.p_neff.clip(1e-9).values, sig2w, sel.values)
    hyper['w'] = (sig2w, tau2w, muw)
    o['raw_WIN'] = wmean
    o['WIN'] = muw + (wmean - muw) * o.p_neff / (o.p_neff + sig2w / tau2w)
    # consistency (beta-binomial EB)
    if 'c' in f:
        p0, tau2c = f['c']
    else:
        s = o.p_matches >= 8
        r = o.succ[s] / o.p_matches[s]
        p0 = o.succ[s].sum() / o.p_matches[s].sum()
        tau2c = max(np.average((r - p0) ** 2, weights=o.p_matches[s]) - np.average(p0 * (1 - p0) / o.p_matches[s], weights=o.p_matches[s]), 1e-6)
    hyper['c'] = (p0, tau2c)
    kc = p0 * (1 - p0) / tau2c
    o['raw_CON'] = o.succ / o.p_matches.replace(0, np.nan)
    o['CON'] = (o.succ + kc * p0) / (o.p_matches + kc)
    # talent-scale z (posterior deviation / prior SD)
    o['zPV'] = (o.PV - mu_pv) / tau_pv
    o['zRES'] = (o.RES - mud) / np.sqrt(tau2d)
    o['zWIN'] = (o.WIN - muw) / np.sqrt(tau2w)
    o['zCON'] = (o.CON - p0) / np.sqrt(tau2c)
    o.attrs['hyper'] = hyper
    return o


def composite(o, weights, ref):
    """score = weighted talent-scale z; RUPI scaled so the league-eligible set has mean 50, sd 15."""
    e = o.copy()
    e['score'] = sum(weights[k] * e[f'z{k}'] for k in weights)
    mu, sd = e.loc[ref, 'score'].mean(), e.loc[ref, 'score'].std(ddof=0)
    e['RUPI'] = 50 + 15 * (e.score - mu) / sd
    e['rank'] = np.nan
    e.loc[ref, 'rank'] = e.loc[ref, 'score'].rank(ascending=False, method='first')
    return e.sort_values('score', ascending=False)


def label_country(pid, name, ipl_balls):
    if name in AV.CFG['nationality_override']:
        return AV.CFG['nationality_override'][name]
    tid = nat.get(pid)
    if tid in A_TEAM_NAT:                    # only ever seen for an 'A' side
        return f"{teams[A_TEAM_NAT[tid]]} (uncapped)"
    if pd.notna(tid):
        return teams.get(tid)
    return 'India (uncapped)' if ipl_balls > 0 else None


nat_raw = both[both.tid.isin(NATIONS)].groupby('pid').tid.agg(lambda s: s.value_counts().index[0])
nat_a2 = both[both.tid.isin(A_TEAM_NAT)].groupby('pid').tid.agg(lambda s: s.value_counts().index[0])
nat = nat_raw.combine_first(nat_a2).to_dict()
WSETS = {'equal': {'PV': .25, 'CON': .25, 'RES': .25, 'WIN': .25},
         'performance_only': {'PV': 1.0, 'CON': 0, 'RES': 0, 'WIN': 0},
         'clutch_heavy': {'PV': .3, 'CON': .2, 'RES': .35, 'WIN': .15},
         'impact_heavy': {'PV': .3, 'CON': .15, 'RES': .15, 'WIN': .4}}

results, summaries = {}, {}
for role in ['bat', 'bowl']:
    params = dict(min_balls_match=4 if role == 'bat' else 6, min_p=60)
    d, pm = player_match_table(role)
    r = rate_players(pm, params)
    hyper = r.attrs['hyper']
    act = d.groupby('pid').agg(last_date=('match_date', 'max'), first_date=('match_date', 'min'),
                               matches=('match_id', 'nunique'),
                               ipl_balls=('comp_type', lambda s: int((s == 'IPL').sum())),
                               emerging_balls=('comp_type', lambda s: int((s == 'Emerging').sum())),
                               all_balls=('comp_type', 'size'), seasons=('season_yr', 'nunique'))
    r = r.join(act)
    min_pb = int(round((120 if role == 'bat' else 150) * (1 - PZ_Q) / 0.25))   # evidence bar scales with zone width
    active = r.last_date >= '2025-01-01'
    min_pm = int(round(12 * (1 - PZ_Q) / 0.25))
    r['eligible'] = (r.p_n >= min_pb) & (r.p_matches >= min_pm) & active                          # league table
    r['rated'] = (r.p_n >= (40 if role == 'bat' else 48)) & (r.p_matches >= 5) & active       # incl. emerging
    _FL = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prices.json'), encoding='utf-8')).get('final_lists', {})
    r['rated'] = r['rated'] | (r.index.map(name_map).isin(_FL.get('bat', []) + _FL.get('bowl', [])) & (r.p_n >= 5))   # curated picks always rated
    r['quality_ok'] = r.p_fq >= (60 if role == 'bat' else 72)   # pressure evidence vs full-member / IPL attacks
    r['name'] = r.index.map(name_map)
    r['ipl26_team'] = r.index.map(ipl26)
    r['country'] = [label_country(pid, nm, ib) for pid, nm, ib in zip(r.index, r.name, r.ipl_balls)]
    cls = [AV.classify(nm, t, c) for nm, t, c in zip(r.name, r.ipl26_team, r.country)]
    r['avail'] = [c[0] for c in cls]
    r['avail_note'] = [c[1] for c in cls]
    r['team27'] = [c[2] for c in cls]
    r['emerging'] = r.rated & ((r.first_date >= '2024-01-01') | (r.emerging_balls > 0)) & ~r.eligible
    r = r[r.rated].copy()
    ref = r.index[r.eligible]
    r['ipl_ok'] = r.p_ipl >= 40                                                     # proven under pressure IN THE IPL
    excl = set(AV.CFG.get('exclude_countries', []))
    released = r.name.isin(AV.CFG['released'].keys())
    base_ok = (r.avail == 'Auction pool') & ~r.country.isin(excl) & (r.p_n >= (60 if role == 'bat' else 72))
    pool = r.index[base_ok & released]                                     # official 17 Nov release lists only
    risers = r.index[(r.avail == 'Auction pool') & ~released & ~r.country.isin(excl) & r.eligible]   # unsold / non-IPL gems (shown, not picked)
    ranked = composite(r, BASE_W, ref)
    ranked['pool_rank'] = np.nan
    _key = ranked.loc[pool, 'score'] + np.where(ranked.loc[pool, 'ipl_ok'], 100, 0)   # IPL-proven first, then by score
    ranked.loc[pool, 'pool_rank'] = _key.rank(ascending=False, method='first')
    ranked['riser_rank'] = np.nan
    ranked.loc[risers, 'riser_rank'] = ranked.loc[risers, 'score'].rank(ascending=False, method='first')
    # ---------------- bootstrap: resample matches (Poisson weights), refit with fixed hyper-parameters
    ids = ranked.index
    matches = pm.match_id.unique()
    B = NBOOT
    rk = np.full((max(B, 1), len(ids)), np.nan)
    rp = np.full((max(B, 1), len(ids)), np.nan)
    is_ref = ids.isin(ref)
    is_pool = ids.isin(pool)
    for i in range(B):
        bw = pd.Series(rng.poisson(1.0, len(matches)), index=matches)
        rb = rate_players(pm.assign(bw=pm.match_id.map(bw).values), params, fixed=hyper).reindex(ids).fillna(0)
        sc = sum(BASE_W[k] * rb[f'z{k}'] for k in BASE_W).values
        rk[i, is_ref] = pd.Series(sc[is_ref]).rank(ascending=False, method='first').values
        rp[i, is_pool] = pd.Series(sc[is_pool]).rank(ascending=False, method='first').values
    with np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        ranked['boot_top10'] = np.where(is_ref, (rk <= 10).mean(axis=0), np.nan)
        ranked['boot_top20'] = np.where(is_ref, (rk <= 20).mean(axis=0), np.nan)
        ranked['boot_med'] = np.nanmedian(rk, axis=0)
        ranked['boot_lo'] = np.nanpercentile(rk, 10, axis=0)
        ranked['boot_hi'] = np.nanpercentile(rk, 90, axis=0)
        ranked['pool_top10'] = np.where(is_pool, (rp <= 10).mean(axis=0), np.nan)
        ranked['pool_med'] = np.nanmedian(rp, axis=0)
        ranked['pool_lo'] = np.nanpercentile(rp, 10, axis=0)
        ranked['pool_hi'] = np.nanpercentile(rp, 90, axis=0)
    # ---------------- weight sensitivity (league-eligible and pool ranks)
    for nm, wts in WSETS.items():
        c = composite(r, wts, ref)
        ranked[f'rk_{nm}'] = c['rank'].reindex(ranked.index)
        ranked[f'prk_{nm}'] = c.loc[pool, 'score'].rank(ascending=False, method='first').reindex(ranked.index)
    # ---------------- readable
    unit = 100 if role == 'bat' else 24
    for c, src in [('PV_u', 'PV'), ('NV_u', 'NV'), ('RES_u', 'RES'), ('PVs_u', 'PV_s'), ('PVo_u', 'PV_o'),
                   ('rawPV_u', 'raw_PV'), ('rawNV_u', 'raw_NV'), ('rawRES_u', 'raw_RES')]:
        ranked[c] = ranked[src] * unit
    muw = hyper['w'][2]
    ranked['WIN_u'] = (ranked.WIN - muw) * unit * 100
    ranked['rawWIN_u'] = (ranked.raw_WIN - muw) * unit * 100
    pb = ranked.p_balls.replace(0, np.nan)
    nb = ranked.n_balls.replace(0, np.nan)
    if role == 'bat':
        ranked['p_SR'] = 100 * ranked.p_runs / pb
        ranked['p_xSR'] = 100 * ranked.p_xr / pb
        ranked['n_SR'] = 100 * ranked.n_runs / nb
        ranked['p_avg'] = ranked.p_runs / ranked.p_outs.replace(0, np.nan)
        ranked['n_avg'] = ranked.n_runs / ranked.n_outs.replace(0, np.nan)
        ranked['p_bpd'] = pb / ranked.p_outs.replace(0, np.nan)
        ranked['p_xbpd'] = pb / ranked.p_xo
        ranked['p_bnd%'] = 100 * ranked.p_bnd / pb
        ranked['p_dot%'] = 100 * ranked.p_dots / pb
    else:
        ranked['p_econ'] = 6 * ranked.p_runs / pb
        ranked['p_xecon'] = 6 * ranked.p_xr / pb
        ranked['n_econ'] = 6 * ranked.n_runs / nb
        ranked['p_sr'] = pb / ranked.p_outs.replace(0, np.nan)
        ranked['p_xsr'] = pb / ranked.p_xo
        ranked['n_sr'] = nb / ranked.n_outs.replace(0, np.nan)
        ranked['p_dot%'] = 100 * ranked.p_dots / pb
        ranked['p_bnd%'] = 100 * ranked.p_bnd / pb
        ranked['p_wkts'] = ranked.p_outs
    ranked['ipl_share'] = ranked.ipl_balls / ranked.all_balls
    ranked['overseas'] = ~ranked.country.fillna('India').str.startswith('India')
    results[role] = ranked
    hy = {k: [float(x) for x in v] for k, v in hyper.items()}
    rel = {f'median_rel_{c}': float(ranked.loc[ref, f'p_rel{c}'].median()) for c in ['1', '2']}
    summaries[role] = dict(hyper=hy, **rel, n_players=int(len(r)), n_eligible=int(len(ref)), n_rated=int(len(ranked)),
                           n_pool=int(len(pool)), min_pressure_balls=min_pb, weights=BASE_W, pz_quantile=PZ_Q,
                           avail_counts=ranked.avail.value_counts().to_dict())
    ranked.index.name = 'pid'
    ranked.to_csv(f'{W}/ratings_{role}{TAG}.csv')
    if not TAG:
        pm.to_parquet(f'{W}/pm_{role}.parquet', index=False)
    show = ['name', 'country', 'avail', 'rank', 'pool_rank', 'RUPI', 'PV_u', 'NV_u', 'RES_u', 'CON', 'p_n',
            'p_matches', 'boot_top10', 'pool_top10', 'emerging']
    print(role.upper(), summaries[role]['avail_counts'], 'eligible', len(ref), 'rated', len(ranked), 'pool', len(pool))
    print(ranked[ranked.avail == 'Auction pool'][show].head(24).round(2).to_string())

if not TAG:
    team_str.round(2).to_csv(f'{W}/team_strength.csv')
    b[['match_id', 'innings', 'seq', 'PI', 'PI_pct', 'pressure', 'extreme', 'SQ_board', 'SQ_wkts']].to_parquet(f'{W}/pressure_flags.parquet', index=False)
    json.dump(summaries, open(f'{W}/ratings_summary.json', 'w'), indent=1, default=float)

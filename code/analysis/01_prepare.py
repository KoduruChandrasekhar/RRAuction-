"""Step 1 - build a clean ball-level state table from the RR data pack.
Usage: python -I analysis/01_prepare.py <data_dir> <out_dir>"""
import sys, numpy as np, pandas as pd
D, OUT = sys.argv[1], sys.argv[2]
b = pd.read_csv(f'{D}/ball_by_ball.csv', low_memory=False)
m = pd.read_csv(f'{D}/matches.csv')
c = pd.read_csv(f'{D}/competitions.csv')
m = m.merge(c, on='comp_id', how='left')

ASSOC = {24, 26, 89, 272, 282, 283, 311, 312, 313, 315, 3886}          # associate nations
A_TEAMS = {222, 341, 3093, 3674, 4444}                                   # 'A' sides
IPL_COMPS = {1358, 1715, 2193, 2725, 3298}
EMERGING = {2670, 3289}

# ---------- match-level info ----------
m['match_date'] = pd.to_datetime(m['match_date'])
m['no_result'] = m['result'].str.contains('No Result|Abandon', case=False, na=False) | m['win_team_id'].isna()
m['tie'] = m['result'].str.contains('Tie|super over', case=False, na=False)
m['comp_type'] = np.where(m.comp_id.isin(IPL_COMPS), 'IPL', np.where(m.comp_id.isin(EMERGING), 'Emerging', 'T20I'))
m['year'] = m.match_date.dt.year
m['env'] = np.where(m.comp_type == 'IPL', 'IPL-' + m.year.astype(str), m.comp_type + '-' + m.year.astype(str))
m['knockout'] = (m.knock_out == 1) | m.stage.isin(['Final', 'Qualifier', 'Eliminator', 'Semi Final'])
mcols = ['match_id', 'comp_id', 'comp_name', 'comp_type', 'env', 'season', 'year', 'match_date', 'stage', 'knockout',
         'venue_id', 'home_team_id', 'home_team', 'away_team_id', 'away_team', 'win_team_id', 'result', 'no_result', 'tie',
         'first_inn_revised_overs', 'second_inn_revised_overs', 'revised_target']

b = b[b.innings.isin([1, 2])].copy()                      # super overs excluded
b = b.merge(m[mcols], on='match_id', how='inner')
b = b[~b.no_result].copy()

# ---------- delivery semantics ----------
b['wide'] = b.extras_type.eq('wide')
b['noball'] = b.extras_type.isin(['Overstep', 'other'])
b['legal'] = ~(b.wide | b.noball)
b['team_runs'] = b.runs + b.extras1_runs + b.extras_2_runs
b['bat_runs'] = b.runs.where(~b.wide, 0)
b['bowl_runs'] = b.runs + np.where(b.wide | b.noball, b.extras1_runs, 0)
b['batter_ball'] = ~b.wide                                     # balls faced (no-balls count, wides don't)
b['wkt'] = b.is_wicket.astype(bool)
b['bowler_wkt'] = b.wkt & b.dismissal_type.isin(['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket'])
b['striker_out'] = b.wkt & (b.dismissed_player_id == b.striker_id) & ~b.dismissal_type.eq('Retire Out')
b['is_boundary'] = b.is_four | b.is_six
b['is_dot'] = b.legal & (b.team_runs == 0)

# ---------- innings state BEFORE each ball ----------
b = b.sort_values(['match_id', 'innings', 'overid', 'over_ball_num', 'innings_ball_num']).reset_index(drop=True)
g = b.groupby(['match_id', 'innings'], sort=False)
b['seq'] = g.cumcount()
b['runs_before'] = g.team_runs.cumsum() - b.team_runs
b['wkts_before'] = g.wkt.cumsum() - b.wkt.astype(int)
b['legal_before'] = g.legal.cumsum() - b.legal.astype(int)
b['wkts_before'] = b.wkts_before.clip(upper=9)

# innings length (balls) incl. rain-revised matches
tot = np.where(b.innings == 1, b.first_inn_revised_overs, b.second_inn_revised_overs)
b['inn_balls'] = pd.Series(tot).fillna(120).clip(upper=120).values
b['balls_left'] = (b.inn_balls - b.legal_before).clip(lower=1)
b['over_no'] = (b.legal_before // 6 + 1).clip(upper=20)
b['phase'] = pd.cut(b.over_no, [0, 6, 15, 20], labels=['Powerplay', 'Middle', 'Death']).astype(str)

inn1 = b[b.innings == 1].groupby('match_id').team_runs.sum().rename('first_inn_total')
b = b.merge(inn1, on='match_id', how='left')
b['target'] = np.where(b.revised_target.notna(), b.revised_target, b.first_inn_total + 1)
b.loc[b.innings == 1, 'target'] = np.nan
b['runs_req'] = b.target - b.runs_before
b['rrr'] = b.runs_req * 6 / b.balls_left
b['crr'] = np.where(b.legal_before > 0, b.runs_before * 6 / b.legal_before.clip(lower=1), np.nan)
b['bat_team_won'] = (b.batting_team_id == b.win_team_id).astype(float)

# opposition quality weight (who you performed AGAINST)
def qual(team):
    return np.where(team.isin(ASSOC), 0.5, np.where(team.isin(A_TEAMS), 0.6, 1.0))
b['q_batter'] = qual(b.bowling_team_id)                       # batter judged by bowling attack faced
b['q_bowler'] = qual(b.batting_team_id)
b.loc[b.comp_type == 'Emerging', ['q_batter', 'q_bowler']] = np.minimum(b.loc[b.comp_type == 'Emerging', ['q_batter', 'q_bowler']], 0.6)
# recency weight: half-life 3 years back from latest match in the pack
ref = b.match_date.max()
b['w_recency'] = 0.5 ** ((ref - b.match_date).dt.days / (365.25 * 3))

# environment scoring levels (league context)
env = b.groupby('env').agg(env_rpb=('team_runs', lambda s: s.sum()), env_legal=('legal', 'sum'),
                           env_wkts=('wkt', 'sum'))
env['env_rpb'] = env.env_rpb / env.env_legal                  # runs per legal ball
env['env_wpb'] = env.env_wkts / env.env_legal
first_tot = b[b.innings == 1].groupby(['env', 'match_id']).team_runs.sum().groupby('env').mean().rename('env_par')
env = env.join(first_tot)
b = b.merge(env[['env_rpb', 'env_wpb', 'env_par']], left_on='env', right_index=True, how='left')
print(env.round(3))
print('balls', len(b), 'matches', b.match_id.nunique())
b.to_parquet(f'{OUT}/balls.parquet', index=False)
m.to_parquet(f'{OUT}/matches.parquet', index=False)

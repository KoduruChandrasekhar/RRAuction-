"""Step 10 - build the interactive RUPI dashboard (single self-contained HTML file).
Usage: python -I analysis/10_dashboard.py <work_dir> <out_html> [version_label]"""
import sys, os, json, numpy as np, pandas as pd
W, OUT = sys.argv[1], sys.argv[2]
VERSION = sys.argv[3] if len(sys.argv) > 3 else 'v2 · post-retention · 18 Nov 2026'
IDX = sys.argv[4] if len(sys.argv) > 4 else 'NeRRve'
HERE = os.path.dirname(os.path.abspath(__file__))
I = json.load(open(f'{W}/insights.json'))
ROB = json.load(open(f'{W}/robustness.json'))
SUM = json.load(open(f'{W}/ratings_summary.json'))
GRID = json.load(open(f'{W}/chase_grid.json'))
PR = json.load(open(os.path.join(HERE, 'prices.json'), encoding='utf-8'))
AVC = json.load(open(os.path.join(HERE, 'availability_config.json'), encoding='utf-8'))
NOTES = json.load(open(os.path.join(HERE, 'player_notes.json'), encoding='utf-8'))


def clean(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


players = {}
for role in ['bat', 'bowl']:
    f = pd.read_csv(f'{W}/ratings_{role}.csv', index_col=0)
    stat_cols = ['p_SR', 'p_xSR', 'n_SR', 'p_avg', 'p_bpd', 'p_xbpd', 'p_bnd%', 'p_dot%'] if role == 'bat' else \
                ['p_econ', 'p_xecon', 'n_econ', 'p_sr', 'p_xsr', 'n_sr', 'p_dot%', 'p_bnd%', 'p_wkts']
    rows = []
    for pid, r in f.iterrows():
        pr = PR['players'].get(r['name'])
        n = NOTES.get(r['name'], {})
        rows.append({k: clean(v) for k, v in dict(
            id=int(pid), name=r['name'], country=r['country'] if isinstance(r['country'], str) else '–', avail=r['avail'],
            note=r['avail_note'], team=r['team27'] if isinstance(r['team27'], str) else None,
            rank=r['rank'], pool=r['pool_rank'], riser=r['riser_rank'], RUPI=r['RUPI'], zPV=r['zPV'], zCON=r['zCON'],
            zRES=r['zRES'], zWIN=r['zWIN'], PV=r['PV_u'], NV=r['NV_u'], RES=r['RES_u'], WIN=r['WIN_u'], CON=r['CON'],
            rawPV=r['rawPV_u'], rawNV=r['rawNV_u'], pn=r['p_n'], pm=r['p_matches'], b10=r['boot_top10'], p10=r['pool_top10'],
            plo=r['pool_lo'], phi=r['pool_hi'], blo=r['boot_lo'], bhi=r['boot_hi'], elig=bool(pd.notna(r['rank'])),
            ovs=bool(r['overseas']), iplok=bool(r.get('ipl_ok', False)), ipl=r['ipl_share'], emg=bool(r['emerging']), last=str(r['last_date'])[:10],
            st={c.replace('%', 'pct'): clean(r[c]) for c in stat_cols},
            price=[pr['lo'], pr['hi'], pr['basis']] if pr else None, role_note=n.get('role'), fit=n.get('fit'), risk=n.get('risk')).items()})
    players[role] = rows

bb = pd.read_parquet(f'{W}/balls_scored.parquet', columns=['innings', 'legal_before', 'wkts_before', 'env', 'env_rpb'])
fl = pd.read_parquet(f'{W}/pressure_flags.parquet', columns=['PI'])
calc = dict(expwk=bb[bb.innings == 2].groupby('legal_before').wkts_before.mean().reindex(range(120)).ffill().round(3).tolist(),
            piq=np.quantile(fl.PI, np.linspace(0, 1, 201)).round(4).tolist(),
            envRate=round(float(bb[bb.env == 'IPL-2026'].env_rpb.iloc[0] * 6), 3))
data = dict(version=VERSION, players=players, calc=calc, plan_core=PR.get("plan_core", {}), plan=PR.get("plan", []), auction=PR.get("auction", {}), young=PR.get("young", []), plan_alt=PR.get("plan_alternatives", {}), news=AVC.get('pending_news', []), deep=I['deep'], heat=I['chase_heat'], heatAxes=I['chase_heat_axes'],
            wpa=I['wpa_share'], profile=I['profile'], close=I['close_record'], closeShare=I['close_vs_onesided'],
            ko=I['knockout_press'], split=I['split_half'], rrRanks=I['rr_ranks'], rrSquad=I['rr_squad'], radar=I['release_radar'], scorecard=I.get('radar_scorecard', {}),
            purse=I['purse'], teamBat=I['team_bat'], teamBowl=I['team_bowl'], data=I['data'], wp=I['wp_metrics'],
            rob={r: {k: ROB[r][k]['overlap_top10'] for k in ['league', 'pool']} for r in ['bat', 'bowl']},
            weights=SUM['bat']['weights'], grid=GRID, envs=I['env_table'], wkcost=I['wkt_cost_phase'])
tpl = open(os.path.join(HERE, 'dashboard_template.html'), encoding='utf-8').read()
tpl = tpl.replace('__IDX__', IDX)
import html as _h
_fl = {
    role: [
        r['name'] for r in sorted(
            (p for p in players[role] if p['pool'] is not None),
            key=lambda p: p['pool']
        )[:10]
    ]
    for role in ['bat', 'bowl']
}
_li = lambda xs: ''.join(f'<li>{_h.escape(x)}</li>' for x in xs)
_plan = ''.join(f'<li><b>{_h.escape(x["name"])}</b> ({_h.escape(x["slot"])}, max ₹{x["max"]:g} cr)</li>' for x in PR.get('plan', []))
static = (f'<h2>{IDX}: picks at a glance</h2><p class="small muted">This summary is shown when the interactive view cannot run '
          '(for example in a file preview with JavaScript turned off). Open the file in Chrome or Edge, or use the live link, for the full dashboard.</p>'
          f'<div class="cols"><div><h3>Top 10 batters (by {IDX})</h3><ol>{_li(_fl.get("bat", []))}</ol></div>'
          f'<div><h3>Top 10 bowlers (by {IDX})</h3><ol>{_li(_fl.get("bowl", []))}</ol></div>'
          f'<div><h3>Auction buy plan (₹{I["purse"]["available"]:.2f} cr, 6 slots)</h3><ul>{_plan}</ul></div></div>')
tpl = tpl.replace('<!--__STATIC__-->', static)
html = tpl.replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':'), default=clean))
open(OUT, 'w', encoding='utf-8').write(html)
print('dashboard written', OUT, round(len(html) / 1e6, 2), 'MB')

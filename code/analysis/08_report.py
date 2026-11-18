"""Step 8 - build the submission report (HTML -> PDF via 09_render.py).
Every number is read from the pipeline outputs, so re-running after the 17 Nov retention lists refreshes it.
Usage: python -I analysis/08_report.py <work_dir> <out_html> [author] [version_label]"""
import sys, os, io, json, re, html, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')
W, OUT = sys.argv[1], sys.argv[2]
AUTHOR = sys.argv[3] if len(sys.argv) > 3 else 'Koduru Chandrasekhar'
VERSION = sys.argv[4] if len(sys.argv) > 4 else 'v2 · post-retention build · 18 Nov 2026'
IDX = sys.argv[5] if len(sys.argv) > 5 else 'NeRRve'
HERE = os.path.dirname(os.path.abspath(__file__))
bat = pd.read_csv(f'{W}/ratings_bat.csv', index_col=0)
bowl = pd.read_csv(f'{W}/ratings_bowl.csv', index_col=0)
I = json.load(open(f'{W}/insights.json'))
ROB = json.load(open(f'{W}/robustness.json'))
SUM = json.load(open(f'{W}/ratings_summary.json'))
PR = json.load(open(os.path.join(HERE, 'prices.json'), encoding='utf-8'))
NOTES = json.load(open(os.path.join(HERE, 'player_notes.json'), encoding='utf-8'))
AVC = json.load(open(os.path.join(HERE, 'availability_config.json'), encoding='utf-8'))
E = html.escape

PINK, NAVY, BLUE, INK, MUTED, LINE, GOLD = '#E2077E', '#1B2A6B', '#2A45B8', '#14172B', '#6B7085', '#E6E7EE', '#C98500'
GRAYPT = '#B9BCC8'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'font.size': 9, 'axes.edgecolor': '#C3C5D0',
                     'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.8})


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index('<svg'):]
    s = re.sub(r"font-family:[^;\"]*", "font-family:Inter,'Segoe UI',sans-serif", s)
    s = re.sub(r'font-family="[^"]*"', 'font-family="Inter,Segoe UI,sans-serif"', s)
    s = re.sub(r'<svg([^>]*?) width="[^"]*" height="[^"]*"', r'<svg\1 width="100%" height="100%"', s, count=1)
    return s


def f1(x, d=1, plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '–'
    s = f'{x:+.{d}f}' if plus else f'{x:.{d}f}'
    return s.replace('-', '−')


def pct(x, d=0):
    return '–' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{100 * x:.{d}f}%'


def prices(name):
    p = PR['players'].get(name)
    return (f"₹{p['lo']:g}–{p['hi']:g} cr" if p else 'tbc'), (p['basis'] if p else 'Price band to be set after the 3 Dec auction list')


pool_b = bat[bat.pool_rank.notna()].sort_values('pool_rank')
pool_w = bowl[bowl.pool_rank.notna()].sort_values('pool_rank')
top_b, top_w = pool_b.head(10), pool_w.head(10)
lg_b = bat[bat['rank'].notna()].sort_values('rank')
lg_w = bowl[bowl['rank'].notna()].sort_values('rank')
ris_b = bat[bat.riser_rank.notna()].sort_values('riser_rank').head(5)
ris_w = bowl[bowl.riser_rank.notna()].sort_values('riser_rank').head(5)
D = I['data']
WS = I['wpa_share']


def usage(role, pid):
    return (I['deep'][role].get(str(int(pid)), {}) or {}).get('usage', {}) or {}


def pick(R, role, cond, exclude=()):
    for pid, r in R.iterrows():
        if r['name'] not in exclude and cond(usage(role, pid)):
            return r
    return R.iloc[0]


def ORD(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


rrk = I['rr_ranks']
death_bowl = pick(pool_w, 'bowl', lambda u: u.get('Death', 0) >= 0.45)
ctrl_bowl = pick(pool_w, 'bowl', lambda u: u.get('Death', 0) < 0.45, {death_bowl['name']})
finisher = pick(pool_b, 'bat', lambda u: u.get('Death', 0) >= 0.4)
alt_death = [r['name'] for pid, r in pool_w.iterrows() if usage('bowl', pid).get('Death', 0) >= 0.4 and r['name'] != death_bowl['name']][:2]
_ind = pd.concat([pool_b, pool_w])
_ind = _ind[_ind.country.fillna('').str.startswith('India')].sort_values('RUPI', ascending=False)
indian = _ind.iloc[0] if len(_ind) else None
PH = {'Powerplay': 'powerplay', 'Middle': 'middle-overs', 'Death': 'death'}
SIDE = {'bat': 'batting', 'bowl': 'bowling'}
strengths = [f"{PH[p]} {SIDE[sd]} ({ORD(rrk[sd][p])})" for sd in ['bat', 'bowl'] for p in ['Powerplay', 'Middle', 'Death'] if rrk[sd][p] <= 3]
gaps = sorted([(rrk[sd][p], f"{PH[p]} {SIDE[sd]}") for sd in ['bat', 'bowl'] for p in ['Powerplay', 'Middle', 'Death'] if rrk[sd][p] >= 6], reverse=True)
NEED = {'death bowling': 'a death-overs bowler who holds his nerve', 'death batting': 'a finisher for overs 16-20',
        'middle-overs bowling': 'a middle-overs control bowler', 'middle-overs batting': 'a middle-order chaser',
        'powerplay bowling': 'a new-ball bowler', 'powerplay batting': 'top-order cover'}
rad_top = [x for x in I['release_radar'] if x['release_prob'] in ('High', 'Medium') and x['RUPI'] >= 60 and x['rank']][:2]
rr_best_bowl = max((x for x in I['rr_squad']['bowl'] if x['rank']), key=lambda x: x['RUPI'], default=None)
rr_best_bat = [x for x in I['rr_squad']['bat'] if x['rank'] and x['rank'] <= 10]


def role_of(r, fallback):
    return NOTES.get(r['name'], {}).get('role', fallback)


def is_spin(r):
    return 'spin' in NOTES.get(r['name'], {}).get('role', '').lower()


spin_bowl = next((r for _, r in pool_w.iterrows() if is_spin(r)), pool_w.iloc[0])
mid_bats = [r for pid, r in pool_b.iterrows() if (usage('bat', pid).get('Middle', 0) + usage('bat', pid).get('Death', 0)) >= 0.5]
FINAL = PR.get('final_lists', {'bat': list(pool_b['name'][:10]), 'bowl': list(pool_w['name'][:10])})

# ============================================================ CHARTS


def chart_heat():
    df = pd.DataFrame(I['chase_heat'])
    runs, balls = I['chase_heat_axes']['runs'], I['chase_heat_axes']['balls'][::-1]
    M = np.full((len(runs), len(balls)), np.nan)
    for r in df.itertuples():
        if r.rb in runs and r.bb in balls:
            M[runs.index(r.rb), balls.index(r.bb)] = r.share
    cmap = LinearSegmentedColormap.from_list('p', ['#FFF4F9', '#F9B8D8', '#EE5AA6', PINK, '#8E0450'])
    cmap.set_bad('#F4F5F8')
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect='auto', origin='lower')
    ax.set_xticks(range(len(balls))); ax.set_xticklabels(balls, fontsize=8)
    ax.set_yticks(range(len(runs))); ax.set_yticklabels(runs, fontsize=8)
    ax.set_xlabel('Balls left in the chase  →  end of match', fontsize=9)
    ax.set_ylabel('Runs still needed', fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]) and (M[i, j] >= 0.6 or M[i, j] <= 0.05):
                ax.text(j, i, f'{M[i, j] * 100:.0f}', ha='center', va='center', fontsize=6.5,
                        color='white' if M[i, j] >= 0.6 else MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('% of balls in the pressure zone', fontsize=8, color=MUTED)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7, length=0)
    return svg(fig)


def chart_scatter(R, top, role):
    el = R[R['rank'].notna()]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.axhline(0, color=LINE, lw=1, zorder=0); ax.axvline(0, color=LINE, lw=1, zorder=0)
    lim = [min(el.NV_u.min(), el.PV_u.min()) - 1, max(el.NV_u.max(), el.PV_u.max()) + 1]
    ax.plot(lim, lim, color='#C3C5D0', lw=0.8, ls=(0, (2, 3)), zorder=0)
    rr = el.iloc[0:0]; oth = el[~el.index.isin(top.index)]
    ax.scatter(oth.NV_u, oth.PV_u, s=16, color=GRAYPT, edgecolor='white', lw=0.6, zorder=2, label='Other rated players')
    t = top[top.index.isin(el.index)]
    ax.scatter(t.NV_u, t.PV_u, s=42, color=PINK, edgecolor='white', lw=1.2, zorder=4, label='Our picks')
    placed = []
    for pid, r in t.sort_values('pool_rank').iterrows():
        lab = r['name'].split()[-1]
        dy = 3
        for (px, py) in placed:   # nudge labels that would collide
            if abs(px - r.NV_u) < 3.5 and abs(py - (r.PV_u)) < 1.6:
                dy -= 9
        placed.append((r.NV_u, r.PV_u))
        ax.annotate(lab, (r.NV_u, r.PV_u), xytext=(5, dy), textcoords='offset points', fontsize=7.5, color=INK, fontweight='bold')
    for pid, r in rr.sort_values('PV_u', ascending=False).head(3).iterrows():
        ax.annotate(r['name'].split()[-1], (r.NV_u, r.PV_u), xytext=(4, -9), textcoords='offset points', fontsize=7, color=BLUE)
    u = 'per 100 balls' if role == 'bat' else 'per 4 overs'
    ax.set_xlabel(f'Normal-situation value (runs above expectation {u})', fontsize=8.5)
    ax.set_ylabel(f'Pressure value ({u})', fontsize=8.5)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.grid(color='#F0F1F5', lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc='upper left', handletextpad=0.3)
    return svg(fig)


def chart_boot(top):
    t = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    y = np.arange(len(t))
    ax.hlines(y, t.pool_lo, t.pool_hi, color='#F6B3D4', lw=6, capstyle='round', zorder=1)
    ax.scatter(t.pool_med, y, color=PINK, s=34, zorder=3, edgecolor='white', lw=1.2)
    ax.scatter(t.pool_rank, y, color=NAVY, s=14, marker='D', zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([n for n in t['name']], fontsize=8, color=INK)
    ax.set_xlabel('Rank inside the auction pool across 300 bootstrap re-samples', fontsize=7.5)
    ax.invert_xaxis(); ax.set_xlim(max(25, t.pool_hi.max() + 1), 0.5)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='#F0F1F5', lw=0.6); ax.set_axisbelow(True)
    return svg(fig)


def chart_shrink():
    df = pd.DataFrame(I['shrink_demo']['bat']).sort_values('p_n')
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for r in df.itertuples():
        ax.plot([r.p_n, r.p_n], [r.rawPV_u, r.PV_u], color='#F3C6DD', lw=0.8, zorder=1)
    ax.scatter(df.p_n, df.rawPV_u, s=12, color=GRAYPT, zorder=2, label='Raw pressure value')
    ax.scatter(df.p_n, df.PV_u, s=14, color=PINK, zorder=3, label='After Empirical-Bayes shrinkage')
    ax.axhline(0, color='#C3C5D0', lw=0.8)
    ax.set_xlabel('Pressure balls faced (evidence)', fontsize=8.5); ax.set_ylabel('Runs above expectation / 100 balls', fontsize=8.5)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(color='#F0F1F5', lw=0.6); ax.set_axisbelow(True)
    return svg(fig)


def chart_close():
    df = pd.DataFrame(I['close_record'])
    df['rate'] = df.close_won / df.close
    df = df.sort_values('rate')
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    y = np.arange(len(df))
    cols = [PINK if t == 'Rajasthan Royals' else '#C9CCD8' for t in df.team]
    ax.barh(y, df.rate * 100, color=cols, height=0.55)
    for i, r in enumerate(df.itertuples()):
        ax.text(r.rate * 100 + 1.5, i, f'{r.close_won}/{r.close}', va='center', fontsize=7.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(df.team, fontsize=7.5, color=INK)
    ax.set_xlabel('Close IPL games won, 2025–26 (%)', fontsize=8)
    ax.set_xlim(0, 100)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    return svg(fig)


def spark(trend, w=150, h=34):
    pts = [(t['year'], t['pv']) for t in trend if t['pv'] is not None]
    if len(pts) < 2:
        return '<span class="muted small">season trend: limited data</span>'
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    lo, hi = min(min(ys), 0), max(max(ys), 0)
    rng_ = (hi - lo) or 1
    X = lambda x: 6 + (x - 2022) / 4 * (w - 12)
    Y = lambda y: h - 6 - (y - lo) / rng_ * (h - 12)
    path = ' '.join(f'{"M" if i == 0 else "L"}{X(x):.1f},{Y(y):.1f}' for i, (x, y) in enumerate(pts))
    dots = ''.join(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.6" fill="{PINK}" stroke="white" stroke-width="1"/>' for x, y in pts)
    zero = f'<line x1="6" x2="{w - 6}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="#D5D7E0" stroke-width="0.8"/>'
    lab = ''.join(f'<text x="{X(y):.1f}" y="{h + 9}" font-size="7" fill="{MUTED}" text-anchor="middle">{str(y)[2:]}</text>' for y in range(2022, 2027))
    return f'<svg viewBox="0 0 {w} {h + 11}" width="{w}" height="{h + 11}">{zero}<path d="{path}" fill="none" stroke="{PINK}" stroke-width="1.6"/>{dots}{lab}</svg>'


# ============================================================ HTML helpers
page_no = [0]


def page(kicker, title, body, sub=None, cls=''):
    page_no[0] += 1
    s = f'<p class="sub">{sub}</p>' if sub else ''
    return f'''<section class="page {cls}"><div class="kicker">{kicker}</div><h2>{title}</h2>{s}<div class="body">{body}</div>
<footer><span>SupeRR Selector S2 · Round 1 · Reliability Under Pressure Index (RUPI)</span><span>{E(VERSION)}</span><span>{page_no[0]:02d}</span></footer></section>'''


def tile(v, l, note=''):
    return f'<div class="tile"><div class="tv">{v}</div><div class="tl">{l}</div>{f"<div class=tn>{note}</div>" if note else ""}</div>'


def bar(v, lo=-2.5, hi=2.5):
    v = 0 if v is None or np.isnan(v) else max(lo, min(hi, v))
    mid = 50
    w = abs(v) / (hi - lo) * 100
    left = mid if v >= 0 else mid - w
    col = PINK if v >= 0 else '#9AA0B4'
    return f'<span class="zbar"><span class="zmid"></span><span class="zfill" style="left:{left:.1f}%;width:{w:.1f}%;background:{col}"></span></span>'


def status_chip(a):
    m = {'Auction pool': 'pool', 'Watch list': 'watch', 'Contracted': 'locked', 'RR squad': 'rr', 'Unavailable': 'out', 'Doubtful': 'watch', 'Unlikely': 'out'}
    return f'<span class="chip {m.get(a, "")}">{E(a)}</span>'


def risk_flags(r):
    flags = []
    if r['country'] == 'Bangladesh':
        flags.append('political risk')
    if r['name'] == 'Harry Brook':
        flags.append('must register')
    if r.get('ipl_share', 0) == 0:
        flags.append('no IPL sample')
    elif r.get('ipl_share', 0) < 0.1:
        flags.append('little IPL sample')
    if r['p_n'] < 200:
        flags.append('smaller sample')
    return ', '.join(flags) if flags else '–'


def list_table(T, role):
    unit = '/100b' if role == 'bat' else '/4ov'
    rows = ''
    for _, r in T.iterrows():
        price, _ = prices(r['name'])
        if role == 'bat':
            core = f'<td class="num">{f1(r.p_SR, 0)}</td><td class="num muted">{f1(r.p_xSR, 0)}</td><td class="num">{f1(r.p_bpd, 0)}</td>'
        else:
            core = f'<td class="num">{f1(r.p_econ, 2)}</td><td class="num muted">{f1(r.p_xecon, 2)}</td><td class="num">{f1(r.p_sr, 0)}</td>'
        rows += f'''<tr><td><b>{E(r["name"])}</b><div class="muted small">{E(str(r.country))} · league rank {("#" + str(int(r["rank"]))) if pd.notna(r["rank"]) else "n/a (smaller sample)"}</div></td>
<td class="num big">{f1(r.RUPI, 0)}</td><td class="num">{f1(r.PV_u, 1, True)}</td><td class="num muted">{f1(r.NV_u, 1, True)}</td>{core}
<td class="num">{pct(r.CON)}</td><td class="num">{int(r.p_n)}</td><td class="small">{price}</td><td class="small muted">{E(NOTES.get(r["name"], {}).get("role", ""))}</td></tr>'''
    if role == 'bat':
        h3 = '<th>SR<br>press.</th><th>xSR</th><th>Balls /<br>dismissal</th>'
    else:
        h3 = '<th>Econ<br>press.</th><th>xEcon</th><th>Balls /<br>wicket</th>'
    return f'''<table class="lt"><thead><tr><th>Player</th><th>RUPI</th><th>Pressure<br>value {unit}</th><th>Normal<br>value</th>{h3}
<th>Consist-<br>ency</th><th>Pressure<br>balls</th><th>Price band</th><th>Role</th></tr></thead><tbody>{rows}</tbody></table>'''


def auto_profile(pid, role, R):
    d = I['deep'][role].get(str(int(pid)))
    if not d:
        return ''
    use = d['usage']
    main = max(use, key=use.get) if use else '–'
    best = [(k, v) for k, v in d['splits'].items() if v is not None and d['split_n'].get(k, 99) >= 40]
    best = sorted(best, key=lambda x: -x[1])[:2]
    b_txt = ', '.join(f'{k} ({f1(v, 1, True)})' for k, v in best)
    return f"{pct(use.get(main, 0))} of pressure balls in the {main.lower()}; {pct(d['chase_share'])} in second innings. Best pressure contexts: {b_txt}."


def card(pid, R, role):
    r = R.loc[pid]
    d = I['deep'][role].get(str(int(pid)), {})
    n = NOTES.get(r['name'], {})
    price, basis = prices(r['name'])
    ASSOC = {'Hong Kong', 'United Arab Emirates', 'Nepal', 'Oman', 'Netherlands', 'Scotland', 'Canada', 'Singapore', 'Kuwait', 'United States Of America', 'Japan'}
    mom = [m for m in d.get('moments', []) if m['opp'] not in ASSOC][:1]
    mtxt = ''
    if mom:
        m = mom[0]
        fig = f"{m['runs']} off {m['balls']}" if role == 'bat' else f"{m['wk']}/{m['runs']} in {m['balls'] // 6}.{m['balls'] % 6} ov"
        mtxt = f"<div class='mom'><b>Biggest pressure moment:</b> {fig} vs {E(m['opp'])}, {E(m['comp'])} ({m['date']}) · +{m['wpa'] * 100:.0f}% win prob.</div>"
    zs = ''.join(f'<div class="zrow"><span>{lab}</span>{bar(r[f"z{k}"])}<span class="num small">{f1(r[f"z{k}"], 2, True)}</span></div>'
                 for k, lab in [('PV', 'Pressure value'), ('CON', 'Consistency'), ('RES', 'Resilience'), ('WIN', 'Win impact')])
    return f'''<div class="card"><div class="ch"><div><b>{E(r["name"])}</b>
<div class="muted small">{E(n.get("role", "")) or E(str(r.country))} · {E(str(r.country))}</div></div><div class="crupi">{f1(r.RUPI, 0)}<span>RUPI</span></div></div>
<div class="cgrid"><div>{zs}<div class="spark">{spark(d.get("trend", []))}<div class="muted tiny">pressure value by season</div></div></div>
<div><p class="small"><b>RR fit.</b> {E(n.get("fit", "")) or auto_profile(pid, role, R)}</p>
{"" if not n.get("fit") else f'<p class="tiny muted">{auto_profile(pid, role, R)}</p>'}
<p class="small"><b>Price:</b> {price} <span class="muted">({E(basis)})</span><br><b>Risk:</b> {E(n.get("risk", risk_flags(r)))}</p>{mtxt}</div></div></div>'''


# ============================================================ PAGES
P = []
# ---- 1 cover
P.append(f'''<section class="page cover"><div class="cv-l">
<div class="kicker light">SupeRR Selector · Season 2 · Round 1 submission</div>
<h1>Pressure-proof.</h1><h1 class="pink">Who RR should back at the IPL 2027 auction.</h1>
<p class="lead">A data system that rates every batter and bowler on how they perform when the match is on the line, compared with their own normal level. It then filters the ratings down to players Rajasthan Royals can actually buy in Goa on 16 December 2026.</p>
<div class="cv-meta"><div><b>{E(AUTHOR)}</b><span>Analyst</span></div><div><b>{D['balls']:,}</b><span>deliveries modelled</span></div>
<div><b>{D['matches']}</b><span>IPL + T20I matches, {D['first'][:4]}–{D['last'][:4]}</span></div><div><b>{SUM['bat']['n_eligible'] + SUM['bowl']['n_eligible']}</b><span>players fully rated</span></div></div>
<div class="ver">{E(VERSION)}. Auction availability from the official retention lists (17 Nov 2026); auction on 16 Dec in Goa.</div></div>
<div class="cv-r"><div class="rupi">RUPI</div><div class="rupi-s">Reliability Under Pressure Index</div>
<div class="pill">Pressure = Stakes × Scoreboard squeeze × Wicket squeeze</div><div class="pill">Expected-value models · opposition-adjusted</div>
<div class="pill">Empirical-Bayes shrinkage · 300-sample bootstrap</div><div class="pill">Filtered to the real 2027 auction pool</div></div></section>''')

# ---- 2 exec summary
TOTMAX = sum(x.get('max', 0) for x in PR.get('plan', []))
GROUPS = PR.get('groups', {})
def glist(keys):
    return ''.join(f'<div class="gh">{E(k)}</div><div class="gn">{E(", ".join(GROUPS[k]))}</div>' for k in keys if k in GROUPS)
li_b = glist([k for k in GROUPS if 'Pace' not in k and 'Spin' not in k])
li_w = glist(['Pace', 'Spin'])
rrk = I['rr_ranks']
rrc = next(x for x in I['close_record'] if x['team'] == 'Rajasthan Royals')
body = f'''<div class="g3"><div class="box"><h3>Batters to buy</h3><div class="glist">{li_b}</div></div>
<div class="box"><h3>Bowlers to buy</h3><div class="glist">{li_w}</div><p class="small muted" style="margin-top:10px">How each pick strengthens RR and the young talent watchlist: section 10.</p></div>
<div class="box soft"><h3>What we found</h3><ul class="find">
<li><b>Our pressure zone is where matches are decided.</b> It holds {pct(WS['pressure_balls_share'])} of balls but {pct(WS['pressure_wpa_share'])} of all win-probability movement.</li>
<li><b>RR lose the close ones.</b> RR played the most close IPL games in 2025–26 ({rrc['close']}) and won only {rrc['close_won']}. Under pressure they rank {rrk['bat']['Death']}th of 10 for death batting and {rrk['bowl']['Death']}th for death bowling.</li>
<li><b>"Clutch" is mostly skill plus sample size.</b> Pressure-minus-normal gaps barely repeat (split-half r = {I['split_half']['bat']['resilience']:.2f}), so we shrink hard and reward proven volume.</li>
<li><b>The strategy:</b> boundary power for overs 12–20 to support Ferreira first ({E(', '.join(FINAL['bat'][:3]))}), backup pace for injuries ({E(', '.join(FINAL['bowl'][:3]))}), spin only if a quality option comes cheap. Bid ceilings for the 6 buys total ₹{TOTMAX:.1f} cr of the ₹{I['purse']['available']:.2f} cr purse, leaving a reserve; RR leaves the expensive overseas quicks to the ₹39–47 cr teams (section 11).</li></ul></div></div>
<div class="tiles t4">{tile(f"{D['balls']:,}", 'deliveries modelled ball by ball')}{tile(f"{D['pressure_balls']:,}", 'pressure balls analysed (top 25% by Pressure Index)')}
{tile(str(SUM['bat']['n_rated'] + SUM['bowl']['n_rated']), 'batter and bowler ratings, each with uncertainty')}{tile('9 + 300', 'alternative definitions + bootstrap re-samples tested')}</div>
<p class="note">Every pick can be bought on 16 Dec: no current Royals, no players retained elsewhere, no retired or unavailable players. The pure league-wide ranking is in section 5. Availability reflects the official 17 Nov retention lists.</p>'''
P.append(page('Executive summary', 'The picks to buy at the IPL 2027 auction', body))
bm = ''.join(f'<tr><td><b>{E(x[0])}</b></td><td class="small">{E(x[1])}</td><td class="small" style="white-space:nowrap">{E(x[2])}</td></tr>' for x in PR.get('brief_map', []))
body = f'''<table class="lt fit"><thead><tr><th>The brief asks for</th><th>Our answer</th><th>Where</th></tr></thead><tbody>{bm}</tbody></table>
<p class="note">Dropdown picks: the ten batters and ten bowlers in sections 8 and 9, in the same order as the submission form.</p>'''
P.append(page('Brief → answer', 'Every requirement of the Round 1 brief, and where this report answers it', body))

# ---- 3 approach
steps = [('1', 'Ball states', f"{D['balls']:,} deliveries rebuilt into match states: score, wickets, balls left, target, rain revisions."),
         ('2', 'Win probability', f"Monotonic gradient-boosted models (chase AUC {I['wp_metrics']['wp_inn2']['auc']:.2f}, out-of-fold)."),
         ('3', 'Pressure Index', 'Leverage (expected win-prob swing) × scoreboard squeeze × wicket squeeze × knockout weight.'),
         ('4', 'Expected outcomes', 'Situation-only models give expected runs and wickets per ball; wickets are valued in runs.'),
         ('5', 'Fair value', 'Runs above expectation, adjusted for opposition strength, recency and associate opposition.'),
         ('6', 'RUPI', 'Four Empirical-Bayes pillars (Pressure Value, Consistency, Resilience, Win impact) on a talent scale.'),
         ('7', 'Reliability', '300 match-bootstrap re-samples, 9 alternative definitions and split-half repeatability tests.'),
         ('8', 'Auction lens', 'Availability rules, purse, auction dynamics and RR gap analysis.')]
flow = ''.join(f'<div class="step"><div class="sn">{n}</div><div><b>{t}</b><p>{d}</p></div></div>' for n, t, d in steps)
P.append(page('Methodology overview', 'From 200,000 balls to an auction shortlist in eight transparent steps',
              f'<div class="flow">{flow}</div><p class="note">Everything is reproducible: <code>analysis/run_all</code> rebuilds every table, chart, this PDF and the dashboard from the raw data pack. Public information is used only for auction availability, contracts, prices and the U19 scouting radar.</p>'))

# ---- 4 pressure definition
body = f'''<div class="g2"><div>
<div class="formula">Pressure Index = <span class="pk">Leverage</span><sup>0.5</sup> × <span class="bl">Scoreboard squeeze</span><sup>0.3</sup> × <span class="gd">Wicket squeeze</span><sup>0.2</sup> × Importance</div>
<dl class="defs">
<dt class="pk">Leverage: are the stakes high?</dt><dd>The expected swing in win probability from this ball, computed over every possible outcome (dot, 1, 2, 4, 6, wicket, wide) using our win-probability models. Scaled so an average ball = 1. It captures closeness, balls left, wickets in hand and required rate together, with weights learned from {D['matches']} matches rather than chosen by hand.</dd>
<dt class="bl">Scoreboard squeeze: is the target hard?</dt><dd>Chasing: required rate ÷ that competition-season's normal scoring rate. Setting: the rate needed to reach par ÷ normal rate. This stops a routine first-innings powerplay looking like a crisis.</dd>
<dt class="gd">Wicket squeeze: is the innings fragile?</dt><dd>Wickets down compared with what is normal at that stage of the innings, which captures collapses.</dd>
<dt>Importance</dt><dd>Knockout and final balls are weighted ×1.25.</dd></dl></div>
<div><h3>Why this definition?</h3><ul class="find">
<li><b>Grounded in outcomes.</b> Leverage is the cricket version of baseball's Leverage Index; ESPNcricinfo's Smart Stats pressure index and CricViz's WinViz rest on the same idea.</li>
<li><b>Matches cricket intuition.</b> Pure leverage over-weights first-innings powerplay wickets. The two squeeze terms put the focus back on tight chases, collapses and death overs.</li>
<li><b>One scale for both disciplines.</b> The same ball is pressure for the batter and the bowler, so batting and bowling ratings stay comparable.</li>
<li><b>Pressure zone</b> = the top 25% of balls on one league-wide scale. The extreme zone is the top 10%.</li></ul>
<div class="tiles">{tile(pct(I['zone_by_phase'].get('2|Middle', 0) + I['zone_by_phase'].get('2|Death', 0) + I['zone_by_phase'].get('2|Powerplay', 0)), 'of pressure balls come in the second innings')}
{tile(pct(I['zone_by_phase'].get('1|Death', 0) + I['zone_by_phase'].get('2|Death', 0)), 'are death-over balls')}
{tile(f"{I['profile']['inn2_pressure']['mean_wp'] * 100:.0f}%", 'average chase win-probability inside the zone: live games')}</div></div></div>'''
P.append(page('1 · Defining pressure', 'Pressure = stakes × squeeze, learned from the data rather than hand-picked', body))

# ---- 5 evidence pressure works
pr1, nr1, pr2, nr2 = (I['profile'][k] for k in ['inn1_pressure', 'inn1_normal', 'inn2_pressure', 'inn2_normal'])
body = f'''<div class="g2"><div><div class="tiles t3">{tile(pct(WS['pressure_wpa_share']), f"of all win-probability movement happens in the {pct(WS['pressure_balls_share'])} of balls we call pressure")}
{tile(pct(WS['extreme_wpa_share']), 'of the swing sits in the extreme 10%')}{tile(pct(WS['bottom50_wpa_share']), 'in the calmest half of all balls')}</div>
<div class="tiles t3">{tile(pct(I['close_vs_onesided']['close']), f"of balls are pressure balls in close games (n={I['n_close']})", f"vs {pct(I['close_vs_onesided']['one_sided'])} in one-sided games")}
{tile(pct(I['knockout_press']['knockout']), 'of balls are pressure balls in knockouts', f"vs {pct(I['knockout_press']['league'])} in league games")}
{tile(f"+{(pr2['dismissal_per_100'] / nr2['dismissal_per_100'] - 1) * 100:.0f}%", 'higher dismissal rate per ball in chase pressure', f"{pr2['dismissal_per_100']:.2f} vs {nr2['dismissal_per_100']:.2f} per 100 balls")}</div>
<table class="mini"><thead><tr><th></th><th>1st inns: pressure</th><th>1st inns: normal</th><th>2nd inns: pressure</th><th>2nd inns: normal</th></tr></thead><tbody>
<tr><td>Average |win-prob swing| per ball</td><td>{pr1['avg_abs_wpa']:.2f}%</td><td>{nr1['avg_abs_wpa']:.2f}%</td><td>{pr2['avg_abs_wpa']:.2f}%</td><td>{nr2['avg_abs_wpa']:.2f}%</td></tr>
<tr><td>Dismissals per 100 balls</td><td>{pr1['dismissal_per_100']:.2f}</td><td>{nr1['dismissal_per_100']:.2f}</td><td>{pr2['dismissal_per_100']:.2f}</td><td>{nr2['dismissal_per_100']:.2f}</td></tr>
<tr><td>Run rate</td><td>{pr1['rpo']:.2f}</td><td>{nr1['rpo']:.2f}</td><td>{pr2['rpo']:.2f}</td><td>{nr2['rpo']:.2f}</td></tr>
<tr><td>Dot-ball %</td><td>{pr1['dot_pct']:.1f}</td><td>{nr1['dot_pct']:.1f}</td><td>{pr2['dot_pct']:.1f}</td><td>{nr2['dot_pct']:.1f}</td></tr></tbody></table></div>
<div><h3>Where chase pressure lives</h3><div class="chart">{chart_heat()}</div><p class="small muted">Cells are the share of balls in each chase state that fall in the pressure zone (all competitions). Pressure runs along the diagonal where the chase is still live (required rate around 7–12 an over) and intensifies as balls run out. Chases that are already won or lost (bottom-left and top-right) score zero.</p></div></div>'''
P.append(page('1 · Proof the definition works', 'Pressure balls are a quarter of the game but decide more than half of it', body))

# ---- 6 performance measurement
wc = I['wkt_cost_phase']
body = f'''<div class="g2"><div><div class="formula small">Ball value = (runs − expected runs) − wicket cost × (out − expected out) − opposition adjustment</div>
<dl class="defs"><dt>Expected runs and wickets</dt><dd>Gradient-boosted models trained only on match situation: over, wickets, balls left, score, required rate, competition scoring level and leverage. No player identity is used. They are cross-fitted by match so no player is judged by a model that has seen him. Calibration: actual {I['wp_metrics']['xmodel_check']['bat_runs'][0]:.3f} vs expected {I['wp_metrics']['xmodel_check']['bat_runs'][1]:.3f} runs per ball.</dd>
<dt>Wickets in runs</dt><dd>A run-expectancy model prices each wicket by state: about {wc['1|Powerplay']:.1f} runs in the powerplay, {wc['1|Middle']:.1f} in the middle overs and {wc['1|Death']:.1f} at the death.</dd>
<dt>Batters</dt><dd>Value = scoring above expectation + survival value. Reported per 100 balls.</dd>
<dt>Bowlers</dt><dd>Value = runs saved vs expectation + wicket value. Byes and leg-byes are excluded and wides and no-balls included. Reported per 4 overs.</dd>
<dt>Baseline</dt><dd>The same metric on each player's non-pressure balls, so every player is compared with his own normal level.</dd></dl></div>
<div><h3>Fair comparisons</h3><ul class="find">
<li><b>IPL and T20I together, each judged against its own environment.</b> Expectations use the competition-season scoring rate (IPL 2026: {next(e['rpo'] for e in I['env_table'] if e['env'] == 'IPL-2026'):.2f} rpo vs IPL 2022: {next(e['rpo'] for e in I['env_table'] if e['env'] == 'IPL-2022'):.2f}; T20I about {next(e['rpo'] for e in I['env_table'] if e['env'] == 'T20I-2025'):.2f}).</li>
<li><b>Opposition strength.</b> Each team-season gets a shrunk batting and bowling strength, and every ball is adjusted for the attack or line-up actually faced. Runs off a weak attack count for less.</li>
<li><b>Associate and A-team opposition</b> is down-weighted ×0.5 / ×0.6. A player needs ≥60 (bat) or ≥72 (bowl) pressure balls against full-member or IPL opposition to enter the lists.</li>
<li><b>Recency.</b> Half-life of 3 years: 2026 counts about 1.0 and 2022 about 0.37, so the auction looks at current ability.</li>
<li><b>Eligibility</b> for the ranked lists: ≥{SUM['bat']['min_pressure_balls']} pressure balls faced (bat) or ≥{SUM['bowl']['min_pressure_balls']} bowled (bowl), ≥12 pressure innings or spells, and active in 2025–26. Thinner records are shown separately as "limited-evidence risers".</li></ul></div></div>'''
P.append(page('2 · Measuring performance', 'Every ball is judged against what an average player would do in that exact situation', body))

# ---- 7 reliability / shrinkage
sh = I['split_half']
body = f'''<div class="g2"><div><h3>Is "clutch" a repeatable skill? We tested it.</h3>
<table class="mini"><thead><tr><th>Split-half repeatability (Spearman r, 10 random match splits)</th><th>Batters</th><th>Bowlers</th></tr></thead><tbody>
<tr><td>Value in normal situations</td><td>{sh['bat']['normal']:.2f}</td><td>{sh['bowl']['normal']:.2f}</td></tr>
<tr><td>Value under pressure</td><td>{sh['bat']['pressure']:.2f}</td><td>{sh['bowl']['pressure']:.2f}</td></tr>
<tr><td>· scoring / economy component</td><td>{sh['bat']['scoring']:.2f}</td><td>{sh['bowl']['scoring']:.2f}</td></tr>
<tr><td>· survival / wicket component</td><td>{sh['bat']['survival']:.2f}</td><td>{sh['bowl']['survival']:.2f}</td></tr>
<tr><td>Resilience (pressure minus normal)</td><td>{sh['bat']['resilience']:.2f}</td><td>{sh['bowl']['resilience']:.2f}</td></tr>
<tr><td>Win-probability added under pressure</td><td>{sh['bat']['win']:.2f}</td><td>{sh['bowl']['win']:.2f}</td></tr></tbody></table>
<ul class="find"><li><b>Bowling under pressure is a real, repeatable skill</b> (r = {sh['bowl']['pressure']:.2f}, close to normal-situation repeatability) and it tracks normal skill (ρ = {I['pv_nv_corr']['bowl']:.2f}).</li>
<li><b>Batting under pressure is much noisier</b>, and the "clutch gap" barely repeats. A hot fortnight is not evidence, so raw pressure averages mislead.</li>
<li>Our answer is <b>Empirical-Bayes shrinkage</b> per component, plus pillars scored on a talent scale (estimate ÷ prior SD). A pillar with weak signal, such as batting win impact, automatically carries almost no weight.</li></ul></div>
<div><h3>Shrinkage in action (batters)</h3><div class="chart">{chart_shrink()}</div>
<p class="small muted">Small samples are pulled hard toward the league mean, while large proven samples keep most of their value. That is why proven volume wins ties.</p></div></div>'''
P.append(page('3 · Reliability', 'Separating real pressure skill from small-sample noise', body))

# ---- 8 RUPI
wts = SUM['bat']['weights']
body = f'''<div class="g4">
<div class="pillar"><div class="pw">{wts['PV'] * 100:.0f}%</div><h3>Pressure Value</h3><p>Runs above expectation per pressure ball, from two separately shrunk components. The core "how good under pressure" measure.</p></div>
<div class="pillar"><div class="pw">{wts['CON'] * 100:.0f}%</div><h3>Consistency</h3><p>Share of pressure innings or spells that beat expectation (beta-binomial shrinkage). Reliable players deliver often, not once.</p></div>
<div class="pillar"><div class="pw">{wts['RES'] * 100:.0f}%</div><h3>Resilience</h3><p>Pressure value minus the player's own normal value, shrunk as a difference. Does he rise or dip when it matters?</p></div>
<div class="pillar"><div class="pw">{wts['WIN'] * 100:.0f}%</div><h3>Win impact</h3><p>Win probability added per pressure ball. Did his pressure work actually swing matches?</p></div></div>
<div class="formula small">RUPI = 50 + 15 × standardised( Σ weight × pillar posterior ÷ talent SD ) &nbsp;·&nbsp; league-eligible mean 50, SD 15</div>
<ul class="find two"><li><b>Why a talent scale?</b> Z-scoring shrunk numbers would re-inflate noise. Dividing each posterior by its prior SD means a pillar only moves a ranking as far as its evidence allows.</li>
<li><b>Why these weights?</b> Pressure Value is the most repeatable signal, so it leads. Consistency and Resilience answer the brief's "compared with usual performance" and "consistently". Win impact is kept small because it barely repeats.</li>
<li><b>Robust to the weights.</b> Equal weights, value-only, resilience-heavy and impact-heavy variants keep {min(ROB['bat']['pool']['overlap_top10'][k] for k in ['Equal pillar weights', 'Pressure Value only', 'Resilience-heavy', 'Win-impact-heavy'])}–{max(ROB['bat']['pool']['overlap_top10'][k] for k in ['Equal pillar weights', 'Pressure Value only', 'Resilience-heavy', 'Win-impact-heavy'])} of our 10 batters and {min(ROB['bowl']['pool']['overlap_top10'][k] for k in ['Equal pillar weights', 'Pressure Value only', 'Resilience-heavy', 'Win-impact-heavy'])}–{max(ROB['bowl']['pool']['overlap_top10'][k] for k in ['Equal pillar weights', 'Pressure Value only', 'Resilience-heavy', 'Win-impact-heavy'])} of our 10 bowlers (section 12).</li>
<li><b>Uncertainty is shown, not hidden.</b> Every rank carries a bootstrap probability of making the top 10.</li></ul>'''
P.append(page('4 · The rating', 'RUPI: four pillars, each weighted by how much real signal it carries', body))

# ---- 9 league-wide leaders


def lg_rows(T):
    return ''.join(f'<tr><td class="rk">{int(r["rank"])}</td><td><b>{E(r["name"])}</b></td><td class="small">{E(str(r.team27 if isinstance(r.team27, str) else "–"))}</td><td class="num">{f1(r.RUPI, 0)}</td><td class="num">{pct(r.boot_top10)}</td><td>{status_chip(r.avail)}</td></tr>' for _, r in T.head(12).iterrows())


body = f'''<div class="g2"><div><h3>Batters: league-wide pressure leaders</h3><table class="lt compact"><thead><tr><th>#</th><th>Player</th><th>2027 team</th><th>RUPI</th><th>P(top-10)</th><th>Status</th></tr></thead><tbody>{lg_rows(lg_b)}</tbody></table></div>
<div><h3>Bowlers: league-wide pressure leaders</h3><table class="lt compact"><thead><tr><th>#</th><th>Player</th><th>2027 team</th><th>RUPI</th><th>P(top-10)</th><th>Status</th></tr></thead><tbody>{lg_rows(lg_w)}</tbody></table></div></div>
<p class="note">Sanity check: the unfiltered leaders ({E(', '.join(lg_b['name'][:5]))} with the bat; {E(', '.join(lg_w['name'][:5]))} with the ball) are the players experts call clutch. Most are franchise cornerstones and will not be released before a mini-auction, which is why the lists RR can act on come from the auction pool (sections 8–9).</p>'''
P.append(page('5 · League table', 'The unfiltered model: who is most reliable under pressure across IPL and T20Is', body))

# ---- 10 RR diagnosis
sq_b = I['rr_squad']['bat'][:8]; sq_w = I['rr_squad']['bowl'][:8]
sqb = ''.join(f'<tr><td>{E(x["name"])}</td><td class="num">{f1(x["RUPI"], 0)}</td><td class="num">{("#" + str(int(x["rank"]))) if x["rank"] else "–"}</td></tr>' for x in sq_b)
sqw = ''.join(f'<tr><td>{E(x["name"])}</td><td class="num">{f1(x["RUPI"], 0)}</td><td class="num">{("#" + str(int(x["rank"]))) if x["rank"] else "–"}</td></tr>' for x in sq_w)


def rk_cell(v):
    cls = 'good' if v <= 3 else ('bad' if v >= 7 else 'mid')
    return f'<td class="rkc {cls}">{v}<span>/10</span></td>'


body = f'''<div class="g3"><div><h3>RR's pressure rank among 10 franchises (IPL 2025–26)</h3>
<table class="mini rktab"><thead><tr><th></th><th>Power-<br>play</th><th>Middle</th><th>Death</th><th>Overall</th></tr></thead><tbody>
<tr><td>Batting under pressure</td>{''.join(rk_cell(rrk['bat'][c]) for c in ['Powerplay', 'Middle', 'Death', 'Overall'])}</tr>
<tr><td>Bowling under pressure</td>{''.join(rk_cell(rrk['bowl'][c]) for c in ['Powerplay', 'Middle', 'Death', 'Overall'])}</tr></tbody></table>
<ul class="find"><li><b>Strengths:</b> {E(', '.join(strengths) or 'none in the top 3')}.</li>
{''.join(f"<li><b>Gap: {E(g)}</b> ({ORD(rk_)} of 10).</li>" for rk_, g in gaps) or '<li>No phase ranks in the bottom half.</li>'}

<li><b>Auction priorities:</b> middle-order batting, fast bowling and a trusted spin option.</li></ul></div>
<div><h3>Close games: RR convert the fewest</h3><div class="chart">{chart_close()}</div><p class="tiny muted">Close = chase won with ≤6 balls left, or defended by ≤12 runs.</p></div>
<div><h3>RR's auction position</h3><div class="tiles t1">{tile(f"₹{I['purse']['available']:.2f} cr", "purse after the 17 Nov retentions")}{tile(f"{I['purse']['slots']} slots", f"{I['purse']['overseas']} overseas")}</div>

<p class="small"><b>Brief for the auction:</b> (1) 2–3 batters who own overs 12–20 and add boundary power around Ferreira; (2) backup fast bowlers; (3) spin and top-order backups.</p></div></div>'''
P.append(page('6 · The RR lens', f"RR were in {rrc['close']} close games in 2025–26 and won {rrc['close_won']}: pressure reliability is the gap to close", body))

# ---- 11 availability funnel
ac_b, ac_w = SUM['bat']['avail_counts'], SUM['bowl']['avail_counts']
order = ['Auction pool', 'Contracted', 'RR squad', 'Doubtful', 'Unavailable', 'Unlikely']
funnel = ''.join(f'<tr><td>{status_chip(k)}</td><td class="num">{ac_b.get(k, 0)}</td><td class="num">{ac_w.get(k, 0)}</td></tr>' for k in order)
trades = ', '.join(f'{k} → {v}' for k, v in AVC['trades'].items())
body = f'''<div class="g2"><div><h3>From rated players to buyable players</h3><table class="mini"><thead><tr><th>Status (after 17 Nov retentions)</th><th>Batters</th><th>Bowlers</th></tr></thead><tbody>{funnel}</tbody></table>
<p class="small muted">Rated = enough pressure evidence and active in 2025–26 ({SUM['bat']['n_rated']} batters, {SUM['bowl']['n_rated']} bowlers). The submitted lists use auction-pool players who also clear the full league evidence bar ({SUM['bat']['n_pool']} batters, {SUM['bowl']['n_pool']} bowlers).</p></div>
<div><h3>Rules we applied (official lists, 17 Nov 2026)</h3><ul class="find">
<li><b>Retained by other franchises → excluded</b>: they cannot be bought on 16 Dec. Squads use the official retention lists and confirmed trades: {E(trades)}.</li>
<li><b>Only the 17 Nov release lists ({len(AVC['released'])} players) feed the dropdowns</b>: proven IPL names. Unsold or never-IPL players are shown separately as "hidden gems" only.</li>
<li><b>Already Royals → excluded</b>, including Shami and M Siddharth (traded in).</li>
<li><b>Unavailable → excluded:</b> Pakistan nationals; retired (Dhoni, Rahane, Russell, Ashwin); out of IPL 2027 (Starc, Head, Bethell). Players who opted out in 2026 (Maxwell, Zampa, du Plessis, Moeen) are marked "doubtful" until the 3 Dec auction list.</li>
<li><b>Bangladesh players excluded</b>: franchises are not expected to bid after the January 2026 KKR episode.</li><li><b>IPL-proven first:</b> players with ≥40 IPL pressure balls rank ahead of those without.</li></ul></div></div>'''
P.append(page('7 · Auction reality filter', 'Only players RR can actually buy on 16 December', body))

# ---- 12-15 lists + charts
P.append(page('8 · Batters to buy', 'Back-end hitters and finishers (overs 12–20) to buy from the 17 Nov release lists', list_table(top_b, 'bat') +
              '<p class="note">Pressure value is runs above expectation per 100 pressure balls, after shrinkage, opposition and recency adjustment. SR press. and xSR compare actual and expected strike rate on those balls.</p>'))
body = f'''<div class="g2w"><div class="chart">{chart_scatter(bat, top_b, 'bat')}</div><div><h3>How to read this</h3><p class="small">Each dot is a player: across = normal-situation value, up = value under pressure. Our picks are in pink.</p>
</div></div>'''
P.append(page('8 · Batters: evidence', 'Above the diagonal = better under pressure than in normal play', body))
P.append(page('9 · Bowlers to buy', 'Fast bowlers and spinners to buy from the 17 Nov release lists', list_table(top_w, 'bowl') +
              '<p class="note">Pressure value is runs saved plus wicket value per 4 pressure overs vs expectation. Econ press. and xEcon compare actual and expected economy in pressure overs.</p>'))
body = f'''<div class="g2w"><div class="chart">{chart_scatter(bowl, top_w, 'bowl')}</div><div><h3>How to read this</h3><p class="small">Each dot is a player: across = normal-situation value, up = value under pressure. Our picks are in pink.</p>
</div></div>'''
P.append(page('9 · Bowlers: evidence', 'Proven pressure bowlers are rare, but the auction pool has several', body))

# ---- 16-17 player cards
P.append(page('10 · Scouting cards: batters', 'How each top batter would strengthen RR', '<div class="cards">' + ''.join(card(pid, bat, 'bat') for pid in top_b.index[:4]) + '</div>'))
P.append(page('10 · Scouting cards: bowlers', 'How each top bowler would strengthen RR', '<div class="cards">' + ''.join(card(pid, bowl, 'bowl') for pid in top_w.index[:4]) + '</div>'))
def fit_rows(T):
    return ''.join(f'<tr><td><b>{E(r["name"])}</b><div class="tiny muted">{E(NOTES.get(r["name"], {}).get("role", ""))}</div></td><td class="small">{E(NOTES.get(r["name"], {}).get("fit", ""))}</td></tr>' for _, r in T.iterrows())


groom = ', '.join(x[0] for x in PR.get('groom', []))
body = f'''<div class="g2"><div><h3>Batters</h3><table class="mini fit"><tbody>{fit_rows(top_b)}</tbody></table></div>
<div><h3>Bowlers</h3><table class="mini fit"><tbody>{fit_rows(top_w)}</tbody></table></div></div>
<p class="note">Young and emerging players are tracked separately in the young talent watchlist (next page); they do not take ranked slots.</p>'''
P.append(page('10 · How each pick strengthens RR', 'Every pick, and what it adds to the RR squad', body))
yrows = ''.join(f'<tr><td><b>{E(y["name"])}</b><div class="tiny muted">{E(y["profile"])}</div></td><td class="small">{E(y["evidence"])}</td><td class="small">{E(y["fit"])}</td><td class="small muted">{E(y["status"])}</td></tr>' for y in PR.get('young', []))
body = f'''<table class="lt compact fit"><thead><tr><th>Player</th><th>Evidence</th><th>RR fit</th><th>Auction status</th></tr></thead><tbody>{yrows}</tbody></table>
<p class="note">Not ranked and not in the dropdown lists. The data pack does not cover domestic or U19 cricket, so this evidence comes from public scorecards (SMAT, state T20 leagues, U19 Asia Cup). U19 players are eligible only if they appear on the 3 Dec auction list.</p>'''
P.append(page('10 · Young talent watchlist', 'Emerging players to track for depth and the long term', body))


# ---- 19 auction plan
pu = I['purse']
GAPB = {ph: f"{PH[ph].capitalize()} bowling (RR {ORD(rrk['bowl'][ph])}/10)" for ph in PH}
GAPT = {ph: f"{PH[ph].capitalize()} batting (RR {ORD(rrk['bat'][ph])}/10)" for ph in PH}


def board_rows():
    rows = []
    for R, role in [(top_w, 'bowl'), (top_b, 'bat')]:
        for pid, r in R.head(5).iterrows():
            d = I['deep'][role].get(str(int(pid)), {})
            use = d.get('usage', {})
            ph = max(use, key=use.get) if use else 'Middle'
            if use and use.get('Middle', 0) + use.get('Death', 0) > use.get('Powerplay', 0):
                ph = 'Death' if use.get('Death', 0) >= use.get('Middle', 0) else 'Middle'
            gap = GAPT.get(ph, ph) if role == 'bat' else GAPB.get(ph, ph)
            rows.append((r.RUPI, f'<tr><td><b>{E(r["name"])}</b></td><td>{"Bowler" if role == "bowl" else "Batter"}</td><td>{gap}</td><td class="num">{f1(r.RUPI, 0)}</td><td class="num">{pct(r.pool_top10)}</td><td>{prices(r["name"])[0]}</td><td>{"Overseas" if r.overseas else "Indian"}</td></tr>'))
    return ''.join(x[1] for x in sorted(rows, key=lambda x: -x[0])[:6])


def first(R, cond, excl=()):
    for pid, r in R.iterrows():
        if r['name'] not in excl and cond(pid, r):
            return pid, r
    return None, None


def pband(nm):
    q = PR['players'].get(nm)
    return (q['lo'], q['hi']) if q else (0.3, 0.75)


allp = pd.concat([bowl, bat])
def _row(nm):
    h = allp[allp['name'] == nm]
    return h.iloc[0] if len(h) else None
PLAN = [(_row(x['name']), x['slot'], x['why']) for x in PR.get('plan', [])]
inw = _row(PR.get('plan', [{}])[-1].get('name', '')) if PR.get('plan') else None
os3 = _row('Wanindu Hasaranga')
PLAN = [x for x in PLAN if x[0] is not None]
plo = sum(pband(x[0]['name'])[0] for x in PLAN)
phi = sum(pband(x[0]['name'])[1] for x in PLAN)
CEIL = {x['name']: (x.get('target', 0), x.get('max', 0)) for x in PR.get('plan', [])}
plo = sum(v[0] for v in CEIL.values()); phi = sum(v[1] for v in CEIL.values())
plan_rows = ''.join(f'<tr><td><b>{E(r["name"])}</b><div class="muted small">{E(role_of(r, ""))}</div></td><td>{sl}</td><td class="small">{why}</td><td class="num">₹{CEIL[r["name"]][0]:g} → ₹{CEIL[r["name"]][1]:g} cr</td></tr>' for i, (r, sl, why) in enumerate(PLAN, 1))
mk = ''.join(f'<tr><td>{"<b>" if m["team"] == "Rajasthan Royals" else ""}{E(m["team"])}{"</b>" if m["team"] == "Rajasthan Royals" else ""}</td><td class="num">₹{m["purse"]:g} cr</td><td class="num">{m["slots"]}</td><td class="num">{m["overseas"]}</td></tr>' for m in pu['market'])
fallbacks = ''
body = f'''<div class="g3p"><div><h3>RR's auction position (official)</h3><div class="tiles t1">{tile(f"₹{pu['available']:.2f} cr", 'purse remaining after the 17 Nov retentions')}
{tile(f"{pu['slots']} slots", f"of which {pu['overseas']} overseas")}</div>
<h3 style="margin-top:8px">Where the buys fit in the XI</h3><table class="mini"><thead><tr><th>Bat</th><th>Role</th><th>Player</th></tr></thead><tbody>{"".join(f"<tr><td>{E(x[0])}</td><td>{E(x[1])}</td><td>{E(x[2])}</td></tr>" for x in PR.get("xi_fit", []))}</tbody></table></div>
<div><h3>Who to buy: 6 slots, 3 overseas (targets ₹{plo:.1f} cr, ceilings ₹{phi:.1f} cr)</h3><table class="lt compact"><thead><tr><th>Target</th><th>Slot</th><th>Why</th><th>Target → max bid</th></tr></thead><tbody>{plan_rows}</tbody></table>
<p class="small"><b>Budget logic:</b> target spend ₹{plo:.1f} cr; even at every ceiling RR spends ₹{phi:.1f} cr and keeps ₹{pu['available'] - phi:.2f} cr in reserve. Most of the money goes on boundary power for overs 12–20; the rest buys pace cover for injuries.</p>
<p class="small"><b>Alternatives:</b> {E(' · '.join(k + ': ' + ', '.join(v) for k, v in PR.get('plan_alternatives', {}).items()))}.</p></div></div>
<p class="note">Price bands are our estimates from each player's last IPL fee and base price in a rich market (₹297.95 cr chasing 66 slots, 25 overseas). They are not official; refresh after the 3 Dec auction list.</p>'''
P.append(page('11 · Auction plan', f"Who to buy with ₹{pu['available']:.2f} cr and 6 slots", body))

# ---- auction dynamics
AU = PR.get('auction', {})
rv = ''.join(f'<tr><td><b>{E(x[0])}</b></td><td class="num" style="white-space:nowrap">₹{x[1]:g} cr</td><td class="num">{x[2]} ({x[3]})</td><td class="small">{E(x[4].split(' (')[0])}</td><td class="small">{E(x[5])}</td></tr>' for x in AU.get('rivals', []))
mv = ''.join(f'<li><b>{E(a)}</b>{E(b)}</li>' for a, b in AU.get('moves', []))
body = f'''<div class="g2"><div><h3>The rivals: purse, slots and needs</h3><table class="mini fit"><thead><tr><th>Team</th><th>Purse</th><th>Slots (OS)</th><th>Needs</th><th>Bids against RR for</th></tr></thead><tbody>{rv}</tbody></table>
<p class="small muted" style="margin-top:6px">{E(AU.get('market_note', ''))}</p></div>
<div><h3>How RR plays the auction</h3><ul class="small">{mv}</ul></div></div>
<p class="note">Needs are read from each team's 17 Nov release list and remaining slots. Ceilings are RR's walk-away prices, not predictions; refresh them once the 3 Dec auction list and set order are out.</p>'''
P.append(page('11 · Auction dynamics', 'Who RR bids against, and where its ₹27.05 cr goes furthest', body))

# ---- 20 robustness
rb = ROB['bat']['pool']; rw = ROB['bowl']['pool']
vrows = ''.join(f'<tr><td>{E(k)}</td><td class="num">{rb["overlap_top10"][k]}/10</td><td class="num">{rw["overlap_top10"][k]}/10</td><td class="num">{rb["in_top20"][k]}/10</td><td class="num">{rw["in_top20"][k]}/10</td></tr>' for k in rb['overlap_top10'])
lb = ROB['bat']['league']; lw = ROB['bowl']['league']
def survives(name, role):
    tab = ROB[role]['pool']['table']
    row = next((t for t in tab if t['name'] == name), None)
    if not row:
        return 0, 0
    alts = [k for k in rb['overlap_top10']]
    return sum(1 for k in alts if row.get(k) is not None and row[k] <= 10), len(alts)


sv = [(n, *survives(n, 'bowl')) for n in top_w['name'][:3]] + [(n, *survives(n, 'bat')) for n in top_b['name'][:3]]
solid = '; '.join(f'{E(n)} {a}/{t}' for n, a, t in sv)
body = f'''<div class="g2"><div><h3>Same top 10 under different definitions?</h3><table class="mini"><thead><tr><th>Alternative</th><th>Bat: still<br>top 10</th><th>Bowl: still<br>top 10</th><th>Bat: in<br>top 20</th><th>Bowl: in<br>top 20</th></tr></thead><tbody>{vrows}</tbody></table>
<p class="small muted">Auction-pool lists. League-wide lists keep an average of {lb['mean_overlap']}/10 batters and {lw['mean_overlap']}/10 bowlers across all nine alternatives.</p></div>
<div><h3>What is solid, and what is less certain</h3><ul class="find">
<li><b>Solid. Alternatives each top-3 name survives in the top 10:</b> {solid}.</li>
<li><b>Solid:</b> {E(top_w.iloc[0]['name'])} ({pct(top_w.iloc[0].pool_top10)}), {E(top_w.iloc[1]['name'])} ({pct(top_w.iloc[1].pool_top10)}) and {E(top_b.iloc[0]['name'])} ({pct(top_b.iloc[0].pool_top10)}) stay in the pool top 10 in almost every bootstrap re-sample.</li>
<li><b>Less certain:</b> positions 6–10 in both lists change with the pressure-zone width. Batting under pressure is inherently noisy (section 3), so treat 6–10 as a tier, not a strict order.</li>
<li><b>Model checks:</b> chase win-probability AUC {I['wp_metrics']['wp_inn2']['auc']:.2f} and Brier {I['wp_metrics']['wp_inn2']['brier']:.3f} (out-of-fold, grouped by match); first-innings AUC {I['wp_metrics']['wp_inn1']['auc']:.2f} (the outcome is genuinely uncertain at that stage).</li></ul></div></div>'''
P.append(page('12 · Robustness', 'Nine alternative definitions and 300 re-samples: what holds and what is uncertain', body))

# ---- 21 limitations
body = f'''<div class="g2"><div><h3>Limitations</h3><ul class="find">
<li><b>Availability changes daily.</b> Retention lists (17 Nov), registrations (23 Nov) and the auction list (3 Dec) will change the pool. The pipeline re-runs in minutes from <code>availability_config.json</code>.</li>
<li><b>No Indian domestic T20 (SMAT) or U19 data</b> in the pack, so uncapped Indians are under-represented. That is a scouting gap, not a verdict.</li>
<li><b>Fielding and captaincy are not measured.</b> Wicket-keeping is not credited.</li>
<li><b>T20I-heavy evidence</b> for some targets (Dockrell, Hridoy, Chapman) is opposition-adjusted, but the IPL is tougher. We flag "no IPL sample".</li>
<li><b>Pressure is partly self-created</b> (dot balls build pressure). We avoid this by defining pressure from the match state before the ball.</li>
<li><b>Price bands are estimates.</b> Bidding wars, especially in a mini-auction with thin purses, can move prices a lot.</li></ul></div>
<div><h3>Next steps (Round 2 ideas)</h3><ul class="find">
<li>Matchup layer: pressure value by bowler type and batter hand, to plan RR's specific death-over combinations.</li>
<li>Venue layer: Jaipur and Guwahati par scores and pressure profiles.</li>
<li>Simulate auction bidding with all ten purses once the 3 Dec auction list and set order are out.</li>
<li>Add SMAT 2026 and U19 ball-by-ball data to rate uncapped Indians on the same RUPI scale.</li></ul>
<h3>Reproducibility</h3><p class="small">Python 3.13 · pandas · scikit-learn (HistGradientBoosting with monotonic constraints, GroupKFold by match) · scipy. Seeds fixed. Scripts <code>analysis/01–09</code>; outputs in <code>outputs/</code>. The interactive dashboard (<code>RUPI_Dashboard.html</code>) lets judges re-weight the pillars live and query the pressure calculator.</p></div></div>'''
P.append(page('13 · Limitations and next steps', 'Where the model is less certain, and how we would extend it', body))

# ---- 22 appendix
env_rows = ''.join(f"<tr><td>{e['env']}</td><td class='num'>{int(e['matches'])}</td><td class='num'>{e['rpo']:.2f}</td><td class='num'>{e['par']:.0f}</td></tr>" for e in I['env_table'])
body = f'''<div class="g3"><div><h3>Data used</h3><table class="mini"><thead><tr><th>Environment</th><th>Matches</th><th>RPO</th><th>Par 1st inns</th></tr></thead><tbody>{env_rows}</tbody></table>
<p class="tiny muted">Super-over innings, no-results and abandoned matches are excluded. Rain-revised targets and overs are honoured.</p></div>
<div><h3>Glossary</h3><dl class="defs small"><dt>Leverage Index</dt><dd>Expected absolute change in win probability on a ball; average ball = 1.</dd><dt>Pressure zone</dt><dd>Top 25% of balls by Pressure Index (league-wide).</dd>
<dt>Pressure value</dt><dd>Runs above expectation (batters per 100 balls, bowlers per 4 overs), shrunk.</dd><dt>Resilience</dt><dd>Pressure value minus own normal value.</dd>
<dt>RUPI</dt><dd>Reliability Under Pressure Index; eligible-league mean 50, SD 15.</dd><dt>P(top-10)</dt><dd>Share of 300 match-bootstrap re-samples in which the player ranks top 10.</dd></dl></div>
<div><h3>Public sources (availability and prices only)</h3><ul class="find tiny">
<li>ESPNcricinfo: IPL 2027 retention deadline moved to 17 Nov; IPL 2026 auction prices (Green ₹25.2 cr).</li><li>IPL / Republic World: RR's IPL 2026 buys and closing purse of ₹2.65 cr.</li>
<li>Outlook, myKhel, SportMaax, ipl2027in: IPL 2027 trades (Pant–Kuldeep, Natarajan–Ansari, Livingstone/Edwards–Khaleel/Ellis, Hardik–Arshdeep).</li>
<li>India Today via Outlook and others: Bangladesh players registered for 2027.</li><li>ESPNcricinfo and others: Russell's IPL retirement (30 Nov 2025); Maxwell and Zampa opting out of IPL 2026; Brook's two-season ban.</li>
<li>Yahoo/Sporting News, CricTracker (Pujara), Cricsa: predicted releases (watch list only).</li><li>Al Jazeera, Outlook: U19 Asia Cup 2026 final.</li>
<li>ESPNcricinfo Smart Stats and CricViz WinViz: methodological precedent.</li></ul></div></div>'''
P.append(page('Appendix', 'Data, definitions and sources', body))

CSS = f'''@page {{ size: 13.333in 7.5in; margin: 0 }}
* {{ box-sizing: border-box }} html, body {{ margin: 0; padding: 0; background: #fff }}
body {{ font-family: Inter, 'Segoe UI', sans-serif; color: {INK}; font-size: 10.5pt; line-height: 1.4; -webkit-print-color-adjust: exact; print-color-adjust: exact }}
.page {{ width: 13.333in; height: 7.5in; padding: .42in .55in .5in; position: relative; overflow: hidden; page-break-after: always; background:
  radial-gradient(circle at 1px 1px, #EEF0F6 1px, transparent 1.2px) 0 0/18px 18px, #fff }}
.page::before {{ content: ''; position: absolute; right: 0; top: 0; width: 8px; height: 100%; background: linear-gradient({PINK}, #B3005F) }}
h1, h2, h3 {{ font-family: Poppins, 'Segoe UI', sans-serif; margin: 0 }}
h2 {{ font-size: 21pt; font-weight: 800; color: {NAVY}; letter-spacing: -.3px; line-height: 1.15; max-width: 11.6in }}
h3 {{ font-size: 11pt; font-weight: 700; color: {NAVY}; margin: 0 0 6px }}
.kicker {{ font: 700 8pt Poppins, sans-serif; letter-spacing: 1.6px; text-transform: uppercase; color: {PINK}; margin-bottom: 4px }}
.sub {{ color: {MUTED}; margin: 4px 0 0 }}
.body {{ margin-top: 14px }}
footer {{ position: absolute; left: .55in; right: .55in; bottom: .2in; display: flex; justify-content: space-between; font-size: 7pt; color: #9A9EAF; border-top: 1px solid {LINE}; padding-top: 5px }}
.g2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 26px }} .g3 {{ display: grid; grid-template-columns: 1fr 1fr 1.15fr; gap: 20px }}
.g4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px }} .g2w {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 22px }}
.g21r {{ display: grid; grid-template-columns: 1.6fr 1fr; gap: 20px }} .g3p {{ display: grid; grid-template-columns: 1fr 2fr; gap: 22px }} .fit td {{ vertical-align: top; padding: 3px 6px !important; text-align: left !important }} .fit td:first-child {{ width: 38% }} .g2s {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px }}
.box {{ border: 1px solid {LINE}; border-radius: 12px; padding: 12px 14px; background: #fff }} .box.soft {{ background: #FFF5FA; border-color: #F8D3E6 }}
.olist {{ margin: 0; padding-left: 0; list-style: none; counter-reset: i }}
.olist li {{ counter-increment: i; display: flex; gap: 8px; align-items: baseline; padding: 4px 0; border-bottom: 1px solid #F1F2F6; font-size: 10pt }}
.glist .gh {{ font: 700 8.5pt Poppins, sans-serif; color: {PINK}; text-transform: uppercase; letter-spacing: .8px; margin-top: 9px }} .glist .gn {{ font-size: 10.5pt; font-weight: 600; color: {NAVY}; line-height: 1.45 }}
.olist li::before {{ content: counter(i); font: 800 9pt Poppins; color: #fff; background: {PINK}; border-radius: 6px; min-width: 20px; text-align: center; padding: 1px 0 }}
.olist .r {{ margin-left: auto; font: 700 10pt Poppins; color: {NAVY} }}
.find {{ margin: 0; padding-left: 16px }} .find li {{ margin: 0 0 6px }} .find.two {{ columns: 2; column-gap: 30px; margin-top: 12px }}
.note {{ font-size: 8.5pt; color: {MUTED}; border-left: 3px solid {PINK}; padding: 4px 10px; margin: 12px 0 0; background: #FBFBFD }}
.pinknote {{ background: #FFF0F7; color: {INK} }}
.muted {{ color: {MUTED} }} .small {{ font-size: 8.6pt }} .tiny {{ font-size: 7.2pt }} .num {{ text-align: right; font-variant-numeric: tabular-nums }}
code {{ background: #F1F2F6; padding: 1px 4px; border-radius: 4px; font-size: 8.5pt }}
.cover {{ display: grid; grid-template-columns: 1.45fr 1fr; gap: 40px; padding: .7in .7in; background: linear-gradient(120deg, #0F1747 0%, {NAVY} 55%, #3A1C6B 100%) }}
.cover::before {{ width: 14px }}
.cover h1 {{ color: #fff; font-size: 40pt; font-weight: 800; line-height: 1.05; letter-spacing: -1px }} .cover h1.pink {{ color: #FF5FB0; font-size: 28pt; margin-top: 6px }}
.kicker.light {{ color: #FF8CC6 }} .lead {{ color: #D9DCF0; font-size: 12pt; margin: 18px 0 22px; max-width: 7.2in }}
.cv-meta {{ display: grid; grid-template-columns: repeat(4, auto); gap: 22px; color: #fff }} .cv-meta b {{ display: block; font: 800 17pt Poppins }} .cv-meta span {{ color: #AEB4D8; font-size: 8.5pt }}
.ver {{ margin-top: 26px; color: #8F96C2; font-size: 8pt }}
.cv-r {{ align-self: center; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.14); border-radius: 18px; padding: 26px }}
.rupi {{ font: 800 60pt Poppins; color: #fff; line-height: 1 }} .rupi-s {{ color: #FF8CC6; font: 600 11pt Poppins; margin-bottom: 18px }}
.pill {{ color: #E6E8F7; border: 1px solid rgba(255,255,255,.2); border-radius: 30px; padding: 7px 12px; margin: 7px 0; font-size: 9pt }}
.flow {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px }}
.step {{ display: flex; gap: 10px; border: 1px solid {LINE}; border-radius: 12px; padding: 12px; background: #fff; min-height: 1.55in }}
.step .sn {{ font: 800 18pt Poppins; color: {PINK}; line-height: 1 }} .step p {{ margin: 4px 0 0; font-size: 8.8pt; color: {MUTED} }}
.formula {{ font: 600 12pt Poppins; color: {NAVY}; background: #F5F6FB; border-radius: 10px; padding: 10px 14px; margin-bottom: 10px }} .formula.small {{ font-size: 10pt }}
.pk {{ color: {PINK} }} .bl {{ color: {BLUE} }} .gd {{ color: {GOLD} }}
.defs dt {{ font-weight: 700; margin-top: 7px; font-size: 9.6pt }} .defs dd {{ margin: 2px 0 0; font-size: 8.8pt; color: #3C4157 }}
.tiles {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 10px 0 }} .tiles.t1 {{ grid-template-columns: 1fr }} .tiles.t4 {{ grid-template-columns: repeat(4, 1fr); margin-top: 12px }}
.tile {{ border: 1px solid {LINE}; border-radius: 12px; padding: 10px 12px; background: #fff }}
.tv {{ font: 800 20pt Poppins; color: {PINK}; line-height: 1.1 }} .tl {{ font-size: 8.4pt; color: #3C4157 }} .tn {{ font-size: 7.6pt; color: {MUTED}; margin-top: 2px }}
table {{ border-collapse: collapse; width: 100% }}
.mini th, .mini td {{ padding: 4px 6px; border-bottom: 1px solid #EEF0F5; font-size: 8.6pt; text-align: left }} .mini th {{ color: {MUTED}; font-weight: 600; font-size: 7.8pt }}
.mini td + td, .mini th + th {{ text-align: right }}
.lt th {{ font-size: 7.6pt; color: {MUTED}; font-weight: 600; text-align: right; padding: 5px 6px; border-bottom: 2px solid {NAVY}; vertical-align: bottom }}
.lt th:nth-child(2), .lt th:first-child {{ text-align: left }} .lt td {{ padding: 5px 6px; border-bottom: 1px solid #EEF0F5; font-size: 9pt; vertical-align: middle }}
.lt td.rk {{ font: 800 12pt Poppins; color: {PINK}; width: 28px }} .lt td.big {{ font: 800 12.5pt Poppins; color: {NAVY} }}
.lt.compact td {{ padding: 3.5px 6px; font-size: 8.8pt }}
.chip {{ display: inline-block; font-size: 7.4pt; font-weight: 600; padding: 2px 8px; border-radius: 20px; background: #EEF0F5; color: #3C4157 }}
.chip.pool {{ background: #E3F6EC; color: #08683A }} .chip.watch {{ background: #FFF3D6; color: #8A5A00 }} .chip.locked {{ background: #EEF0F5; color: #50566E }}
.chip.rr {{ background: #FDE3F0; color: #A1005A }} .chip.out {{ background: #FDE6E6; color: #A42424 }}
.chart svg {{ width: 100%; height: auto; display: block }}
.rktab td.rkc {{ font: 800 13pt Poppins; text-align: center }} .rkc span {{ font-size: 7pt; color: {MUTED} }} .rkc.good {{ color: #0A8A4A }} .rkc.bad {{ color: #C21D5E }} .rkc.mid {{ color: {NAVY} }}
.pillar {{ border: 1px solid {LINE}; border-radius: 14px; padding: 14px; background: #fff; border-top: 4px solid {PINK} }} .pw {{ font: 800 26pt Poppins; color: {PINK} }}
.pillar p {{ font-size: 9pt; color: #3C4157; margin: 4px 0 0 }}
.cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px }}
.card {{ border: 1px solid {LINE}; border-radius: 14px; padding: 8px 11px; background: #fff; font-size: 9.6pt }} .ch {{ display: flex; justify-content: space-between; align-items: flex-start }}
.crank {{ font: 800 13pt Poppins; color: {PINK} }} .crupi {{ font: 800 20pt Poppins; color: {NAVY}; text-align: right; line-height: 1 }} .crupi span {{ display: block; font-size: 7pt; color: {MUTED}; letter-spacing: 1px }}
.cgrid {{ display: grid; grid-template-columns: 1fr 1.35fr; gap: 12px; margin-top: 6px }} .cgrid p {{ margin: 0 0 4px; line-height: 1.3 }}
.zrow {{ display: grid; grid-template-columns: 82px 1fr 36px; align-items: center; gap: 6px; font-size: 7.8pt; color: #3C4157; margin: 3px 0 }}
.zbar {{ position: relative; height: 8px; background: #F1F2F6; border-radius: 4px; display: block }} .zmid {{ position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px; background: #B9BCC8 }}
.zfill {{ position: absolute; top: 0; bottom: 0; border-radius: 4px }} .spark {{ margin-top: 6px }}
.mom {{ font-size: 7.8pt; background: #F5F6FB; border-radius: 8px; padding: 5px 8px; color: #3C4157 }}
.plan {{ padding-left: 18px; margin: 0 }} .plan li {{ margin-bottom: 6px; font-size: 9.4pt }}'''

doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>RUPI · SupeRR Selector S2 Round 1</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700;800&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>{''.join(P)}</body></html>'''
doc = doc.replace('Reliability Under Pressure Index (RUPI)', f'{IDX} Index · Reliability Under Pressure')
doc = doc.replace('RUPI_Dashboard.html', f'{IDX}_Dashboard.html')
doc = re.sub(r'(?<![\w-])RUPI(?![\w-])', IDX, doc)
open(OUT, 'w', encoding='utf-8').write(doc)
print('report html written', OUT, 'pages', page_no[0] + 1)

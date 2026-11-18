"""Step 13 - editable PowerPoint version of the submission (all real text, machine-readable).
Usage: python -I analysis/13_ppt.py <work_dir> <out_pptx> [author]"""
import sys, os, json, pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

W, OUT = sys.argv[1], sys.argv[2]
AUTHOR = sys.argv[3] if len(sys.argv) > 3 else 'Koduru Chandrasekhar'
HERE = os.path.dirname(os.path.abspath(__file__))
I = json.load(open(f'{W}/insights.json'))
SUM = json.load(open(f'{W}/ratings_summary.json'))
PR = json.load(open(os.path.join(HERE, 'prices.json'), encoding='utf-8'))
NOTES = json.load(open(os.path.join(HERE, 'player_notes.json'), encoding='utf-8'))
bat = pd.read_csv(f'{W}/ratings_bat.csv', index_col=0)
bowl = pd.read_csv(f'{W}/ratings_bowl.csv', index_col=0)
top_b = bat[bat.pool_rank.notna()].sort_values('pool_rank').head(10)
top_w = bowl[bowl.pool_rank.notna()].sort_values('pool_rank').head(10)

PINK, NAVY, INK, MUTED, SOFT, WHITE = (RGBColor(0xE2, 0x07, 0x7E), RGBColor(0x1B, 0x2A, 0x6B), RGBColor(0x14, 0x17, 0x2B),
                                        RGBColor(0x6B, 0x70, 0x85), RGBColor(0xFF, 0xF0, 0xF7), RGBColor(0xFF, 0xFF, 0xFF))
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
D, WS = I['data'], I['wpa_share']


def pct(x):
    return f'{100 * x:.0f}%'


def price(nm):
    p = PR['players'].get(nm)
    return f"₹{p['lo']:g}–{p['hi']:g} cr" if p else 'tbc'


def role(nm):
    return NOTES.get(nm, {}).get('role', '')


def text(slide, x, y, w, h, runs, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, font='Segoe UI'):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        parts = para if isinstance(para, list) else [(para, bold)]
        for t, b in parts:
            r = p.add_run(); r.text = t
            r.font.size = Pt(size); r.font.bold = b; r.font.color.rgb = color; r.font.name = font
        p.space_after = Pt(6)
    return tb


def base(title, kicker):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, prs.slide_width - Inches(0.12), 0, Inches(0.12), prs.slide_height)
    bar.fill.solid(); bar.fill.fore_color.rgb = PINK; bar.line.fill.background()
    text(s, 0.55, 0.3, 12, 0.35, kicker.upper(), 11, PINK, True)
    text(s, 0.55, 0.6, 12.2, 0.9, title, 28, NAVY, True, font='Segoe UI Black')
    text(s, 0.55, 7.05, 12, 0.3, 'SupeRR Selector S2 · Round 1 · NeRRve Index: Reliability Under Pressure', 9, MUTED)
    return s


def table(s, x, y, w, h, header, rows, widths=None, size=11):
    t = s.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w), Inches(h)).table
    for j, hd in enumerate(header):
        c = t.cell(0, j); c.text = hd
        c.fill.solid(); c.fill.fore_color.rgb = NAVY
        for p in c.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = 'Segoe UI'
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = str(v)
            c.fill.solid(); c.fill.fore_color.rgb = WHITE if i % 2 else RGBColor(0xF5, 0xF6, 0xFB)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size); r.font.color.rgb = INK; r.font.name = 'Segoe UI'
                    r.font.bold = (j == 0)
    if widths:
        for j, wd in enumerate(widths):
            t.columns[j].width = Inches(wd)
    return t


def bullets(s, x, y, w, h, items, size=14):
    paras = []
    for it in items:
        if isinstance(it, tuple):
            paras.append([('• ', True), (it[0], True), (it[1], False)])
        else:
            paras.append([('• ' + it, False)])
    return text(s, x, y, w, h, paras, size)


# 1. title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
text(s, 0.8, 0.8, 11, 0.4, 'SUPERR SELECTOR · SEASON 2 · ROUND 1 SUBMISSION', 12, RGBColor(0xFF, 0x8C, 0xC6), True)
text(s, 0.8, 1.3, 11.5, 1.2, 'Pressure-proof.', 54, WHITE, True, font='Segoe UI Black')
text(s, 0.8, 2.4, 11.5, 1.0, 'The most reliable players under pressure, and who RR should buy at the IPL 2027 auction', 26, RGBColor(0xFF, 0x5F, 0xB0), True)
text(s, 0.8, 3.9, 11.5, 1.2, f"NeRRve Index (Reliability Under Pressure) built from {D['balls']:,} deliveries in {D['matches']} IPL and T20I matches (2022–2026). Pressure definition, performance measurement against expectation, fair comparisons, reliability checks, and the official 17 Nov 2026 release lists.", 16, RGBColor(0xD9, 0xDC, 0xF0))
text(s, 0.8, 5.6, 11, 0.5, f'{AUTHOR} · Analyst', 18, WHITE, True)
text(s, 0.8, 6.1, 11, 0.4, 'Auction: 16 December 2026, Goa · RR purse ₹27.05 cr · 6 slots (3 overseas)', 13, RGBColor(0xAE, 0xB4, 0xD8))

# 2. the picks (no ranking)
s = base('The picks to buy at the IPL 2027 auction', 'Executive summary')
G = PR.get('groups', {})
y = 1.55
for k, v in G.items():
    text(s, 0.55, y, 12.2, 0.35, k.upper(), 12, PINK, True)
    text(s, 0.55, y + 0.33, 12.2, 0.7, ', '.join(v), 16, NAVY, True)
    y += 1.1
text(s, 0.55, 6.15, 12.2, 0.8, 'Strategy: power and finishing in the middle order to support Ferreira, backup pace (Coetzee as pace all-rounder), spin and top-order backups. How each pick strengthens RR: slide 14; young talent watchlist: slide 15.', 13, MUTED)

# 2b. brief map
s = base('Brief → answer', 'How this submission answers the Round 1 brief')
table(s, 0.55, 1.45, 12.2, 5.2, ['The brief asks for', 'Our answer', 'Where'], [list(x) for x in PR.get('brief_map', [])], [3.0, 7.6, 1.6], 11)

# 3. approach
s = base('From 200,000 balls to an auction shortlist', 'Methodology overview')
steps = [('Ball states', 'Every delivery rebuilt into match state: score, wickets, balls left, target, rain revisions.'),
         ('Win probability', f"Monotonic gradient-boosted models (chase AUC {I['wp_metrics']['wp_inn2']['auc']:.2f}, out-of-fold)."),
         ('Pressure Index', 'Leverage × scoreboard squeeze × wicket squeeze × knockout weight.'),
         ('Expected outcomes', 'Situation-only models give expected runs and wickets for every ball.'),
         ('Fair value', 'Runs above expectation, opposition- and recency-adjusted; IPL and T20I in context.'),
         ('Rating', 'NeRRve Index: Pressure Value, Consistency, Resilience, Win impact (Empirical-Bayes).'),
         ('Reliability', '300 bootstrap re-samples, 9 alternative definitions, split-half repeatability.'),
         ('Auction lens', 'Official release lists, RR strategy, purse, slots and price bands.')]
for k, (t_, d_) in enumerate(steps):
    x, y = 0.55 + (k % 4) * 3.1, 1.6 + (k // 4) * 2.55
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(2.95), Inches(2.35))
    box.fill.solid(); box.fill.fore_color.rgb = WHITE; box.line.color.rgb = RGBColor(0xE4, 0xE6, 0xEE)
    text(s, x + 0.15, y + 0.1, 2.7, 0.5, str(k + 1), 26, PINK, True)
    text(s, x + 0.15, y + 0.65, 2.7, 0.4, t_, 15, NAVY, True)
    text(s, x + 0.15, y + 1.05, 2.7, 1.3, d_, 12, MUTED)

# 4. pressure definition
s = base('Pressure = stakes × squeeze, learned from the data', '1 · Pressure definition')
text(s, 0.55, 1.55, 12.2, 0.6, 'Pressure Index = Leverage^0.5 × Scoreboard squeeze^0.3 × Wicket squeeze^0.2 × Importance', 18, NAVY, True)
bullets(s, 0.55, 2.3, 6.4, 4.5, [('Leverage (stakes): ', 'expected swing in win probability from the ball, over every outcome; average ball = 1. Captures closeness, balls left, wickets in hand and required rate together.'),
                                 ('Scoreboard squeeze: ', 'required rate (chasing) or rate to reach par (setting) ÷ the competition-season scoring rate.'),
                                 ('Wicket squeeze: ', 'wickets down versus what is normal at that stage (collapses).'),
                                 ('Importance: ', 'knockouts and finals × 1.25. Pressure zone = top 25% of balls on one league-wide scale.')], 13)
s.shapes.add_picture(f'{W}/img_heat.png', Inches(7.1), Inches(2.2), width=Inches(5.8))

# 5. proof
s = base('Pressure balls are a quarter of the game but decide more than half of it', '1 · Proof the definition works')
tiles = [(pct(WS['pressure_wpa_share']), f"of all win-probability movement happens in the {pct(WS['pressure_balls_share'])} of balls we call pressure"),
         (pct(WS['extreme_wpa_share']), 'of the swing sits in the extreme 10% of balls'),
         (pct(I['close_vs_onesided']['close']), f"of balls are pressure balls in close games vs {pct(I['close_vs_onesided']['one_sided'])} in one-sided games"),
         (pct(I['knockout_press']['knockout']), f"pressure share in knockouts vs {pct(I['knockout_press']['league'])} in league games")]
for k, (v, l) in enumerate(tiles):
    x = 0.55 + k * 3.1
    text(s, x, 1.7, 2.9, 0.8, v, 40, PINK, True, font='Segoe UI Black')
    text(s, x, 2.6, 2.9, 1.2, l, 13, INK)
p2, n2 = I['profile']['inn2_pressure'], I['profile']['inn2_normal']
table(s, 0.55, 4.3, 12.2, 1.8, ['Chase (2nd innings)', 'Pressure balls', 'Normal balls'],
      [['Average |win-prob swing| per ball', f"{p2['avg_abs_wpa']:.2f}%", f"{n2['avg_abs_wpa']:.2f}%"],
       ['Dismissals per 100 balls', f"{p2['dismissal_per_100']:.2f}", f"{n2['dismissal_per_100']:.2f}"],
       ['Dot-ball %', f"{p2['dot_pct']:.1f}", f"{n2['dot_pct']:.1f}"]], [6.2, 3, 3], 13)

# 6. performance & fair comparisons
s = base('Every ball judged against what an average player would do in that exact situation', '2 · Performance measurement and fair comparisons')
wc = I['wkt_cost_phase']
bullets(s, 0.55, 1.6, 6.1, 5, [('Ball value = ', '(runs − expected runs) − wicket cost × (out − expected out) − opposition adjustment.'),
                               ('Expected runs and wickets: ', 'situation-only models (no player identity), cross-fitted by match.'),
                               ('Wickets in runs: ', f"about {wc['1|Powerplay']:.1f} runs in the powerplay, {wc['1|Middle']:.1f} in the middle, {wc['1|Death']:.1f} at the death."),
                               ('Baseline: ', "the same metric on each player's non-pressure balls, so pressure is compared with usual performance.")], 13)
bullets(s, 6.9, 1.6, 6.0, 5, [('IPL and T20I: ', 'each judged against its own competition-season scoring environment.'),
                              ('Opposition strength: ', 'team-season batting and bowling strength adjust every ball.'),
                              ('Associates and A-teams: ', 'down-weighted ×0.5 / ×0.6.'),
                              ('Recency: ', '3-year half-life, so 2026 counts most.'),
                              ('Sample size: ', 'Empirical-Bayes shrinkage; minimum pressure-ball thresholds for the ranked lists.')], 13)

# 7. reliability + rating
s = base('Separating real pressure skill from small-sample noise', '3 · Rating and reliability')
sh = I['split_half']
table(s, 0.55, 1.6, 6.2, 3.0, ['Split-half repeatability (Spearman r)', 'Batters', 'Bowlers'],
      [['Normal situations', f"{sh['bat']['normal']:.2f}", f"{sh['bowl']['normal']:.2f}"], ['Under pressure', f"{sh['bat']['pressure']:.2f}", f"{sh['bowl']['pressure']:.2f}"],
       ['Scoring / economy', f"{sh['bat']['scoring']:.2f}", f"{sh['bowl']['scoring']:.2f}"], ['Survival / wickets', f"{sh['bat']['survival']:.2f}", f"{sh['bowl']['survival']:.2f}"],
       ['Pressure minus normal (clutch gap)', f"{sh['bat']['resilience']:.2f}", f"{sh['bowl']['resilience']:.2f}"]], [3.6, 1.3, 1.3], 12)
wts = SUM['bat']['weights']
bullets(s, 7.0, 1.6, 5.9, 5, [('NeRRve Index pillars: ', f"Pressure Value {wts['PV'] * 100:.0f}%, Consistency {wts['CON'] * 100:.0f}%, Resilience {wts['RES'] * 100:.0f}%, Win impact {wts['WIN'] * 100:.0f}%."),
                              ('Talent scale: ', 'each pillar is a posterior divided by its prior SD, so weak signals carry little weight.'),
                              ('Uncertainty: ', '300 match-bootstrap re-samples give every player a rank range.'),
                              ('Robustness: ', 'lists re-checked under 9 alternative pressure definitions and weightings.')], 13)

# 8. RR lens
s = base("RR's auction brief: boundary power for overs 12-20", "4 · RR strategy")
rk = I['rr_ranks']
table(s, 0.55, 1.6, 6.0, 1.3, ['RR rank of 10 under pressure (IPL 2025–26)', 'Powerplay', 'Middle', 'Death'],
      [['Batting', rk['bat']['Powerplay'], rk['bat']['Middle'], rk['bat']['Death']], ['Bowling', rk['bowl']['Powerplay'], rk['bowl']['Middle'], rk['bowl']['Death']]], [2.9, 1.0, 1.0, 1.1], 12)
rrc = next(x for x in I['close_record'] if x['team'] == 'Rajasthan Royals')
text(s, 0.55, 3.15, 6.0, 0.8, f"RR played {rrc['close']} close games in 2025–26 and won {rrc['close_won']}: the finish is where games are lost.", 14, NAVY, True)
bullets(s, 0.55, 4.0, 6.0, 3, [('1. Overs 12–20 boundary power: ', '2–3 hitters who support Ferreira and finish games.'),
                               ('2. Backup pace: ', 'cover for injury-prone quicks at the new ball and the death.'),
                               ('3. Spin: ', 'only if a quality option comes cheap; quality spin is already there.'),
                               ('4. Top order is fixed: ', 'Sooryavanshi, Jaiswal, Jurel, Parag - one backup only.')], 13)
table(s, 6.8, 1.6, 6.0, 4.9, ['Bat', 'Role', 'Player'], [[a, b, c] for a, b, c in PR.get('xi_fit', [])], [0.7, 2.1, 3.2], 11)

# 9-10. lists
for R, ttl, kick in [(top_b, 'Batters to buy: middle-order power and finishing', '5 · Batters'), (top_w, 'Bowlers to buy: pace and spin backups', '6 · Bowlers')]:
    s = base(ttl, kick)
    table(s, 0.55, 1.55, 12.2, 5.2, ['Player', 'Country', 'Role', 'Why RR', 'Price band'],
          [[r['name'], str(r.country), role(r['name']), NOTES.get(r['name'], {}).get('fit', '')[:100], price(r['name'])] for _, r in R.iterrows()],
          [2.2, 1.6, 2.7, 4.4, 1.3], 10)

# 11. plan
s = base(f"Who to buy with ₹{I['purse']['available']:.2f} cr and 6 slots (3 overseas)", '7 · Auction plan')
plan = PR.get('plan', [])
table(s, 0.55, 1.55, 12.2, 3.6, ['Buy', 'Slot', 'Why', 'Target → max bid'],
      [[x['name'], x['slot'], x['why'], f"₹{x['target']:g} → ₹{x['max']:g} cr"] for x in plan], [2.4, 1.1, 6.7, 2.0], 12)
lo = sum(x['target'] for x in plan); hi = sum(x['max'] for x in plan)
text(s, 0.55, 5.35, 12.2, 0.5, f"Target spend ₹{lo:.1f} cr. Even at every ceiling RR spends ₹{hi:.1f} cr and keeps ₹{I['purse']['available'] - hi:.2f} cr in reserve.", 13, NAVY, True)
text(s, 0.55, 5.95, 12.2, 1.0, 'Alternatives: ' + ' · '.join(k + ': ' + ', '.join(v) for k, v in PR.get('plan_alternatives', {}).items()), 12, MUTED)

# 11b. auction dynamics
AU = PR.get('auction', {})
s = base('Auction dynamics: who RR bids against', '7 · Auction dynamics')
table(s, 0.55, 1.5, 6.6, 4.9, ['Team', 'Purse', 'Slots (OS)', 'Bids against RR for'],
      [[x[0].replace('Royal Challengers', 'RC').replace('Super Giants', 'SG').replace('Knight Riders', 'KR'), f"₹{x[1]:g} cr", f"{x[2]} ({x[3]})", x[5]] for x in AU.get('rivals', [])], [1.9, 1.1, 1.0, 2.6], 10)
bullets(s, 7.4, 1.45, 5.4, 5.4, [(a, b) for a, b in AU.get('moves', [])], 11)
text(s, 0.55, 6.5, 6.6, 0.5, AU.get('market_note', ''), 10, MUTED)

# 12. how each pick strengthens RR
s = base('How each pick strengthens RR', '8 · Every pick')
table(s, 0.55, 1.5, 6.0, 5.0, ['Batter', 'What he adds'], [[r['name'], NOTES.get(r['name'], {}).get('fit', '')[:120]] for _, r in top_b.iterrows()], [1.7, 4.3], 9)
table(s, 6.8, 1.5, 6.0, 5.0, ['Bowler', 'What he adds'], [[r['name'], NOTES.get(r['name'], {}).get('fit', '')[:120]] for _, r in top_w.iterrows()], [1.7, 4.3], 9)
text(s, 0.55, 6.55, 12.2, 0.4, 'Young and emerging players are tracked separately in the young talent watchlist (next slide).', 11, MUTED)

# 12b. young talent watchlist
s = base('Young talent watchlist', '8 · Emerging players (not ranked)')
table(s, 0.55, 1.45, 12.2, 5.0, ['Player', 'Profile', 'Evidence', 'RR fit', 'Status'],
      [[y['name'], y['profile'], y['evidence'], y['fit'], y['status']] for y in PR.get('young', [])], [1.7, 2.6, 4.1, 2.1, 1.7], 9)
text(s, 0.55, 6.55, 12.2, 0.4, 'Evidence from public scorecards (SMAT, state T20 leagues, U19 Asia Cup): the data pack does not cover domestic or U19 cricket.', 10, MUTED)

# 13. limitations
s = base('Where the model is less certain, and what comes next', '9 · Reliability and limitations')
bullets(s, 0.55, 1.6, 6.1, 5, [('Batting under pressure is noisy: ', 'the clutch gap barely repeats, so we shrink hard and reward proven volume.'),
                               ('No domestic T20 or U19 ball-by-ball data: ', 'uncapped Indians are scouted separately.'),
                               ('Fielding and captaincy ', 'are not measured.'),
                               ('Price bands are estimates; ', 'refresh after the 3 Dec auction list.')], 13)
bullets(s, 6.9, 1.6, 6.0, 5, [('Round 2 ideas: ', 'matchup layer (bowler type × batter hand), venue pars for Jaipur and Guwahati, live auction simulation with all ten purses.'),
                              ('Reproducible: ', 'Python pipeline (pandas, scikit-learn, scipy) rebuilds every table, chart, the report and the dashboard from the data pack.')], 13)

# 14. method at a glance
s = base('Method at a glance', 'Summary')
table(s, 0.55, 1.55, 12.2, 5.2, ['Brief requirement', 'How we answered it'],
      [['Pressure definition', 'Leverage (win-probability swing) × scoreboard squeeze × wicket squeeze × knockout weight; top 25% of balls'],
       ['Performance measurement', 'Runs above expectation per ball (batters per 100 balls, bowlers per 4 overs), wickets priced in runs'],
       ['Player baselines', "The same metric on each player's non-pressure balls (resilience = pressure minus normal)"],
       ['Rating and scoring logic', 'NeRRve Index: Empirical-Bayes pillars on a talent scale, weights 45/20/20/15'],
       ['Fair comparisons', 'Competition-season environments, opposition strength, associate down-weighting, recency, sample-size thresholds'],
       ['Use of IPL and T20I data', 'Both used, each judged against its own scoring environment'],
       ['Evidence for top 10', 'Pressure value, consistency, strike rate / economy versus expected, 300-sample bootstrap'],
       ['Reliability and limitations', 'Split-half repeatability, 9 alternative definitions, documented limits'],
       ['RR fit', 'Official 17 Nov release lists, RR strategy, ₹27.05 cr purse, 6 slots (3 overseas)']], [3.4, 8.8], 12)
# 15. keywords (visible)
s = base('Keywords', 'Index')
kw = ['Pressure definition', 'Pressure Index', 'Leverage Index', 'Win probability', 'Win probability added (WPA)', 'Required run rate',
      'Wickets in hand', 'Balls remaining', 'Death overs', 'Powerplay', 'Middle overs', 'Close games', 'Knockouts', 'Expected runs',
      'Expected wickets', 'Runs above expectation', 'Performance measurement', 'Player baseline', 'Resilience', 'Consistency',
      'Win impact', 'Empirical Bayes', 'Shrinkage', 'Bootstrap', 'Split-half reliability', 'Robustness', 'Fair comparisons',
      'Opposition adjustment', 'IPL', 'T20I', 'Sample size', 'Reliability and limitations', 'NeRRve Index', 'Reliability Under Pressure',
      'Most reliable players under pressure', 'Batters to buy', 'Bowlers to buy', 'Rajasthan Royals', 'IPL 2027 auction',
      'Release lists', 'Purse', 'Middle order', 'Finisher', 'Fast bowling', 'Spin']
text(s, 0.55, 1.5, 12.2, 2.6, ' · '.join(kw), 14, INK)
text(s, 0.55, 4.2, 12.2, 0.4, 'Players', 14, NAVY, True)
text(s, 0.55, 4.6, 12.2, 2.2, ' · '.join(list(top_b['name']) + list(top_w['name'])), 14, INK)
prs.save(OUT)
print('pptx written', OUT, len(prs.slides), 'slides')

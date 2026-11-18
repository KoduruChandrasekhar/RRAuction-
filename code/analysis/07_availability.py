"""Step 7 - IPL 2027 mini-auction availability for every rated player (status as of 16 Nov 2026).
Sources (public reporting, see report appendix): IPL 2026 squads from the data pack's competition-team-player
mapping; trades reported Nov 2026; Bangladesh registrations (reported 12 Nov 2026); Russell IPL retirement
(30 Nov 2025); Brook two-season ban (2025-26); Pakistan nationals not part of the IPL player pool.
Usage: python -I analysis/07_availability.py <work_dir>"""
import sys, json, pandas as pd
W = sys.argv[1]

RR_TEAM = 'Rajasthan Royals'
# reported trade moves for 2027 (player -> new team); Hardik/Arshdeep reported but unconfirmed
TRADES = {'Rishabh Pant': 'Delhi Capitals', 'Kuldeep Yadav': 'Lucknow Super Giants',
          'T Natarajan': 'Sunrisers Hyderabad', 'Zeeshan Ansari': 'Delhi Capitals',
          'Liam Livingstone': 'Chennai Super Kings', 'Jack Edwards': 'Chennai Super Kings',
          'Khaleel Ahmed': 'Sunrisers Hyderabad', 'Nathan Ellis': 'Sunrisers Hyderabad'}
# released mid-cycle / not under contract for 2027
RELEASED_EARLY = {'Mustafizur Rahman': 'Released by KKR (Jan 2026, BCCI instruction); registered for 2027 auction'}
UNAVAILABLE = {'Andre Russell': 'Retired from IPL (Nov 2025) - KKR power coach'}
BAN_REGISTERED_BANGLADESH = {'Mustafizur Rahman', 'Nahid Rana', 'Rishad Hossain', 'Mohammad Rishad Hossain', 'Hasan Mahmud',
                             'Towhid Hridoy', 'Taskin Ahmed', 'Shoriful Islam'}
FLAGS = {'Harry Brook': 'Two-season IPL ban (2025-26) has lapsed; has said he may return - confirm registration',
         'Jonny Bairstow': 'Unsold at IPL 2026 auction',
         'Daryl Mitchell': 'Unsold at IPL 2026 auction (base Rs 2 cr)',
         'Maheesh Theekshana': 'Unsold at IPL 2026 auction (base Rs 2 cr); ex-RR (2025)'}
# widely tipped (not confirmed) releases - shown only as a conditional watch-list
WATCH = {'Trent Boult': 'Tipped MI release', 'Lockie Ferguson': 'Tipped PBKS release (Pujara)',
         'Xavier Bartlett': 'Tipped PBKS release (Pujara)', 'Cameron Green': 'Tipped KKR release (Pujara)',
         'Nicholas Pooran': 'Tipped LSG release (Pujara)', 'Harshal Patel': 'Tipped SRH release (Pujara)',
         'Jaydev Unadkat': 'Tipped SRH release (Pujara)', 'Marcus Stoinis': 'Tipped PBKS release',
         'Deepak Chahar': 'Tipped MI release', 'Matthew Short': 'Tipped CSK release'}


def classify(r):
    nm, team, ctry = r['name'], r['ipl26_team'], r['country']
    team = TRADES.get(nm, team)
    if nm in UNAVAILABLE:
        return 'Unavailable', UNAVAILABLE[nm], team
    if ctry == 'Pakistan':
        return 'Unavailable', 'Pakistan nationals are not part of the IPL player pool', None
    if nm in RELEASED_EARLY:
        return 'Auction pool', RELEASED_EARLY[nm] + ' - political-risk flag', None
    if isinstance(team, str):
        if team == RR_TEAM:
            return 'RR squad', 'Already a Royal - retain/core', team
        if nm in WATCH:
            return 'Watch list', WATCH[nm] + f' ({team}); only if released by 17 Nov', team
        return 'Contracted', f'Under contract with {team} - not expected in auction', team
    if ctry == 'Bangladesh':
        if nm in BAN_REGISTERED_BANGLADESH:
            return 'Auction pool', 'Registered for 2027 auction (Bangladesh) - political-risk flag', None
        return 'Unlikely', 'Bangladesh player not on reported 2027 registration list', None
    note = FLAGS.get(nm, 'Not in any IPL 2026 squad - expected in auction pool (subject to registration by 23 Nov)')
    return 'Auction pool', note, None


out = {}
for role in ['bat', 'bowl']:
    f = pd.read_csv(f'{W}/ratings_{role}.csv', index_col=0)
    cls = f.apply(classify, axis=1, result_type='expand')
    f['avail'], f['avail_note'], f['team27'] = cls[0], cls[1], cls[2]
    f['overseas'] = ~f.country.fillna('').str.startswith('India')
    f.to_csv(f'{W}/ratings_{role}.csv')
    pool = f[f.avail == 'Auction pool'].sort_values('rank')
    f['pool_rank'] = pool['rank'].rank(method='first').astype(int).reindex(f.index)
    f.to_csv(f'{W}/ratings_{role}.csv')
    print(role, f.avail.value_counts().to_dict())
    print(f[f.avail.isin(['Auction pool', 'Watch list'])].sort_values('rank')[
        ['name', 'country', 'avail', 'rank', 'pool_rank', 'RUPI', 'PV_u', 'NV_u', 'RES_u', 'CON', 'p_n', 'p_matches', 'ipl_share', 'boot_top10', 'last_date']].head(25).round(2).to_string())

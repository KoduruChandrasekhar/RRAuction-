"""IPL 2027 mini-auction availability (status as of 16 Nov 2026) - editable in availability_config.json.
Re-run 03_ratings.py after the official retention lists (17 Nov) to refresh every ranking.

Status meanings
  Auction pool  - not under an IPL contract for 2027 and eligible -> can be bought on 16 Dec
  Watch list    - under contract but widely tipped for release (only bid if released by 17 Nov)
  Contracted    - under contract with another franchise (cornerstones are not released before a mini-auction)
  RR squad      - already a Royal
  Unavailable   - retired from IPL / Pakistan national / banned
  Doubtful      - eligible but opted out of IPL 2026 (PSL / possible IPL retirement)
  Unlikely      - eligible in principle but not reported as registered"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, 'availability_config.json'), encoding='utf-8'))


def classify(name, ipl26_team, country):
    team = CFG['trades'].get(name, ipl26_team if isinstance(ipl26_team, str) else None)
    if name in CFG['released']:
        team = None
    if name in CFG['unavailable']:
        return 'Unavailable', CFG['unavailable'][name], team
    if country == 'Pakistan':
        return 'Unavailable', 'Pakistan nationals are not part of the IPL player pool', None
    if team == CFG['rr_team']:
        return 'RR squad', 'Already a Royal - retain / core', team
    if team:
        if name in CFG['watch']:
            return 'Watch list', f"{CFG['watch'][name]} ({team}) - bid only if released on 17 Nov", team
        return 'Contracted', f'Under contract with {team} - not expected at the mini-auction', team
    if name in CFG['released']:
        return 'Auction pool', CFG['released'][name], None
    if name in CFG.get('doubtful', {}):
        return 'Doubtful', CFG['doubtful'][name], None
    if country == 'Bangladesh':
        if name in CFG['bangladesh_registered']:
            return 'Auction pool', 'Registered for the 2027 auction (Bangladesh) - political-risk flag', None
        return 'Unlikely', 'Bangladesh player not on the reported 2027 registration list', None
    return 'Auction pool', CFG['notes'].get(name, 'Not in any IPL 2026 squad - expected in the auction pool (registration closes 23 Nov)'), None

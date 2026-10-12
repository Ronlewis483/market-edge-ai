"""User-selected platforms and diversity-first prop display policy."""
PLATFORMS = {'polymarket': 'Polymarket', 'underdog': 'Underdog',
             'draftkings': 'DraftKings', 'prizepicks': 'PrizePicks', 'fanduel': 'FanDuel'}
BOOKMAKERS = ','.join(PLATFORMS)
PICKEM = {'underdog', 'prizepicks'}

def allowed(key):
    return str(key).lower() in PLATFORMS

def standalone_value(key):
    return allowed(key) and str(key).lower() not in PICKEM

def diversify(rows, limit=5, market_key='market', identity_keys=('event_id', 'player', 'market')):
    """Keep ranked unique picks; category panels provide variety without caps."""
    selected, seen = [], set()
    for row in rows:
        identity = tuple(row.get(key) for key in identity_keys)
        if identity not in seen:
            selected.append(row); seen.add(identity)
        if len(selected) >= limit:
            break
    return selected

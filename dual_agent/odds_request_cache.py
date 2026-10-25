"""Shared 30-minute Odds API cache; exact-game invalidation on explicit request."""
import copy
import hashlib
import json
import threading
import time
import requests
from dual_agent.preferred_sportsbooks import BOOKMAKERS

TTL = 1800
_entries = {}
_locks = {}
_guard = threading.RLock()

def get(url, params=None, cache_ttl=TTL, **kwargs):
    if not str(url).startswith('https://api.the-odds-api.com/'):
        return requests.get(url, params=params, **kwargs)
    params = dict(params or {})
    if str(url).rstrip('/').endswith(('/odds', '/markets')):
        params.pop('regions', None)
        params.setdefault('bookmakers', BOOKMAKERS)
    identity = hashlib.sha256(json.dumps([url, params or {}], sort_keys=True, default=str).encode()).hexdigest()
    with _guard:
        lock = _locks.setdefault(identity, threading.Lock())
    with lock:
        with _guard:
            cached = _entries.get(identity)
            if cached and time.monotonic()-cached[0] < min(TTL, max(0, cache_ttl)):
                return copy.deepcopy(cached[2])
        response = requests.get(url, params=params, **kwargs)
        if response.status_code == 200:
            # Preserve actual retrieval time across cache hits; never relabel a
            # saved response as freshly fetched by the caller.
            from datetime import datetime, timezone
            response.headers['X-Market-Edge-Fetched-At'] = datetime.now(timezone.utc).isoformat()
            with _guard:
                _entries[identity] = (time.monotonic(), str(url), copy.deepcopy(response))
        return response

def refresh_game_props(home, away):
    """Invalidate only cached player markets for an exact matched event."""
    ids = set()
    with _guard:
        for _, _, response in _entries.values():
            try:
                data = response.json()
                events = data if isinstance(data, list) else [data]
                ids.update(str(e['id']) for e in events if isinstance(e, dict)
                           and e.get('home_team') == home and e.get('away_team') == away and e.get('id'))
            except (ValueError, TypeError, KeyError):
                continue
        for identity, entry in list(_entries.items()):
            if any('/events/'+event+'/' in entry[1] for event in ids):
                _entries.pop(identity, None)
    return len(ids)

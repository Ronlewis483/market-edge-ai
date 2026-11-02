"""Cached, last-known-good numeric prop weights from private storage."""
import threading
import time
from dual_agent.sports_connected_winner import ROOT,Store

_CACHE={};_LOCK=threading.RLock()


def weights(sport):
    if sport not in ('NFL','MLB'):return None
    with _LOCK:
        prior=_CACHE.get(sport)
        if prior and time.monotonic()-prior[0]<600:return prior[1]
        value=prior[1] if prior else None
        try:
            incoming=Store().read(ROOT+sport.lower()+'/props/active.json.gz')
            if incoming:
                from dual_agent.player_prop_regression import FEATURES
                if incoming.get('schema')!=1 or incoming.get('features')!=FEATURES:raise ValueError('Invalid prop registry')
                if any(not key.startswith(sport+'|') for key in incoming.get('markets',{})):raise ValueError('Cross-sport prop weights')
                value=incoming
        except Exception:
            pass  # Existing neural/local inference remains available on storage failure.
        _CACHE[sport]=(time.monotonic(),value)
        return value

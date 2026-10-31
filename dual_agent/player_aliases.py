"""Conservative player-name matching, scoped to sport and provider IDs.

No network requests or fuzzy surname-only guesses. Provider-supplied name
variants and confirmed resolutions are reused; ambiguous IDs never resolve.
"""
import json
import re
import threading
import unicodedata
from pathlib import Path

_LOCK = threading.RLock()
_CACHE = None
_PATH = Path('sports_prop_workspace/player_aliases_v1.json')
ALIASES = {'NBA': {'alexsarr':'alexandresarr','nicclaxton':'nicolasclaxton',
                  'acebailey':'airiousbailey','herbjones':'herbertjones',
                  'boneshyland':'nahshonhyland'}}


def tokens(name):
    text = unicodedata.normalize('NFKD', str(name or '')).encode('ascii','ignore').decode().lower()
    if ',' in text:
        last, first = text.split(',', 1)
        # Surname-first format only when the second part is not a suffix.
        if first.strip().strip('.') not in {'jr','sr','ii','iii','iv','v'}:
            text = first+' '+last
    parts = re.findall(r'[a-z0-9]+', text)
    while parts and parts[-1] in {'jr','sr','ii','iii','iv','v'}:
        parts.pop()
    return parts


def name_key(name, sport=None):
    value = ''.join(tokens(name))
    return ALIASES.get(str(sport).upper(), {}).get(value, value)


def _registry():
    global _CACHE
    with _LOCK:
        if _CACHE is None:
            try:
                data = json.loads(_PATH.read_text())
                _CACHE = data if isinstance(data,dict) else {}
            except (OSError,ValueError):
                _CACHE = {}
        return _CACHE


def _remember(scope, alias, identity):
    with _LOCK:
        registry = _registry()
        key = scope+'|'+alias
        previous = set(registry.get(key, []))
        if identity in previous:
            return
        registry[key] = sorted(previous | {identity})
        try:
            _PATH.parent.mkdir(parents=True,exist_ok=True)
            temporary = _PATH.with_suffix('.tmp')
            temporary.write_text(json.dumps(registry,sort_keys=True))
            temporary.replace(_PATH)
        except OSError:
            pass  # In-memory matching remains available if local storage is absent.


def record_names(record):
    names = []
    for field in ('name','full_name','fullName','display_name','displayName','shortName','nickname','player','fullFMLName','firstLastName','lastFirstName','boxscoreName'):
        if isinstance(record.get(field),str) and record[field].strip():
            names.append(record[field])
    for first,last in (('first_name','last_name'),('firstName','lastName')):
        if record.get(first) and record.get(last):
            names.append(str(record[first])+' '+str(record[last]))
    for first in ('useName','nickName'):
        last = record.get('useLastName') or record.get('lastName')
        if record.get(first) and last:
            names.append(str(record[first])+' '+str(last))
    aliases = record.get('aliases', [])
    if isinstance(aliases,str): aliases = [aliases]
    names.extend(n for n in aliases if isinstance(n,str))
    return list(dict.fromkeys(names))


class PlayerResolver:
    def __init__(self, records, sport, provider):
        self.sport = str(sport).upper()
        self.scope = self.sport+'|'+str(provider)
        self.records, self.exact, self.initials = {}, {}, {}
        for record in records:
            if record.get('id') is None:
                continue
            identity = str(record['id'])
            self.records[identity] = record
            for name in record_names(record):
                key = name_key(name,self.sport)
                if key: self.exact.setdefault(key,set()).add(identity)
                parts = tokens(name)
                if len(parts)>=2:
                    initial = parts[0][0]+parts[-1]
                    self.initials.setdefault(initial,set()).add(identity)

    def resolve(self, name, team_id=None):
        key = name_key(name,self.sport)
        ids = set(self.exact.get(key, ()))
        ids.update(identity for identity in _registry().get(self.scope+'|'+key,())
                   if identity in self.records)
        if not ids:
            parts = tokens(name)
            if len(parts)>=2 and len(parts[0])==1:
                ids = set(self.initials.get(parts[0]+parts[-1], ()))
        if team_id is not None:
            ids = {i for i in ids if str((self.records[i].get('team') or {}).get('id',self.records[i].get('team_id'))) == str(team_id)}
        if len(ids)!=1:
            return None
        identity = next(iter(ids))
        _remember(self.scope,key,identity)
        return self.records[identity]


def resolver_from_history(history):
    columns = ['provider_player_id','player']
    if 'provider_player_aliases' in history:
        columns.append('provider_player_aliases')
    records = []
    for row in history[columns].dropna(subset=['provider_player_id','player']).drop_duplicates(['provider_player_id','player']).to_dict('records'):
        aliases = row.get('provider_player_aliases', [])
        records.append({'id':row['provider_player_id'],'name':row['player'],
                        'aliases':aliases if isinstance(aliases,(list,tuple,str)) else []})
    return PlayerResolver(records, 'NFL', 'nflverse')

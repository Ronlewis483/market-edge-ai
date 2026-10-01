"""Resumable historical MLB moneyline snapshots. No credentials are persisted."""
import json
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

MODULE_VERSION=1


def stamp(value):
    result=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if result.tzinfo is None:raise ValueError('Timestamp must include timezone')
    return result.astimezone(timezone.utc)


def team(value):
    name=' '.join(str(value).lower().replace('.','').split())
    return {'oakland athletics':'athletics','sacramento athletics':'athletics'}.get(name,name)


def validate_snapshot(payload, requested):
    if not isinstance(payload,dict) or not isinstance(payload.get('data'),list):raise ValueError('Invalid snapshot')
    snapshot=stamp(payload['timestamp'])
    if snapshot>requested or requested-snapshot>timedelta(minutes=30):raise ValueError('Ineligible snapshot time')
    return snapshot


def extract_quotes(game,payload):
    cap=datetime.strptime(game['timecode'],'%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
    start=stamp(game['start_time']);snapshot=validate_snapshot(payload,cap)
    if cap>=start:raise ValueError('Capture must precede start')
    matches=[]
    for event in payload['data']:
        try:
            if team(event['home_team'])==team(game['home_team']) and team(event['away_team'])==team(game['away_team']) and abs((stamp(event['commence_time'])-start).total_seconds())<=900:matches.append(event)
        except (KeyError,ValueError,TypeError):continue
    if len(matches)!=1:return [],'No unique team/start match; doubleheaders are not guessed'
    event=matches[0];rows=[]
    for book in event.get('bookmakers',[]):
        for market in book.get('markets',[]):
            if market.get('key')!='h2h':continue
            try:
                updated=stamp(market.get('last_update') or book['last_update'])
                if updated>snapshot or updated>=start or cap-updated>timedelta(minutes=30):continue
                if len(market['outcomes'])!=2:continue
                prices={team(o['name']):float(o['price']) for o in market['outcomes']}
                h,a=prices[team(game['home_team'])],prices[team(game['away_team'])]
                if min(h,a)<=1 or not all(map(math.isfinite,[h,a])):continue
                rows.append({'game_id':int(game['game_id']),'captured_at':updated.isoformat(),'bookmaker':book['key'],'home_decimal':h,'away_decimal':a,'snapshot_at':snapshot.isoformat(),'event_id':event['id']})
            except (KeyError,ValueError,TypeError):continue
    return rows,'Matched' if rows else 'No eligible two-sided quotes'


def collect_historical_odds(archive,api_key,max_requests=10,cache_root='mlb_accuracy_results',progress=None):
    max_requests=int(max_requests)
    if not 1<=max_requests<=200:raise ValueError('Request limit must be 1–200')
    games=archive.get('retained_feature_rows',[])
    if not games:raise ValueError('Load the full saved matchup archive first')
    groups={}
    for g in games:
        if int(g['season']) not in {2025,2026}:continue
        cap=datetime.strptime(g['timecode'],'%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
        requested=cap.replace(minute=5*(cap.minute//5),second=0)
        groups.setdefault(requested,[]).append(g)
    # Interleave years and dates rather than spending every small batch on one season.
    by_year={year:sorted(t for t in groups if t.year==year) for year in [2025,2026]}
    ordered=[]
    for i in range(max([len(v) for v in by_year.values()]+[0])):
        for year in [2025,2026]:
            if i<len(by_year[year]):ordered.append(by_year[year][i])
    root=Path(cache_root)/'historical_odds_cache';root.mkdir(parents=True,exist_ok=True)
    bucket=None;errors=[]
    try:
        from dual_agent import supabase_db as storage
        ready=storage.ensure_market_edge_storage_bucket()
        if ready.get('success'):bucket=storage.get_supabase_client().storage.from_(storage.MLB_STORAGE_BUCKET)
    except Exception:errors.append('Permanent cache unavailable; local cache remains usable')
    rows=[];coverage=[];requests=0;credits=0;remaining=None;cached=0;pending=0;verified=False
    for i,requested in enumerate(ordered,1):
        name=requested.strftime('%Y%m%dT%H%M%SZ')+'_us_h2h_decimal_v1.json'
        local=root/name;remote='mlb/historical_odds/'+name;payload=None
        if local.exists():
            try:payload=json.loads(local.read_text());validate_snapshot(payload,requested)
            except Exception:payload=None
        if payload is None and (requests>=max_requests or (remaining is not None and remaining<10)):
            pending+=1;continue
        if payload is None and bucket is not None:
            try:payload=json.loads(bucket.download(remote));validate_snapshot(payload,requested);local.write_text(json.dumps(payload))
            except Exception:payload=None
        if payload is not None:cached+=1
        else:
            if requests>=max_requests or (remaining is not None and remaining<10):pending+=1;continue
            if not api_key:raise ValueError('Set ODDS_API_KEY in Streamlit secrets for uncached requests')
            query=urlencode({'apiKey':api_key,'regions':'us','markets':'h2h','oddsFormat':'decimal','date':requested.isoformat().replace('+00:00','Z')})
            requests+=1
            try:
                with urlopen('https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds?'+query,timeout=25) as response:
                    payload=json.load(response);credits+=int(response.headers.get('x-requests-last',10))
                    value=response.headers.get('x-requests-remaining');remaining=int(value) if value is not None else None
                validate_snapshot(payload,requested);verified=True
                temp=local.with_suffix('.tmp');temp.write_text(json.dumps(payload));temp.replace(local)
                if bucket is not None:
                    try:bucket.upload(remote,json.dumps(payload).encode(),{'content-type':'application/json','upsert':'false'})
                    except Exception:errors.append('Snapshot permanent save failed; local copy retained')
            except HTTPError as exc:
                errors.append(f'HTTP {exc.code}: stopped collection. Check historical access, key, or quota.');pending+=len(ordered)-i+1;break
            except (URLError,TimeoutError,ValueError,KeyError):
                errors.append('Request failed or invalid snapshot returned; cached successes retained');payload=None
        if payload is not None:
            for g in groups[requested]:
                quotes,note=extract_quotes(g,payload);rows.extend(quotes)
                coverage.append({'game_id':g['game_id'],'season':g['season'],'eligible_books':len({r['bookmaker'] for r in quotes}),'status':note})
        if progress:progress(i,len(ordered))
    return {'schema_version':1,'rows':rows,'coverage':coverage,'requests_used':requests,'credits_used_reported':credits,'credits_remaining':remaining,'cached_snapshots':cached,'pending_snapshots':pending,'historical_access_verified':verified,'errors':errors,'maximum_estimated_batch_credits':10*max_requests}

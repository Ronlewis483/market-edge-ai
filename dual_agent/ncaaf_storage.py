"""Private Supabase storage for NCAA jobs, runtime models and frozen forecasts."""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote
import gzip
import hashlib
import json
import os
import threading
import time

PREFIX='ncaaf/learning_v1/'
ROOT=Path('ncaaf_history_cache')
_CACHE=None
_LOCK=threading.RLock()


def setting(name,default=None):
    value=os.environ.get(name)
    if value is not None:return value
    try:
        import streamlit as st
        return st.secrets.get(name,default)
    except Exception:return default


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class Store:
    def __init__(self):
        import requests
        self.url=setting('SUPABASE_URL');self.key=setting('SUPABASE_KEY')
        self.bucket=setting('NCAAF_STORAGE_BUCKET','market-edge-data')
        if not self.url or not self.key:raise RuntimeError('Configure SUPABASE_URL and SUPABASE_KEY for NCAA storage')
        self.base=self.url.rstrip('/')+'/storage/v1/object/'
        self.http=requests.Session()
    def headers(self):return {'apikey':self.key,'Authorization':'Bearer '+self.key}
    def read(self,path):
        r=self.http.get(self.base+'authenticated/'+quote(self.bucket,safe='')+'/'+quote(path,safe='/'),headers=self.headers(),timeout=(5,60))
        if r.status_code in (400,404):
            try:missing=str(r.json().get('code','')).lower() in ('nosuchkey','not_found','404') or 'not found' in str(r.json().get('message','')).lower()
            except ValueError:missing=False
            if r.status_code==404 or missing:return None
        if r.status_code!=200:raise RuntimeError('NCAA storage read failed (HTTP '+str(r.status_code)+')')
        return json.loads(gzip.decompress(r.content))
    def write(self,path,value,replace=False):
        r=self.http.post(self.base+quote(self.bucket,safe='')+'/'+quote(path,safe='/'),
            headers=dict(self.headers(),**{'Content-Type':'application/gzip','x-upsert':str(replace).lower(),'cache-control':'no-cache'}),
            data=gzip.compress(json.dumps(value,allow_nan=False,separators=(',',':')).encode()),timeout=(5,60))
        if not replace and r.status_code in (400,409):
            try:duplicate='already exists' in str(r.json().get('message','')).lower() or str(r.json().get('code','')).lower() in ('duplicate','resourcealreadyexists')
            except ValueError:duplicate=False
            if duplicate or r.status_code==409:return False
        if r.status_code not in (200,201):raise RuntimeError('NCAA storage write failed (HTTP '+str(r.status_code)+')')
        return True
    def list(self,prefix):
        result=[];offset=0
        while True:
            r=self.http.post(self.base+'list/'+quote(self.bucket,safe=''),headers=self.headers(),
                json={'prefix':prefix,'offset':offset,'limit':100,'sortBy':{'column':'name','order':'asc'}},timeout=(5,30))
            if r.status_code!=200:raise RuntimeError('NCAA forecast listing failed (HTTP '+str(r.status_code)+')')
            batch=r.json()
            if not isinstance(batch,list):raise ValueError('Invalid NCAA storage listing')
            result.extend(x for x in batch if str(x.get('name','')).endswith('.json.gz'))
            if len(batch)<100:return result
            offset+=len(batch)


def restore_state(store):
    manifest=store.read(PREFIX+'state/latest.json.gz')
    if manifest is None:return {}
    if manifest.get('schema')!=1:raise ValueError('NCAA state schema mismatch; refusing to reset saved history')
    for name,meta in manifest['files'].items():
        rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or rel.suffix!='.json':raise ValueError('Invalid NCAA state path')
        value=store.read(meta['object'])
        if value is None or digest(value)!=meta['sha256']:raise ValueError('Missing or corrupt NCAA state object; no reset performed')
        path=ROOT.joinpath(*rel.parts);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,allow_nan=False))
    return manifest


def save_state(store,previous=None):
    previous=previous or {};files={}
    for path in sorted(ROOT.rglob('*.json')):
        rel=path.relative_to(ROOT)
        # Runtime and per-request winner snapshots are published separately.
        if rel.parts[0] in ('runtime','forecasts'):continue
        value=json.loads(path.read_text());sha=digest(value);name=rel.as_posix()
        meta={'object':PREFIX+'state/objects/'+sha+'.json.gz','sha256':sha}
        if previous.get('files',{}).get(name,{}).get('sha256')!=sha:store.write(meta['object'],value)
        files[name]=meta
    manifest={'schema':1,'files':files,'saved_at':datetime.now(timezone.utc).isoformat()}
    store.write(PREFIX+'state/latest.json.gz',manifest,replace=True)
    return manifest


def publish_runtime(store,bundle):
    identity=digest(bundle)
    path=PREFIX+'runtime/versions/'+identity+'.json.gz'
    store.write(path,bundle)
    # Publish the pointer only after the immutable version is safely stored.
    store.write(PREFIX+'runtime/latest.json.gz',{'schema':1,'object':path,'sha256':identity},replace=True)


def runtime_bundle(force=False):
    global _CACHE
    with _LOCK:
        if not force and _CACHE and time.monotonic()-_CACHE[0]<900:return _CACHE[1]
        try:
            store=Store();pointer=store.read(PREFIX+'runtime/latest.json.gz')
            if pointer is None:raise RuntimeError('No saved NCAA runtime bundle yet. Run the NCAAF Actions workflow in bootstrap mode first.')
            bundle=store.read(pointer['object'])
            if bundle is None or digest(bundle)!=pointer['sha256'] or bundle.get('schema')!=1:raise ValueError('NCAA runtime bundle failed verification')
            ROOT.joinpath('runtime').mkdir(parents=True,exist_ok=True)
            path=ROOT/'runtime/last_good.json';tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(bundle,allow_nan=False));tmp.replace(path)
            _CACHE=(time.monotonic(),bundle);return bundle
        except Exception:
            # An outage must not silently reset a working saved model/history.
            if _CACHE:return _CACHE[1]
            try:return json.loads((ROOT/'runtime/last_good.json').read_text())
            except (OSError,ValueError):raise RuntimeError('Saved NCAA models unavailable. Check Supabase settings and run the NCAAF workflow; the app does not train models while loading.') from None


def freeze_games(games,store=None,now=None):
    from dual_agent.ncaaf_prediction import stamp
    store=store or Store();now=now or datetime.now(timezone.utc)
    saved=0
    for game in games:
        start=stamp(game.get('commence_time'))
        if not start or start<=now:continue
        identity=digest([str(game['game_id']),game['model_id']])
        row=dict(game,captured_at=now.isoformat(),schema=1)
        saved+=int(store.write(PREFIX+'forecasts/'+identity+'.json.gz',row))
    return saved


def freeze_props(picks,store=None,now=None):
    from dual_agent.ncaaf_prediction import stamp
    store=store or Store();now=now or datetime.now(timezone.utc);saved=0
    for row in picks:
        start,quoted=stamp(row.get('commence_time')),stamp(row.get('quote_time'))
        if not start or not quoted or not quoted<=now<start:continue
        identity=digest([row['game_id'],row['player_id'],row['market'],row['line'],row['side'],row['book'],row['regression_model_id']])
        saved+=int(store.write(PREFIX+'prop_forecasts/'+identity+'.json.gz',dict(row,captured_at=now.isoformat(),schema=1)))
    return saved

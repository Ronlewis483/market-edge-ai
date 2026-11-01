"""NCAAF-only saved-history regression and exact listed pregame props.

No NFL weights, invented lines or fabricated zero appearances. Collection and
training are explicit bounded maintenance actions, not part of winner generation.
"""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import threading

ROOT = Path('ncaaf_history_cache/player_props')
_LOCK = threading.RLock()
MARKETS = {
 'player_pass_yds': ('Passing yards','passing','YDS'),
 'player_pass_tds': ('Passing touchdowns','passing','TD'),
 'player_pass_interceptions': ('Passing interceptions','passing','INT'),
 'player_pass_attempts': ('Passing attempts','passing','C/ATT'),
 'player_pass_completions': ('Passing completions','passing','C/ATT'),
 'player_rush_yds': ('Rushing yards','rushing','YDS'),
 'player_rush_attempts': ('Rushing attempts','rushing','CAR'),
 'player_reception_yds': ('Receiving yards','receiving','YDS'),
 'player_receptions': ('Receptions','receiving','REC'),
}


def _read(name, default):
    try: return json.loads((ROOT/name).read_text())
    except (OSError, ValueError): return default


def _write(name, data):
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/name; tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, allow_nan=False)); tmp.replace(path)


def _weights_path():
    local=ROOT/'weights.json'
    return local if local.exists() else Path(__file__).with_name('ncaaf_player_prop_regression_weights.json')


def _weights():
    try:return json.loads(_weights_path().read_text())
    except (OSError,ValueError):return {}


def _number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError): return None


def parse_boxes(boxes, games, now=None):
    """Only explicit category stats, linked to a completed CFBD game/team."""
    from dual_agent.ncaaf_prediction import stamp, name_key
    now = now or datetime.now(timezone.utc)
    records = {}
    for box in boxes:
        game = games.get(str(box.get('id')))
        if not game or game.get('completed') is not True: continue
        start = stamp(game.get('startDate'))
        if not start or start >= now-timedelta(days=2): continue
        team_ids = {name_key(game.get(side+'Team')):str(game.get(side+'Id')) for side in ('home','away')}
        for team in box.get('teams', []):
            tid = team_ids.get(name_key(team.get('team')))
            if not tid or tid == 'None': continue
            for cat in team.get('categories', []):
                for stat in cat.get('types', []):
                    for market, (_, category, field) in MARKETS.items():
                        if str(cat.get('name','')).lower()!=category or str(stat.get('name','')).upper()!=field: continue
                        for athlete in stat.get('athletes', []):
                            if not athlete.get('id') or not athlete.get('name'): continue
                            value = athlete.get('stat')
                            if field == 'C/ATT':
                                pair = str(value).split('/')
                                if len(pair)!=2: continue
                                completed, attempted = map(_number, pair)
                                if completed is None or attempted is None or not 0<=completed<=attempted: continue
                                value = attempted if market.endswith('attempts') else completed
                            else: value = _number(value)
                            if value is None or not float(value).is_integer(): continue
                            if 'yds' not in market and value<0: continue
                            row = {'player_id':str(athlete['id']), 'player':athlete['name'],
                                'team_id':tid,'team':team['team'], 'game_id':str(game['id']),
                                'date':start.isoformat(),'market':market,'value':float(value)}
                            key = (row['player_id'],row['game_id'],market)
                            # Conflicts are retained as invalid, never last-write-wins.
                            if key in records and records[key] and records[key]['value']!=row['value']: records[key]=None
                            elif key not in records: records[key]=row
    return [r for r in records.values() if r is not None]


def prepare_history(cfbd_key, max_weeks=8, now=None, fit=True):
    """Resume bounded season/week collection, then fit eligible saved markets."""
    if not 1<=int(max_weeks)<=100: raise ValueError('max_weeks must be between 1 and 100')
    from dual_agent.ncaaf_prediction import load_history, _cfbd, stamp
    now = now or datetime.now(timezone.utc)
    with _LOCK:
        schedules, _, issues = load_history(cfbd_key, now)
        games = {str(g['id']):g for g in schedules if g.get('id') is not None and g.get('completed') is True
                 and stamp(g.get('startDate')) and stamp(g['startDate'])<now-timedelta(days=2)}
        buckets=defaultdict(list)
        for g in games.values():
            if g.get('season') is None or g.get('week') is None: continue
            season_type = g.get('seasonType')
            if season_type not in ('regular','postseason'): continue
            buckets[(int(g['season']),season_type,int(g['week']))].append(g)
        manifest = _read('collection.json', {})
        rows = _read('history.json', [])
        index = {(r['player_id'],r['game_id'],r['market']):r for r in rows}
        queried=0
        # Recent seasons first so first batches can help upcoming games sooner.
        for (year,season_type,week), group in sorted(buckets.items(), reverse=True):
            key = f'{year}|{season_type}|{week}'
            signature=hashlib.sha256(json.dumps(sorted(str(g['id']) for g in group)).encode()).hexdigest()
            previous=manifest.get(key,{})
            if isinstance(previous,dict) and previous.get('signature')==signature:
                checked=stamp(previous.get('checked_at'))
                if previous.get('complete') or (checked and now-checked<timedelta(hours=6)): continue
            if queried>=max_weeks: break
            try:
                boxes = _cfbd('/games/players',{'year':year,'week':week,'seasonType':season_type,'classification':'fbs'},cfbd_key,900 if year==now.year else 30*86400)
                parsed = parse_boxes(boxes, games, now)
                # Empty or partial feeds are retried on another preparation run.
                returned = {str(b.get('id')) for b in boxes}
                expected = {str(g['id']) for g in group}
                for row in parsed: index[(row['player_id'],row['game_id'],row['market'])]=row
                manifest[key]={'signature':signature,'complete':expected <= returned,'checked_at':now.isoformat()}
                if not expected <= returned: issues.append(f'{year} {season_type} week {week}: incomplete player box coverage; will retry.')
                queried+=1
                _write('history.json',list(index.values())); _write('collection.json',manifest)
            except Exception as exc:
                issues.append('Player collection stopped: '+str(exc));break
        report={'status':'history_saved','weeks_processed':queried,'saved_stat_rows':len(index),
                'weeks_ready':len(manifest),'weeks_in_schedule':len(buckets),'issues':issues,
                'training_performed':False,'promoted_markets':[], 'checked_at':now.isoformat()}
        if fit and index:
            import pandas as pd
            from dual_agent.train_player_prop_regression import dataset, train
            from dual_agent.player_prop_regression import FEATURES
            markets=train(dataset(pd.DataFrame(index.values())),'NCAAF')
            if markets:
                # NCAAF gets a separate artifact; other sport weights stay untouched.
                _write('weights.json',{'schema':1,'features':FEATURES,'markets':markets})
                report.update(training_performed=True,status='training_completed',
                    promoted_markets=[k.split('|',1)[1] for k,v in markets.items() if v['promoted']],
                    comparison={k:v['checks'] for k,v in markets.items()})
            else: report['status']='waiting_for_training_history'
        if report['training_performed'] and not report['promoted_markets']:
            report['status']='trained_no_eligible_markets'
        _write('latest_training_report.json',report)
        return report


def _odds_get(url, params, **kwargs):
    from dual_agent.odds_request_cache import get
    try:
        return get(url, params=params, **kwargs)
    except Exception as exc:
        # Transport exception messages can include the apiKey query parameter.
        raise RuntimeError('NCAAF odds connection failed ('+type(exc).__name__+'); retry later.') from None


def _event(game, odds_key, cfbd_key, now):
    """Reuse a matched event, or resolve an event by both team IDs and kickoff."""
    if game.get('odds_event_id'): return game['odds_event_id']
    from dual_agent.ncaaf_prediction import _cfbd, attach_markets
    response = _odds_get('https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/events',
        params={'apiKey':odds_key},cache_ttl=300,timeout=(5,20))
    if response.status_code!=200: raise RuntimeError('NCAAF event lookup returned HTTP '+str(response.status_code))
    data=response.json()
    if not isinstance(data,list): raise ValueError('Invalid NCAAF events response')
    empty=dict(moneylines=[],spreads=[],totals=[],market_favorite=None,favorite_probability=None,home_probability=None,away_probability=None)
    candidates=[dict(e,game_id=str(e.get('id')),**empty) for e in data if e.get('id')]
    season=now.year if now.month>=7 else now.year-1
    teams=_cfbd('/teams/fbs',{'year':season},cfbd_key,86400)
    matched=attach_markets([game],candidates,teams)
    return matched[0].get('odds_event_id')


def collect_quotes(game, odds_key, cfbd_key, now=None, supported_markets=None):
    from dual_agent.preferred_sportsbooks import BOOKMAKERS, allowed
    from dual_agent.ncaaf_prediction import stamp
    now=now or datetime.now(timezone.utc)
    start=stamp(game.get('commence_time'))
    if not start or not now<start<=now+timedelta(days=7): raise ValueError('Props support upcoming games within seven days')
    requested=sorted(set(supported_markets or MARKETS) & set(MARKETS))
    if not requested:return []
    event_id=_event(game,odds_key,cfbd_key,now)
    if not event_id: raise ValueError('No unique sportsbook event matched this CFBD game')
    response=_odds_get('https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/events/'+str(event_id)+'/odds',
        params={'apiKey':odds_key,'bookmakers':BOOKMAKERS,'markets':','.join(requested),'oddsFormat':'decimal'},
        cache_ttl=120,timeout=(5,20))
    if response.status_code!=200: raise RuntimeError('NCAAF props returned HTTP '+str(response.status_code)+'. Check your Odds API market access.')
    event=response.json()
    observed=stamp(event.get('commence_time'))
    if str(event.get('id'))!=str(event_id) or not observed or abs((observed-start).total_seconds())>900:
        raise ValueError('Prop event identity or kickoff does not match the selected game')
    from dual_agent.ncaaf_prediction import _cfbd, attach_markets
    season=now.year if now.month>=7 else now.year-1
    teams=_cfbd('/teams/fbs',{'year':season},cfbd_key,86400)
    candidate=dict(event,game_id=str(event_id),moneylines=[],spreads=[],totals=[],market_favorite=None,
                   favorite_probability=None,home_probability=None,away_probability=None)
    if attach_markets([game],[candidate],teams)[0].get('odds_event_id')!=event_id:
        raise ValueError('The prop payload teams do not match the selected game')
    rows=[]
    for book in event.get('bookmakers',[]):
        if not allowed(book.get('key')): continue
        for market in book.get('markets',[]):
            if market.get('key') not in requested: continue
            quoted=stamp(market.get('last_update') or book.get('last_update'))
            if not quoted or not timedelta(0)<=now-quoted<=timedelta(minutes=30) or quoted>=start: continue
            for outcome in market.get('outcomes',[]):
                line,price=_number(outcome.get('point')),_number(outcome.get('price'))
                if outcome.get('name') not in ('Over','Under') or not outcome.get('description') or line is None or price is None or price<=1: continue
                rows.append({'player':outcome['description'],'market':market['key'],'line':line,'side':outcome['name'],
                    'book':book['key'],'sportsbook':book.get('title',book['key']), 'decimal_odds':price,
                    'quote_time':quoted.isoformat(),'event_id':str(event_id)})
    return rows


def score_quotes(game, quotes, roster, history, now=None):
    from dual_agent.player_aliases import PlayerResolver
    from dual_agent.player_prop_regression import estimate
    from dual_agent.ncaaf_prediction import stamp
    now=now or datetime.now(timezone.utc); start=stamp(game['commence_time'])
    records=[]
    for person in roster:
        if person.get('id') is None or str(person.get('team_id')) not in {str(game['home_id']),str(game['away_id'])}: continue
        names={r.get('player') for r in history if str(r.get('player_id'))==str(person['id']) and r.get('player')}
        records.append(dict(person,first_name=person.get('firstName'),last_name=person.get('lastName'),aliases=sorted(names)))
    resolver=PlayerResolver(records,'NCAAF','cfbd')
    issues=[]; scored=[]
    cutoff=min(start,now)-timedelta(days=2)
    for quote in quotes:
        qtime=stamp(quote.get('quote_time'))
        if not qtime or not timedelta(0)<=now-qtime<=timedelta(minutes=30):continue
        person=resolver.resolve(quote['player'])
        if person is None:
            issues.append('No unique current-roster identity for '+quote['player']);continue
        rows=[r for r in history if str(r['player_id'])==str(person['id']) and r['market']==quote['market'] and stamp(r['date']) and stamp(r['date'])<cutoff and str(r['game_id'])!=str(game['game_id'])]
        unique={str(r['game_id']):r for r in rows}
        rows=sorted(unique.values(),key=lambda r:(r['date'],r['game_id']))
        if len(rows)<8:
            issues.append(quote['player']+': fewer than eight explicitly recorded prior category results');continue
        # A historical name alone cannot establish a current role after a transfer.
        recent=rows[-1]
        if str(recent['team_id'])!=str(person['team_id']) or (now-stamp(recent['date'])).days>60:
            issues.append(quote['player']+': no recent recorded category result for the current team');continue
        prediction=estimate([r['value'] for r in rows],quote['line'],quote['side'],'NCAAF',quote['market'],
                            cutoff=min(start,now), artifact_path=_weights_path())
        if prediction is None:
            issues.append(MARKETS[quote['market']][0]+': no eligible NCAAF regression weights (train or review the saved training report)');continue
        chance=prediction['historical_support']; push=prediction['estimated_push_chance']
        scored.append(dict(quote,**prediction,player_id=str(person['id']),team_id=str(person['team_id']),
            label=MARKETS[quote['market']][0], game_id=str(game['game_id']),commence_time=game['commence_time'],
            estimated_return_per_unit=(1-push)*(chance*quote['decimal_odds']-1),
            participation_status='Provisional: current roster and recent recorded stats; injuries and starter status unverified'))
    # Keep exact book/side/line offers, sorted by estimated cover chance.
    unique={(r['event_id'],r['player_id'],r['market'],r['line'],r['side'],r['book']):r for r in scored}
    return sorted(unique.values(),key=lambda r:(-r['historical_support'],-r['decimal_odds'])),list(dict.fromkeys(issues))


def generate_props(game, cfbd_key, odds_key, now=None):
    from dual_agent.ncaaf_prediction import _cfbd
    now=now or datetime.now(timezone.utc)
    if not odds_key or not cfbd_key: raise ValueError('CFBD_API_KEY and ODDS_API_KEY are required for NCAAF props')
    with _LOCK: history=_read('history.json',[])
    if not history: return {'picks':[],'issues':['Prepare NCAAF player history and train the model first.'],'checked_at':now.isoformat()}
    with _LOCK: weights=_weights()
    eligible=[market for market in MARKETS if weights.get('markets',{}).get('NCAAF|'+market,{}).get('promoted')]
    if not eligible:
        return {'picks':[],'issues':['No NCAAF markets passed the saved regression checks. Prepare/train history or review the training comparison; no sportsbook prop request was made.'],'checked_at':now.isoformat()}
    quotes=collect_quotes(game,odds_key,cfbd_key,now,supported_markets=eligible)
    if not quotes: return {'picks':[],'issues':['No fresh supported player lines returned by the configured sportsbooks.'],'checked_at':now.isoformat()}
    season=now.year if now.month>=7 else now.year-1
    roster=[]
    for side in ('home','away'):
        people=_cfbd('/roster',{'team':game[side+'_team'],'year':season},cfbd_key,86400)
        roster.extend(dict(p,team_id=str(game[side+'_id'])) for p in people)
    picks,issues=score_quotes(game,quotes,roster,history,now)
    result={'picks':picks,'issues':issues,'listed_offers':len(quotes),'checked_at':now.isoformat(),
        'limitations':'Only explicitly recorded category results are used; missing appearances are not zero. College injury/lineup availability is unverified. Probabilities are experimental, not established betting accuracy.'}
    with _LOCK:_write('game_'+str(game['game_id'])+'.json',result)
    return result



def _credentials():
    import os
    import streamlit as st
    def setting(*names):
        for name in names:
            value=os.environ.get(name)
            if value:return value
            try:
                value=st.secrets.get(name)
                if value:return value
            except Exception:pass
        return None
    return setting('CFBD_API_KEY'),setting('ODDS_API_KEY','THE_ODDS_API_KEY')


def _ui_pool():
    import streamlit as st
    from concurrent.futures import ThreadPoolExecutor
    @st.cache_resource(show_spinner=False)
    def pool():return ThreadPoolExecutor(max_workers=2)
    return pool()


def render_training_controls():
    import streamlit as st
    with st.expander('NCAAF player history and regression training'):
        st.caption('Collect completed-game player stats with CFBD_API_KEY and train college-only regression weights. This is separate from winner prediction generation and does not request sportsbook odds.')
        limit=st.number_input('Maximum player-history weeks per preparation run',min_value=1,max_value=100,value=8,step=1,key='ncaaf_prop_weeks')
        future=st.session_state.get('ncaaf_prop_training_future')
        if st.button('Prepare / train NCAAF player model',key='ncaaf_prepare_player_model',disabled=future is not None and not future.done()):
            key,_=_credentials()
            if not key:st.error('Add CFBD_API_KEY to app secrets first.')
            else:
                st.session_state['ncaaf_prop_training_future']=_ui_pool().submit(prepare_history,key,int(limit))
                future=st.session_state['ncaaf_prop_training_future']
        st.button('Show latest training status',key='ncaaf_prop_training_status')
        if future is not None and future.done():
            try:st.session_state['ncaaf_prop_training_report']=future.result()
            except Exception as exc:st.error('NCAAF player preparation failed: '+str(exc))
            st.session_state.pop('ncaaf_prop_training_future',None);future=None
        if future is not None:st.caption('Preparing saved history and training in the background. Use Show latest training status; the page does not refresh automatically.')
        report=st.session_state.get('ncaaf_prop_training_report') or _read('latest_training_report.json',{})
        if report:
            st.write('Status: '+report.get('status','Unknown'))
            st.caption(f"{report.get('saved_stat_rows',0):,} saved stat rows \xb7 {report.get('weeks_ready',0)} saved week batches \xb7 {len(report.get('promoted_markets',[]))} eligible markets")
            for issue in report.get('issues',[]):st.warning(str(issue))
            st.json(report)
        saved_weights=_weights()
        if saved_weights.get('markets'):
            st.download_button('Download NCAAF regression weights',data=json.dumps(saved_weights,allow_nan=False),
                file_name='ncaaf_player_prop_regression_weights.json',mime='application/json',key='ncaaf_prop_weights_download')
            st.caption('To preserve trained weights across app restarts, save the downloaded file in dual_agent/ncaaf_player_prop_regression_weights.json. Player histories still need persistent local storage or recollection.')
        st.caption('Collection resumes from saved week batches. More than one bounded run may be needed. A market needs at least 700 training examples and sufficient chronological validation/test coverage. Missing stat appearances are not filled with zeros. Weights and history are stored locally; this does not configure Supabase backup or scheduled retraining.')


def render_props_panel(game):
    import streamlit as st
    from dual_agent.ncaaf_prediction import stamp
    now=datetime.now(timezone.utc); gid=str(game['game_id'])
    st.markdown('#### \U0001f3af NCAAF player props')
    st.caption('College-only regression \xb7 exact listed Over/Under lines \xb7 no NFL weights')
    key='ncaaf_props_result_'+gid
    jobs=st.session_state.setdefault('ncaaf_prop_jobs',{})
    future=jobs.get(gid)
    if st.button('Generate / refresh this game\u2019s player props',key='ncaaf_prop_scan_'+gid,disabled=future is not None and not future.done()):
        cfbd,odds=_credentials()
        if not cfbd or not odds:st.error('CFBD_API_KEY and ODDS_API_KEY are required.')
        else:
            jobs[gid]=_ui_pool().submit(generate_props,dict(game),cfbd,odds)
            future=jobs[gid]
    st.button('Show completed props',key='ncaaf_prop_show_'+gid)
    if future is not None and future.done():
        try:st.session_state[key]=future.result()
        except Exception as exc:st.session_state[key]={'picks':[],'issues':['Player props unavailable: '+str(exc)]}
        jobs.pop(gid,None);future=None
    if future is not None:st.caption('Checking this game in the background. Existing cards remain visible; use Show completed props when ready.')
    result=st.session_state.get(key) or _read('game_'+gid+'.json',{})
    if not result:
        st.info('Prepare/train the NCAAF player model above, then generate this game\u2019s listed player props.')
        return
    picks=[r for r in result.get('picks',[]) if stamp(r.get('quote_time')) and timedelta(0)<=now-stamp(r['quote_time'])<=timedelta(minutes=30)
           and stamp(r.get('commence_time')) and now<stamp(r['commence_time'])]
    labels=['All supported markets']+sorted({r['label'] for r in picks})
    selected=st.selectbox('Prop category',labels,key='ncaaf_prop_market_'+gid)
    if selected!='All supported markets':picks=[r for r in picks if r['label']==selected]
    st.caption('Availability is provisional: roster identity and recent recorded stats are checked, but injuries and starters are not verified. Estimates are not established betting accuracy.')
    if not picks:
        st.info('No fresh modeled props are available for this selection. See the input/training details below.')
    else:
        st.caption(f'{len(picks)} fresh listed offers \xb7 strongest estimated cover chances first')
        for row in picks[:20]:
            with st.container(border=True):
                st.markdown('**'+row['player']+' \xb7 '+row['side'].upper()+f" {row['line']:g} "+row['label']+'**')
                st.write(f"Estimated cover chance: **{row['historical_support']:.1%}** \xb7 Projected total: **{row['projection']:.1f}**")
                st.caption(f"{row['sportsbook']} \xb7 decimal odds {row['decimal_odds']:g} \xb7 {row['sample_size']} prior recorded results")
                st.caption('Quote '+stamp(row['quote_time']).astimezone(__import__('zoneinfo').ZoneInfo('America/Chicago')).strftime('%b %d, %I:%M %p CT'))
                if row['book'] not in ('underdog','prizepicks'):
                    st.caption(f"Estimated net per $1: {row['estimated_return_per_unit']:+.2f} \xb7 push chance {row['estimated_push_chance']:.1%}")
                st.caption('Model: '+row['regression_model_id'])
    with st.expander('NCAAF prop input and training details'):
        for issue in result.get('issues',[]):st.warning(str(issue))
        st.caption(result.get('limitations',''))
        if len(result.get('picks',[]))>len(picks):st.caption('Expired or filtered offers are omitted from this view.')
    st.caption('Refreshing this game can spend Odds API credits. Viewing cached cards makes no requests. Listed offers do not verify that a platform accepts a combined ticket.')

if __name__=='__main__':
    import os
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-weeks',type=int,default=8)
    args=parser.parse_args()
    print(json.dumps(prepare_history(os.environ.get('CFBD_API_KEY'),args.max_weeks),indent=2))

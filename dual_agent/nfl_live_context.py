"""Fresh NFL results and timestamped pregame context. No arbitrary probability boosts.
Feeds are read-only. Polling runs while the Streamlit page is open.
"""
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests
import streamlit as st

TEAMS = dict(zip(
 'ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS'.split(),
 ['Arizona Cardinals','Atlanta Falcons','Baltimore Ravens','Buffalo Bills','Carolina Panthers','Chicago Bears','Cincinnati Bengals','Cleveland Browns','Dallas Cowboys','Denver Broncos','Detroit Lions','Green Bay Packers','Houston Texans','Indianapolis Colts','Jacksonville Jaguars','Kansas City Chiefs','Los Angeles Rams','Los Angeles Chargers','Las Vegas Raiders','Miami Dolphins','Minnesota Vikings','New England Patriots','New Orleans Saints','New York Giants','New York Jets','Philadelphia Eagles','Pittsburgh Steelers','Seattle Seahawks','San Francisco 49ers','Tampa Bay Buccaneers','Tennessee Titans','Washington Commanders']))
ALIASES = {'LAR':'LA','OAK':'LV','SD':'LAC','STL':'LA','JAC':'JAX','WSH':'WAS'}

def team_code(name):
    value = str(name).strip()
    if value.upper() in TEAMS or value.upper() in ALIASES:
        return ALIASES.get(value.upper(),value.upper())
    return next((k for k,v in TEAMS.items() if v.casefold()==value.casefold()),None)

def _get(url, **kwargs):
    response = requests.get(url, timeout=20, **kwargs)
    if response.status_code != 200:
        raise ValueError(f'Feed returned HTTP {response.status_code}')
    return response

@st.cache_data(ttl=180, show_spinner=False)
def fresh_schedule():
    frame = pd.read_csv(StringIO(_get('https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv').text))
    local = pd.to_datetime(frame.gameday.astype(str)+' '+frame.gametime.fillna('12:00').astype(str),errors='coerce')
    frame['start_time'] = local.dt.tz_localize('America/New_York',ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC')
    return frame, pd.Timestamp.now(tz='UTC').isoformat()

def completed_history(schedule, now=None):
    current = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    season = current.year if current.month>=8 else current.year-1
    frame = schedule.loc[(schedule.season>=season-2)&(schedule.season<=season)&schedule.game_type.isin(['REG','WC','DIV','CON','SB','POST'])].copy()
    # nflverse publishes final game results; never fabricate missing scores.
    for col in ['home_score','away_score','result']:
        frame[col] = pd.to_numeric(frame[col],errors='coerce')
    frame = frame.loc[(frame.start_time<current)&frame[['home_score','away_score','result']].notna().all(axis=1)].copy()
    frame['home_team'] = frame.home_team.map(lambda x:TEAMS.get(ALIASES.get(x,x)))
    frame['away_team'] = frame.away_team.map(lambda x:TEAMS.get(ALIASES.get(x,x)))
    frame['status'] = 'closed'
    return frame.dropna(subset=['home_team','away_team']).drop_duplicates('game_id').sort_values('start_time')

@st.cache_data(ttl=180, show_spinner=False)
def player_feed(kind, season):
    try:
        url = f'https://github.com/nflverse/nflverse-data/releases/download/{kind}/{kind}_{season}.csv'
        return pd.read_csv(StringIO(_get(url).text),dtype={'gsis_id':str,'espn_id':str}), None
    except Exception as exc:
        return pd.DataFrame(), f'{kind}: {type(exc).__name__}; current-season feed unavailable'

@st.cache_data(ttl=86400, show_spinner=False)
def stadium_locations():
    return pd.read_csv(StringIO(_get('https://raw.githubusercontent.com/greerreNFL/Stadiums/main/data/stadiums.csv').text))

@st.cache_data(ttl=180, show_spinner=False)
def kickoff_weather(stadium_id, kickoff):
    try:
        venues = stadium_locations()
        selected = venues.loc[venues.stadium_id==stadium_id]
        if len(selected)!=1:
            return {'issue':'No exact stadium-coordinate match'}
        venue = selected.iloc[0]
        lat,lon = float(venue.lat),float(venue.lon)
        if not np.isfinite([lat,lon]).all():
            return {'issue':'Stadium coordinates unavailable'}
        payload = _get('https://api.open-meteo.com/v1/forecast',params={'latitude':lat,'longitude':lon,'hourly':'temperature_2m,precipitation_probability,wind_speed_10m','timezone':'UTC','forecast_days':16,'temperature_unit':'fahrenheit','wind_speed_unit':'mph'}).json()
        hourly = payload.get('hourly',{})
        times = pd.to_datetime(hourly.get('time',[]),utc=True,errors='coerce')
        target = pd.Timestamp(kickoff)
        if len(times)==0:
            return {'issue':'Kickoff forecast unavailable'}
        index = int(np.argmin(abs(times-target)))
        if abs(times[index]-target)>pd.Timedelta(hours=1):
            return {'issue':'Kickoff outside available forecast'}
        return {'source':'Open-Meteo kickoff forecast','fetched_at':pd.Timestamp.now(tz='UTC').isoformat(), 'forecast_time':times[index].isoformat(),
                'temperature_f':hourly['temperature_2m'][index], 'wind_mph':hourly['wind_speed_10m'][index], 'rain_chance':hourly['precipitation_probability'][index],
                'venue':venue.stadium_name, 'roof_type':str(venue.roof_type), 'roof_status':'Actual game-day roof status not verified'}
    except Exception as exc:
        return {'issue':f'Weather feed unavailable ({type(exc).__name__})'}

def _clean(value):
    return None if value is None or pd.isna(value) else str(value)

def team_context(code, depth, injuries, week, game_type, now):
    result = {'team':TEAMS.get(code,code),'qb':None,'depth':[],'injuries':[], 'issues':[]}
    if not depth.empty and {'team','dt','pos_abb','pos_rank','player_name'}.issubset(depth):
        rows = depth.loc[depth.team.map(team_code)==code].copy()
        rows['stamp'] = pd.to_datetime(rows.dt,utc=True,errors='coerce')
        rows = rows.loc[rows.stamp<=now]
        if not rows.empty:
            latest = rows.stamp.max()
            rows = rows.loc[rows.stamp==latest]
            result['depth_updated_at'] = latest.isoformat()
            result['depth'] = rows[['player_name','pos_abb','pos_rank']].where(pd.notna(rows),None).to_dict('records')
            if now-latest>pd.Timedelta(hours=72):
                result['issues'].append('Depth chart older than 72 hours')
            qbs = rows.loc[(rows.pos_abb=='QB')&(pd.to_numeric(rows.pos_rank,errors='coerce')==1)].drop_duplicates('player_name')
            if len(qbs)==1:
                qb=qbs.iloc[0]
                result['qb']={'name':qb.player_name,'gsis_id':_clean(qb.get('gsis_id')),'designation':'Expected starter from depth chart; final starter not confirmed','injury_status':'Not verified','practice_status':'Not verified'}
    if result['qb'] is None:
        result['issues'].append('Expected starting quarterback not resolved from current depth chart')
    if not injuries.empty and {'team','week','season_type','date_modified','full_name'}.issubset(injuries):
        injury_type = 'REG' if game_type=='REG' else 'POST'
        rows = injuries.loc[(injuries.team.map(team_code)==code)&(pd.to_numeric(injuries.week,errors='coerce')==week)&(injuries.season_type==injury_type)].copy()
        rows['stamp'] = pd.to_datetime(rows.date_modified,utc=True,errors='coerce')
        rows = rows.loc[rows.stamp<=now].sort_values('stamp').drop_duplicates('full_name',keep='last')
        for _,r in rows.iterrows():
            item={'player':r.full_name,'gsis_id':_clean(r.get('gsis_id')),'position':_clean(r.get('position')),'status':_clean(r.get('report_status')),'injury':_clean(r.get('report_primary_injury')) or _clean(r.get('practice_primary_injury')),'practice':_clean(r.get('practice_status')),'updated_at':r.stamp.isoformat()}
            result['injuries'].append(item)
            qb=result['qb']
            if qb and ((qb['gsis_id'] and qb['gsis_id']==item['gsis_id']) or qb['name'].casefold()==str(item['player']).casefold()):
                qb.update(injury_status=item['status'] or 'Designation pending',practice_status=item['practice'] or 'Practice report pending',injury=item['injury'],report_updated_at=item['updated_at'])
        if rows.empty:
            result['issues'].append('No current-week team injury/practice report returned; healthy status not assumed')
        elif now-rows.stamp.max()>pd.Timedelta(hours=72):
            result['issues'].append('Injury/practice report older than 72 hours')
        elif result['qb'] and result['qb']['injury_status']=='Not verified':
            result['qb']['injury_status']='Not listed in returned current-week injury report; not a medical clearance'
    else:
        result['issues'].append('Current injury/practice feed unavailable')
    return result

def market_context(game, odds):
    frame = odds.copy() if isinstance(odds,pd.DataFrame) else pd.DataFrame(odds)
    if frame.empty:return {'issue':'No sportsbook quotes returned'}
    frame=frame.loc[(frame.home_team==game['home_team'])&(frame.away_team==game['away_team'])].copy()
    if 'commence_time' in frame:
        stamps=pd.to_datetime(frame.commence_time,utc=True,errors='coerce')
        frame=frame.loc[abs(stamps-pd.Timestamp(game['commence_time']))<=pd.Timedelta(minutes=15)]
    pairs=[]
    def implied(value):
        v=float(value)
        if not np.isfinite(v) or abs(v)<100:raise ValueError('Invalid American odds')
        return -v/(-v+100) if v<0 else 100/(v+100)
    for _,r in frame.drop_duplicates('sportsbook').iterrows():
        try:
            h,a=implied(r.home_moneyline),implied(r.away_moneyline)
            pairs.append({'sportsbook':r.sportsbook,'home_probability':h/(h+a),'away_probability':a/(h+a),'home_moneyline':float(r.home_moneyline),'away_moneyline':float(r.away_moneyline)})
        except (ValueError,TypeError):pass
    if not pairs:return {'issue':'No usable paired sportsbook prices'}
    hp=float(np.median([q['home_probability'] for q in pairs]))
    return {'home_probability':hp,'away_probability':1-hp,'books':pairs,'source':'Freshly fetched Odds API moneylines; margin removed per book','fetched_at':pd.Timestamp.now(tz='UTC').isoformat(), 'gap':abs(hp-float(game['home_win_probability']))}

def attach_context(predictions, odds):
    if predictions is None or predictions.empty:return predictions
    schedule, fetched = fresh_schedule()
    seasons = sorted({int(s) for s in schedule.loc[(schedule.start_time>pd.Timestamp.now(tz='UTC'))&(schedule.start_time<=pd.Timestamp.now(tz='UTC')+pd.Timedelta(days=8)),'season']})
    feeds={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={(season,kind):pool.submit(player_feed,kind,season) for season in seasons for kind in ['depth_charts','injuries']}
        for pair,future in pending.items():feeds[pair]=future.result()
    frame=predictions.copy();contexts=[]
    weather_jobs={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _,game in frame.iterrows():
            home,away=team_code(game.home_team),team_code(game.away_team)
            matches=schedule.loc[(schedule.home_team.map(team_code)==home)&(schedule.away_team.map(team_code)==away)&(abs(schedule.start_time-pd.Timestamp(game.commence_time))<=pd.Timedelta(minutes=15))]
            if len(matches)==1:
                match=matches.iloc[0]
                key=(home,away,game.commence_time.isoformat())
                weather_jobs[key]=pool.submit(kickoff_weather,match.stadium_id,game.commence_time.isoformat())
        weather_results={key:future.result() for key,future in weather_jobs.items()}

    for _,game in frame.iterrows():
        now=pd.Timestamp.now(tz='UTC'); home,away=team_code(game.home_team),team_code(game.away_team)
        matches=schedule.loc[(schedule.home_team.map(team_code)==home)&(schedule.away_team.map(team_code)==away)&(abs(schedule.start_time-pd.Timestamp(game.commence_time))<=pd.Timedelta(minutes=15))]
        context={'fetched_at':fetched,'market':market_context(game,odds),'issues':[],'probability_use':'Current team-history model remains the probability equation. Context controls readiness and explains gaps; no invented injury/weather percentage adjustments.'}
        if len(matches)!=1:
            context['issues'].append('No unique current schedule match; QB, injury and venue context not attached to a guessed game')
        else:
            match=matches.iloc[0];season=int(match.season)
            depth,de=feeds.get((season,'depth_charts'),(pd.DataFrame(),'Depth feed unavailable'))
            injuries,ie=feeds.get((season,'injuries'),(pd.DataFrame(),'Injury feed unavailable'))
            context['issues'] += [e for e in [de,ie] if e]
            context['home']=team_context(home,depth,injuries,int(match.week),match.game_type,now)
            context['away']=team_context(away,depth,injuries,int(match.week),match.game_type,now)
            context['weather']=weather_results.get((home,away,game.commence_time.isoformat()),{'issue':'Weather lookup unavailable'})
            for side in ['home','away']:
                context['issues']+=context[side]['issues']
                qb=context[side].get('qb')
                if qb and str(qb['injury_status']).casefold() in ['out','doubtful','questionable']:
                    context['issues'].append(f"{qb['name']}: {qb['injury_status']}; expected QB availability unresolved")
            if context['weather'].get('issue'):context['issues'].append(context['weather']['issue'])
        context['readiness']='Context needs review' if context['issues'] else 'Context retrieved; final availability may still change'
        contexts.append(context)
    frame['game_context']=contexts
    return frame

"""Independent point-margin estimates from prior completed scoring data."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_margin(frame,columns,time,home_score,away_score,cutoff):
    frame=frame.copy()
    frame[time]=pd.to_datetime(frame[time],utc=True,errors='coerce')
    for c in columns+[home_score,away_score]:frame[c]=pd.to_numeric(frame[c],errors='coerce')
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=columns+[time,home_score,away_score])
    frame=frame.loc[frame[time]<pd.Timestamp(cutoff)].sort_values(time)
    if len(frame)<150:raise ValueError('Insufficient completed scoring history')
    boundary=frame.iloc[int(.8*len(frame))][time]
    train=frame[frame[time]<boundary];test=frame[frame[time]>=boundary]
    if len(train)<100 or len(test)<30:raise ValueError('Insufficient held-out margins')
    model=make_pipeline(StandardScaler(),Ridge(alpha=10.))
    model.fit(train[columns],train[home_score]-train[away_score])
    residuals=(test[home_score]-test[away_score]).to_numpy()-model.predict(test[columns])
    model.fit(frame[columns],frame[home_score]-frame[away_score])
    return model,residuals


def estimate(home,away,start,margin,residuals,home_line,books,now):
    from .game_prediction_addons import nfl_cover_distribution
    hc,ac,push=nfl_cover_distribution(margin,residuals,home_line)
    home_side=hc>=ac
    return dict(home_team=home,away_team=away,commence_time=pd.Timestamp(start).isoformat(),
        spread_team=home if home_side else away,spread_line=home_line if home_side else -home_line,
        cover_chance=(hc if home_side else ac)/(1-push),push_chance=push,
        projected_home_margin=float(margin),books=books,captured_at=pd.Timestamp(now).isoformat(),
        validated=False,method='Prior scoring margin model with chronological held-out errors',sample_size=len(residuals))


def nba_spreads(result,key,now=None):
    from .nba_research import NBA_LIVE_FEATURE_COLUMNS,build_balldontlie_future_matchup_features,normalize_nba_team_name
    from .game_prediction_addons import fetch_spread_events,paired_spread_lines,matching_game_rows,records
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    games=records(result.get('all_predictions',result.get('predictions')))
    if not games:return {'picks':[]}
    cutoff=min(pd.Timestamp(g['commence_time']).normalize() for g in games)
    model,residuals=fit_margin(result['feature_games'],NBA_LIVE_FEATURE_COLUMNS,'game_date','home_points','away_points',min(now,cutoff))
    events=fetch_spread_events('NBA',key);picks=[]
    for game in games:
        start=pd.Timestamp(game['commence_time'])
        if start<=now:continue
        matched=matching_game_rows(game,events)
        if len(matched)!=1:continue
        lines=paired_spread_lines(matched[0],now)
        if not lines:continue
        try:
            x=build_balldontlie_future_matchup_features(result['historical_games'],normalize_nba_team_name(game['home_team']),normalize_nba_team_name(game['away_team']),start.date())
            margin=float(model.predict(x[NBA_LIVE_FEATURE_COLUMNS])[0])
            line,books=sorted(lines,key=lambda v:(-len(v[1]),abs(v[0]),v[0]))[0]
            picks.append(estimate(game['home_team'],game['away_team'],start,margin,residuals,line,len(books),now))
        except (KeyError,ValueError,TypeError):continue
    return {'picks':picks}


def ncaaf_spreads(games,records,now):
    from .ncaaf_prediction import build_dataset
    rows,future,_=build_dataset(records,now)
    columns=['x'+str(i) for i in range(7)]
    frame=pd.DataFrame([dict(zip(columns,r['x']),time=r['start'],hs=r['home_score'],aws=r['away_score']) for r in rows])
    model,residuals=fit_margin(frame,columns,'time','hs','aws',pd.Timestamp(now))
    upcoming={r['id']:r for r in future};output=[]
    for game in games:
        game=dict(game);quotes=game.get('spreads') or [];r=upcoming.get(str(game['game_id']))
        if quotes and r:
            try:
                grouped={}
                for q in quotes:
                    line=float(q['home_line']);other=float(q['away_line'])
                    at=pd.Timestamp(q['Quote UTC'])
                    if not np.isfinite([line,other]).all() or abs(line+other)>1e-8:continue
                    if not pd.Timedelta(0)<=pd.Timestamp(now)-at<=pd.Timedelta(minutes=30):continue
                    grouped.setdefault(line,set()).add(q['book'])
                if grouped:
                    line,books=sorted(grouped.items(),key=lambda v:(-len(v[1]),abs(v[0]),v[0]))[0]
                    margin=float(model.predict(pd.DataFrame([r['x']],columns=columns))[0])
                    game['spread_prediction']=estimate(game['home_team'],game['away_team'],game['commence_time'],margin,residuals,line,len(books),now)
            except (KeyError,ValueError,TypeError):pass
        output.append(game)
    return output


def render_spread(st,spread):
    if not spread:return
    with st.expander('📏 Spread prediction',expanded=True):
        st.markdown(f"**{spread['spread_team']} {spread['spread_line']:+g}**")
        st.write(f"Estimated cover chance: {spread['cover_chance']:.1%} · excludes pushes")
        st.caption(f"Projected home margin: {spread['projected_home_margin']:+.1f} · {spread['books']} sportsbook(s)")
        st.caption('Spread estimates are separate from winner confidence; spread accuracy is not established.')

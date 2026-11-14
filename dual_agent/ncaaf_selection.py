"""Prospective NCAA winner selection; independent of NFL models and evidence."""
from datetime import datetime, timezone
import math
import pandas as pd

RULE='ncaaf_winner_confidence_65_v1'
THRESHOLD=.65


def finite(value):
    if value is None or isinstance(value,bool):return None
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (ValueError,TypeError):return None


def decision(probability):
    p=finite(probability)
    if p is None or not 0<p<1:raise ValueError('Invalid NCAA winner probability')
    confidence=max(p,1-p)
    return dict(rule_id=RULE,threshold=THRESHOLD,selected=confidence>=THRESHOLD,
                confidence=confidence,validation_status='prospective_evaluation_pending')


def tag(game):
    result=dict(game,winner_selection=decision(game['model_home_probability']))
    quotes=game.get('moneylines') or []
    p=finite(game.get('home_probability'))
    result['selection_market_home_probability']=p if quotes and p is not None and 0<p<1 else None
    return result


def summarize(records):
    earliest={}
    for row in records:
        try:
            if row.get('status')!='Graded' or row.get('home_won') not in (0,1):continue
            if row.get('source')=='historical_reconstruction' or row.get('historical_reconstruction'):continue
            captured=pd.Timestamp(row['captured_at']);start=pd.Timestamp(row['commence_time'])
            if captured.tzinfo is None or start.tzinfo is None or not captured<start:continue
            expected=decision(row['model_home_probability']);saved=row.get('winner_selection') or {}
            if any(saved.get(k)!=expected[k] for k in ('rule_id','threshold','selected')):continue
            key=(str(row['game_id']),row['model_id'])
            if key not in earliest or captured<pd.Timestamp(earliest[key]['captured_at']):earliest[key]=row
        except (ValueError,TypeError,KeyError):continue
    def score(rows):
        hits=sum((r['model_home_probability']>=.5)==bool(r['home_won']) for r in rows)
        return dict(games=len(rows),correct=hits,accuracy=hits/len(rows) if rows else None)
    def cohort(rows):
        selected=[r for r in rows if r['winner_selection']['selected']]
        paired=[r for r in selected if finite(r.get('selection_market_home_probability')) is not None
                and 0<r['selection_market_home_probability']<1 and r['selection_market_home_probability']!=.5]
        hits=sum((r['selection_market_home_probability']>.5)==bool(r['home_won']) for r in paired)
        return dict(all_tagged=score(rows),selected=score(selected),
            coverage=len(selected)/len(rows) if rows else None,
            matched_market=dict(model=score(paired),favorite_correct=hits,
                favorite_accuracy=hits/len(paired) if paired else None),
            missing_market_games=len(selected)-len(paired))
    groups={}
    for row in earliest.values():groups.setdefault(row['model_id'],[]).append(row)
    # Different model versions may forecast the same game; never pool those as independent games.
    return dict(rule_id=RULE,threshold=THRESHOLD,status='collecting_prospective_results',
        by_model={k:cohort(v) for k,v in groups.items()},
        limitation='65% is a fixed evaluation rule, not a validated NCAA win rate. Missing pregame quotes are excluded from market comparisons. Historical results remain separate.')


def weekly_winner(read,write,action,active,now):
    local=pd.Timestamp(now).tz_convert('America/Chicago')
    monday=local.normalize()-pd.Timedelta(days=local.weekday())
    previous=read('weekly_winner_training.json',{})
    if previous and previous.get('schema')!=1:raise ValueError('Invalid NCAA weekly state')
    if previous.get('week_start')==monday.date().isoformat():
        winner=active;report=dict(status='weekly_training_not_due',last_training=previous.get('last_training'))
    else:
        winner,report=action()
        # train_winner returns candidate_id only after a completed fit. Readiness waits retry.
        if report.get('candidate_id'):
            write('weekly_winner_training.json',dict(schema=1,week_start=monday.date().isoformat(),last_training=report))
    report=dict(report,learning_cadence='weekly',learning_timezone='America/Chicago',
                next_week_start=(monday+pd.DateOffset(weeks=1)).isoformat())
    return winner,report


def render_panel(st,games):
    selected=[g for g in games if decision(g['model_home_probability'])['selected']]
    selected.sort(key=lambda g:max(g['model_home_probability'],1-g['model_home_probability']),reverse=True)
    st.markdown('**NCAAF strongest winner picks · 65%+ model confidence**')
    st.caption('Fixed selection rule under evaluation. NCAAF future accuracy is not yet established. Other predictions remain available below.')
    if not selected:
        st.info('No NCAA winner forecasts currently meet the 65% confidence rule.');return
    frame=pd.DataFrame(selected[:10])
    columns=[c for c in ('commence_time','away_team','home_team','predicted_team','model_confidence') if c in frame]
    st.dataframe(frame[columns],hide_index=True,use_container_width=True)

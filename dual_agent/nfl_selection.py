"""Fixed NFL winner selection and prospective evaluation; never tunes on outcomes."""
from datetime import datetime, timezone
import math
import pandas as pd

RULE = 'nfl_winner_confidence_65_v1'
THRESHOLD = .65


def number(value):
    if value is None or isinstance(value, bool): return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError): return None


def decision(probability):
    p = number(probability)
    if p is None or not 0 < p < 1: raise ValueError('Invalid NFL probability')
    confidence = max(p, 1-p)
    return dict(rule_id=RULE, threshold=THRESHOLD, confidence=confidence,
                selected=confidence >= THRESHOLD, validation_status='prospective_evaluation_pending')


def summarize(rows):
    """Include only forecasts tagged before kickoff, never relabel old history."""
    eligible=[]
    for row in rows:
        rule=row.get('winner_selection') or {}
        try:
            captured=pd.Timestamp(row['captured_at']); start=pd.Timestamp(row['start_time'])
            if captured.tzinfo is None or start.tzinfo is None or not captured < start: continue
            if row.get('source')=='historical_reconstruction' or row.get('historical_reconstruction'): continue
            if row.get('outcome') not in (0,1): continue
            expected=decision(row.get('connected_probability',row.get('original_probability')))
            if any(rule.get(k)!=expected[k] for k in ('rule_id','threshold','selected')): continue
            eligible.append(row)
        except (ValueError, TypeError, KeyError): continue
    # A repeated scan of a game cannot inflate its count.
    unique={}
    for row in sorted(eligible,key=lambda r:pd.Timestamp(r['captured_at'])):
        key=(row['base_id'],row['home_team'],row['away_team'],pd.Timestamp(row['start_time']).isoformat())
        unique.setdefault(key,row)
    eligible=list(unique.values())
    selected=[r for r in eligible if r['winner_selection']['selected']]
    def score(group):
        hits=sum((r.get('connected_probability',r['original_probability'])>=.5)==bool(r['outcome']) for r in group)
        return dict(games=len(group),correct=hits,accuracy=hits/len(group) if group else None)
    paired=[r for r in selected if number((r.get('context') or {}).get('market_home_probability')) not in (None,.5)]
    favorite_hits=sum((r['context']['market_home_probability']>.5)==bool(r['outcome']) for r in paired)
    observed=[r for r in rows if r.get('source')!='historical_reconstruction' and r.get('outcome') in (0,1)]
    keys=['home_qb_out','away_qb_out','home_qb_uncertain','away_qb_uncertain',
          'home_reported_out','away_reported_out','weather_temperature_f',
          'weather_wind_mph','weather_rain_chance','market_home_probability']
    coverage={k:sum(number(r.get('context',{}).get(k)) is not None for r in observed) for k in keys}
    return dict(rule_id=RULE,threshold=THRESHOLD,status='collecting_prospective_results',
                all_tagged=score(eligible),selected=score(selected),
                coverage=len(selected)/len(eligible) if eligible else None,
                matched_market=dict(model=score(paired),favorite_correct=favorite_hits,
                    favorite_accuracy=favorite_hits/len(paired) if paired else None),
                input_coverage=dict(graded_games=len(observed),observed_games=coverage),
                limitation='Threshold chosen from historical exploration; prospective results are separate. Accuracy is not betting return.')


def render_panel(st, predictions):
    frame=pd.DataFrame(predictions).copy()
    if frame.empty or 'home_win_probability' not in frame: return
    p=pd.to_numeric(frame.home_win_probability,errors='coerce')
    selected=frame.loc[p.between(THRESHOLD,1,inclusive='left') | ((1-p)>=THRESHOLD)&p.gt(0)].copy()
    st.markdown('**NFL strongest winner picks · 65%+ model confidence**')
    st.caption('Fixed rule under prospective evaluation. Historical accuracy is not a guaranteed win rate. All other game cards remain available below.')
    if selected.empty:
        st.info('No winner forecasts currently meet the 65% confidence rule.')
        return
    selected['model_confidence']=selected.home_win_probability.map(lambda p:max(float(p),1-float(p)))
    selected=selected.sort_values('model_confidence',ascending=False).head(10)
    columns=[c for c in ('commence_time','away_team','home_team','predicted_team','model_confidence') if c in selected]
    st.dataframe(selected[columns],hide_index=True,use_container_width=True)

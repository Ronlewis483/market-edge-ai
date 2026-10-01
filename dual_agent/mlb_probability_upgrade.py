"""Research-only, frozen-base probability comparison using saved matchup features.

No network calls, odds purchases, scanner, live promotion, or historical downloads.
Odds format: {"rows": [{"game_id": 123, "captured_at": "...Z",
"bookmaker": "book-name", "home_decimal": 1.5, "away_decimal": 2.8}]}.
Both prices must be from one bookmaker and the same captured snapshot.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from threadpoolctl import threadpool_limits

MODULE_VERSION = 1


def consensus_probabilities(games, dataset):
    """Latest eligible quote per book, then median normalized two-way probability."""
    rows = (dataset or {}).get('rows', [])
    if not isinstance(rows, list):
        raise ValueError('Odds JSON must contain a rows list')
    indexed = {}
    rejected = 0
    for row in rows:
        try:
            gid = int(row['game_id'])
            stamp = pd.Timestamp(row['captured_at'])
            book = str(row['bookmaker']).strip()
            home, away = float(row['home_decimal']), float(row['away_decimal'])
            if stamp.tzinfo is None or not book or not np.isfinite([home, away]).all() or min(home, away) <= 1:
                raise ValueError('Invalid quote')
            qh, qa = 1/home, 1/away
            indexed.setdefault(gid, []).append((stamp, book, qh/(qh+qa)))
        except (KeyError, TypeError, ValueError, OverflowError):
            rejected += 1
    values, coverage = [], []
    for game in games:
        cutoff = pd.to_datetime(game['timecode'], format='%Y%m%d_%H%M%S', utc=True)
        start = pd.Timestamp(game['start_time'])
        if start.tzinfo is None or cutoff >= start:
            raise ValueError('Game prediction must have an aware pregame capture time')
        books = {}
        conflicts = set()
        for stamp, book, p in indexed.get(int(game['game_id']), []):
            # Quotes must precede the historical decision and be <=30 minutes old.
            if stamp > cutoff or stamp >= start or cutoff-stamp > pd.Timedelta(minutes=30):
                continue
            previous = books.get(book)
            if previous is None or stamp > previous[0]:
                books[book] = (stamp, p)
                conflicts.discard(book)
            elif stamp == previous[0] and abs(p-previous[1]) > 1e-12:
                conflicts.add(book)
        usable = [p for book, (_, p) in books.items() if book not in conflicts]
        values.append(float(np.median(usable)) if len(usable) >= 2 else np.nan)
        coverage.append({'game_id':game['game_id'], 'eligible_books':len(usable),
                         'market_available':len(usable) >= 2})
    return np.asarray(values), {'coverage':coverage,'invalid_quote_rows':rejected,
        'minimum_books':2,'maximum_quote_age_minutes':30,
        'note':'Two-sided decimal moneyline quotes; normalize within each book, then take the median. Later quotes and conflicting same-time quotes are excluded.'}


def _logit(p):
    p = np.clip(np.asarray(p,dtype=float), 1e-6, 1-1e-6)
    return np.log(p/(1-p))


def _fit(X, y):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1,max_iter=2000,random_state=42))
    model.fit(X, y)
    return model


def _bundle(model, columns):
    scaler, layer = model.steps[0][1], model.steps[1][1]
    return {'columns':columns,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'weights':layer.coef_[0].tolist(),'intercept':float(layer.intercept_[0])}


def _score(label, y, p):
    from dual_agent.mlb_matchup_elo import confidence_diagnostics
    return {'Model':label,'Games':len(y),'Correct':int(((p>=.5)==y).sum()),
            'Accuracy':float(((p>=.5)==y).mean()),
            'Log loss':float(log_loss(y,p,labels=[0,1])),
            'Brier':float(brier_score_loss(y,p)),
            'confidence':confidence_diagnostics(p,y)}


def run_saved_probability_upgrade(archive, odds_dataset=None):
    """2024 selects runs; early 2025 fits layers; late 2025 selects; 2026 grades.

    The 2024 runs model is frozen throughout layer fitting, selection, and test.
    Nothing is refit after late-2025 selection, preventing base forecast drift.
    """
    from dual_agent.mlb_matchup_model import fit_run_model, predict_matchups, FEATURES, BULLPEN_FEATURES
    from dual_agent.mlb_matchup_elo import select_recommendations, recommendation_metrics
    from dual_agent.mlb_handedness import HAND_FEATURES
    from dual_agent.mlb_matchup_context import FORM_FEATURES, ENVIRONMENT_FEATURES
    all_rows = archive.get('retained_feature_rows', [])
    if not all_rows:
        raise ValueError('Load a full matchup JSON archive containing retained_feature_rows')
    if len({int(g['game_id']) for g in all_rows}) != len(all_rows):
        raise ValueError('Duplicate archived game IDs')
    all_rows = sorted(all_rows,key=lambda g:(g['start_time'],int(g['game_id'])))
    rows24 = [g for g in all_rows if int(g['season'])==2024]
    rows25 = [g for g in all_rows if int(g['season'])==2025]
    rows26 = [g for g in all_rows if int(g['season'])==2026]
    if min(map(len,[rows24,rows25,rows26])) < 100:
        raise ValueError('Need at least 100 saved feature games in each of 2024, 2025, 2026')
    def split_dates(rows, fraction):
        dates = sorted({pd.Timestamp(g['start_time']).date() for g in rows})
        boundary = dates[min(len(dates)-1,max(1,int(len(dates)*fraction)))]
        early = [g for g in rows if pd.Timestamp(g['start_time']).date()<boundary]
        late = [g for g in rows if pd.Timestamp(g['start_time']).date()>=boundary]
        cap = min(pd.to_datetime(g['timecode'],format='%Y%m%d_%H%M%S',utc=True) for g in late)-pd.Timedelta(hours=48)
        early = [g for g in early if pd.Timestamp(g['start_time'])<cap]
        if min(len(early),len(late)) < 40:
            raise ValueError('Chronological blocks need at least 40 eligible games each')
        return early,late
    early24,late24 = split_dates(rows24,.65)
    calibration,selection = split_dates(rows25,.5)
    handed = bool(archive.get('handedness_connected',False))
    context = bool(archive.get('context_connected',False))
    # Reuse feature schema, but select bullpen and regularization on 2024 only.
    bullpen_choices = [False,True] if all('bullpen_sides' in g for g in all_rows) else [False]
    development_runs=[]
    with threadpool_limits(limits=2):
        for bullpen in bullpen_choices:
            for alpha in [.1,1.,10.]:
                fitted=fit_run_model(early24,alpha,bullpen,handed,context)
                _,p=predict_matchups(fitted,late24)
                development_runs.append({'alpha':alpha,'bullpen':bullpen,'log_loss':float(log_loss([g['home_win'] for g in late24],p,labels=[0,1]))})
        chosen=min(development_runs,key=lambda g:g['log_loss'])
        first_calibration=min(pd.to_datetime(g['timecode'],format='%Y%m%d_%H%M%S',utc=True) for g in calibration)-pd.Timedelta(hours=48)
        eligible24=[g for g in rows24 if pd.Timestamp(g['start_time'])<first_calibration]
        base=fit_run_model(eligible24,chosen['alpha'],chosen['bullpen'],handed,context)
        blocks=[calibration,selection,rows26]
        rates=[];raw=[];ys=[];market=[];market_sources=[]
        for games in blocks:
            r,p=predict_matchups(base,games)
            rates.append(r);raw.append(p);ys.append(np.asarray([g['home_win'] for g in games],dtype=int))
            mp,source=consensus_probabilities(games,odds_dataset)
            market.append(mp);market_sources.append(source)
        candidates={'Frozen matchup runs':(raw[1],raw[2],None)}
        candidates_bundles={}
        feature_sets={
            'Calibrated matchup':(['matchup_log_odds'],[np.column_stack([_logit(p)]) for p in raw]),
            'Consistent matchup + Elo':(['matchup_log_odds','pregame_elo_difference','projected_run_difference'],[
                np.column_stack([_logit(p),[g['elo_difference'] for g in games],r[:,0]-r[:,1]]) for games,r,p in zip(blocks,rates,raw)])}
        for label,(columns,X) in feature_sets.items():
            layer=_fit(X[0],ys[0]);candidates[label]=(layer.predict_proba(X[1])[:,1],layer.predict_proba(X[2])[:,1],layer)
            candidates_bundles[label]=_bundle(layer,columns)
        choice_scores=[_score(label,ys[1],values[0]) for label,values in candidates.items()]
        chosen_label=min(choice_scores,key=lambda g:g['Log loss'])['Model']
        chosen_dev,chosen_test,_=candidates[chosen_label]
        cutoff,cutoff_rows=select_recommendations(chosen_dev,ys[1])
        _,recommended=recommendation_metrics(chosen_test,ys[2],cutoff)
        market_report={'status':'Not fitted: provide timestamped two-sided odds for calibration, selection, and evaluation games.', 'sources':market_sources}
        masks=[np.isfinite(p) for p in market]
        counts=[int(m.sum()) for m in masks]
        if counts[0]>=80 and counts[1]>=40 and counts[2]>=1 and len(np.unique(ys[0][masks[0]]))==2:
            MX=[np.column_stack([_logit(r[m]),_logit(p[m])]) for r,p,m in zip(raw,market,masks)]
            layer=_fit(MX[0],ys[0][masks[0]])
            mdev=layer.predict_proba(MX[1])[:,1];mtest=layer.predict_proba(MX[2])[:,1]
            alternatives=[('Market consensus',market[1][masks[1]],market[2][masks[2]]),('Matchup on odds-covered games',raw[1][masks[1]],raw[2][masks[2]]),('Matchup + market',mdev,mtest)]
            matched_dev=[_score(label,ys[1][masks[1]],p) for label,p,_ in alternatives]
            selected_market=min(matched_dev,key=lambda g:g['Log loss'])['Model']
            md,mt=next((d,t) for label,d,t in alternatives if label==selected_market)
            mc,mcrows=select_recommendations(md,ys[1][masks[1]])
            _,mm=recommendation_metrics(mt,ys[2][masks[2]],mc)
            market_report.update(status='Fitted on odds-covered games only',covered_games=counts,selected_on_late_2025=selected_market,
                development=matched_dev,evaluation=[_score(label,ys[2][masks[2]],p) for label,_,p in alternatives],
                bundle=_bundle(layer,['matchup_log_odds','market_log_odds']),threshold=mc,recommendations=mm,
                threshold_development=mcrows,note='Uncovered games excluded, never filled with market averages. These comparisons do not replace the full-cohort model.')
    names=FEATURES+(BULLPEN_FEATURES if chosen['bullpen'] else [])+(HAND_FEATURES if handed else [])+(FORM_FEATURES+ENVIRONMENT_FEATURES if context else [])
    imputer,scaler,reg= [step[1] for step in base.steps]
    retained=[name for name,value in zip(names,imputer.statistics_) if np.isfinite(value)]
    return {'version':1,'created_at':pd.Timestamp.now(tz='UTC').isoformat(),'source_archive':archive.get('created_at'),
        'protocol':'2024 chronological run selection; frozen 2024 base; early 2025 layer fitting; late 2025 layer/cutoff selection; 2026 diagnostic evaluation. No post-selection refitting.',
        'counts':dict(zip(['run_training','calibration','selection','evaluation'],[len(eligible24),len(calibration),len(selection),len(rows26)])),
        'run_selection':development_runs,'selected_run_configuration':chosen,'development':choice_scores,
        'selected_layer':chosen_label,'evaluation':[_score(label,ys[2],values[1]) for label,values in candidates.items()],
        'threshold':cutoff,'threshold_development':cutoff_rows,'recommendations':recommended,'market':market_report,
        'model_bundle':{'columns':retained,'median':imputer.statistics_[np.isfinite(imputer.statistics_)].tolist(),
            'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'weights':reg.coef_.tolist(),'intercept':float(reg.intercept_)},
        'layer_bundles':candidates_bundles,'predictions':[{'game_id':g['game_id'],'home_probability':float(p),'Correct':bool((p>=.5)==g['home_win'])} for g,p in zip(rows26,chosen_test)],
        'limitations':['2026 outcomes have already been inspected; this is diagnostic, not a fresh holdout.',
            'Existing feature definitions were developed during earlier experiments; chronological splits cannot undo that research history.',
            'Frozen 2024 base fixes layer input consistency but does not learn from 2025; any benefit must be measured.',
            '70% is a goal, not guaranteed. Rules need a new prospective record.'], 'live_model_changed':False}

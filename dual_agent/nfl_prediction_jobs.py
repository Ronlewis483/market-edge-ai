"""NFL background orchestration; model equations remain in their existing modules.

Initial team forecasts, connected winners and props publish separate snapshots.
Only the collector owns its queue; no UI polling or automatic reruns live here.
"""
import time
import pandas as pd


def prioritize_today(predictions, now=None):
    """Today's local games cannot be buried below a stronger Sunday slate."""
    frame = predictions.copy()
    current = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    current = current.tz_localize('UTC') if current.tzinfo is None else current.tz_convert('UTC')
    dates = pd.to_datetime(frame['commence_time'],utc=True,errors='coerce').dt.tz_convert('America/Chicago').dt.date
    frame['_nfl_today'] = dates.eq(current.tz_convert('America/Chicago').date())
    return frame.sort_values(['_nfl_today','confidence'],ascending=[False,False],kind='stable')


def enrich_result(result, feature_games, odds_key, progress_queue=None, context=None):
    """Run external enrichment without reading or writing session state."""
    import copy
    result = copy.deepcopy(result)
    if context is not None:
        import threading
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        add_script_run_ctx(threading.current_thread(),context)
    from dual_agent.sports_prop_workspace import merge_result,checkpoint,track
    previous = copy.deepcopy(result)
    def publish_props(props):
        partial = merge_result('NFL',previous,dict(result,nfl_player_props=props))
        partial.update(workspace_pending=True,enrichment_pending=True)
        if progress_queue is not None: progress_queue.put(copy.deepcopy(partial))
        try:
            checkpoint('NFL',partial); track('NFL',partial)
        except Exception:
            partial['workspace_issue'] = 'NFL predictions are ready; local saving is unavailable.'
    from time import perf_counter
    context_clock = perf_counter()
    try:
        from dual_agent.nfl_live_context import attach_context
        result['predictions'] = attach_context(result.get('predictions'), result.get('live_odds'))
    except Exception as exc:
        result['context_issue'] = 'NFL context retrieval unavailable (' + type(exc).__name__ + ')'
    result.setdefault('nfl_stage_timings_seconds', {})['context'] = round(perf_counter()-context_clock, 3)
    correction_clock = perf_counter()
    try:
        from dual_agent.nfl_connected_engine import finalize_connected_result
        result = finalize_connected_result(result)
    except Exception as exc:
        result['connected_learning_issue'] = 'NFL connected finalization unavailable (' + type(exc).__name__ + ')'
    result.setdefault('nfl_stage_timings_seconds', {})['connected_finalization'] = round(perf_counter()-correction_clock, 3)
    result['nfl_generation_stage'] = ('Initial team forecasts retained; connected update unavailable; player props are loading'
        if result.get('connected_context_pending') else 'Winner predictions ready; player props are loading')
    result = merge_result('NFL',previous,result)
    if progress_queue is not None:
        progress_queue.put(copy.deepcopy(result))
    # Persist winners before potentially long prop downloads. A restart should
    # not restore yesterday's slate merely because today's props are pending.
    try:
        checkpoint('NFL',result)
    except Exception:
        result['workspace_issue'] = 'NFL predictions are ready; local saving is unavailable.'
    props_clock = perf_counter()
    try:
        from dual_agent.game_prediction_addons import enhance_result
        result = enhance_result("NFL", result, odds_key, feature_games,on_props=publish_props)
    except Exception as exc:
        result['prop_issues'] = ['NFL add-ons unavailable ('+type(exc).__name__+'); winner predictions are preserved.']
    result.setdefault('nfl_stage_timings_seconds', {})['player_props_and_spreads'] = round(perf_counter()-props_clock, 3)
    result['nfl_generation_stage'] = ('NFL scan complete; initial forecasts retained because the connected update is unavailable'
        if result.get('connected_context_pending') else 'NFL scan complete')
    result = merge_result('NFL',previous,result)
    result.update(enrichment_pending=False,workspace_pending=False,workspace_saved_at=pd.Timestamp.now(tz='UTC').isoformat())
    if progress_queue is not None: progress_queue.put(copy.deepcopy(result))
    try:
        track('NFL',result); checkpoint('NFL',result)
    except Exception:
        result['workspace_issue'] = 'NFL predictions are ready; local saving is unavailable.'
    try:
        from dual_agent.nfl_player_props_ui import cached_player_history
        from dual_agent.sports_prop_workspace import settle_nfl
        year = pd.Timestamp.now(tz='UTC').year
        settle_nfl(cached_player_history((year-1,year)))
    except Exception: pass
    try: checkpoint('NFL',result,remote=True)
    except Exception: pass
    return result



def generate(previous, historical_accuracy, historical_sample, odds_key,
             updates, progress, revision, context=None, *, load_history, load_odds, run_pipeline, enrich):
    """Run cold history, odds, connected winners and props off the UI thread."""
    import copy
    from time import perf_counter
    from concurrent.futures import ThreadPoolExecutor
    def bind_context():
        if context is not None:
            import threading
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            add_script_run_ctx(threading.current_thread(), context)
    bind_context()
    def report(message):
        progress.put(('stage', message))
    def timed_load(name, function):
        bind_context()
        started = perf_counter()
        value = function()
        return value, round(perf_counter()-started, 3)
    report('Loading current NFL results and sportsbook lines')
    # Independent requests overlap; neither source is fetched a second time.
    with ThreadPoolExecutor(max_workers=2) as pool:
        history_job = pool.submit(timed_load, 'history', load_history)
        odds_job = pool.submit(timed_load, 'odds', load_odds)
        bundle, history_seconds = history_job.result()
        progress.put(('history', bundle))
        report('NFL history ready; checking sportsbook lines')
        try:
            live_odds, odds_seconds = odds_job.result()
        except Exception:
            live_odds, odds_seconds = pd.DataFrame(), 0
            report("Sportsbook lines unavailable; checking the NFL schedule for winner forecasts")
    report('Calculating NFL matchups')
    result = run_pipeline(feature_games=bundle[1],
        historical_accuracy=historical_accuracy, historical_sample=historical_sample,
        live_odds=live_odds, on_progress=report)
    result.setdefault('nfl_stage_timings_seconds', {}).update(
        history=history_seconds, odds=odds_seconds)
    from dual_agent.sports_prop_workspace import merge_result
    result = merge_result('NFL', copy.deepcopy(previous or {}), result)
    result.update(nfl_winner_revision=revision, workspace_pending=True, enrichment_pending=True)
    result['nfl_generation_stage'] = 'Initial team forecasts ready; QB, injury, weather and learned correction are pending'
    updates.put(copy.deepcopy(result))
    # These are explicitly initial forecasts, not the final connected output.
    # Context and approved correction still run before the winner-ready stage.
    from dual_agent.sports_prop_workspace import checkpoint
    try:
        checkpoint('NFL',result)
    except Exception:
        result['workspace_issue'] = 'Initial NFL forecasts are ready; local saving is unavailable.'
    report('Checking quarterbacks, injuries and weather for connected predictions')
    return enrich(result, bundle[1], odds_key, updates, context)


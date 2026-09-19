import pandas as pd
from dual_agent.nba_data import test_nba_connections
from dual_agent.nba_history import test_nba_history
from dual_agent.balldontlie_data import (
    test_balldontlie_connection,
    test_full_historical_season,
    test_multiple_historical_seasons,
    audit_historical_games,
)
from dual_agent.nba_research import (
    run_nba_research,
    run_multi_season_nba_research,
    nba_calibration_summary,
    prepare_balldontlie_games_for_research,
    build_balldontlie_pregame_features,
    audit_balldontlie_pregame_features,
    run_balldontlie_walkforward_model,
    build_balldontlie_future_matchup_features,
    predict_balldontlie_matchup,
    calculate_no_vig_model_edge,
    get_live_nba_moneylines,
    normalize_nba_team_name,
)

from dual_agent.nfl_data import (
    test_sportradar_connection,
    get_nfl_seasons,
    get_nfl_season_games,
    get_multiple_nfl_seasons,
    audit_nfl_games,
)

from dual_agent.nfl_research import (
    build_nfl_pregame_features,
    run_nfl_walkforward_model,
    build_nfl_future_matchup_features,
    predict_nfl_matchup,
)

from dual_agent.nfl_decision import (
    build_nfl_confidence_profile,
    get_nfl_decision,
)

from dual_agent.nfl_market import (
    calculate_nfl_market_edge,
    classify_nfl_market_edge,
)

from dual_agent.nfl_odds import (
    get_live_nfl_moneylines,
    get_best_nfl_moneylines,
)

from dual_agent.nfl_live_engine import (
    build_live_nfl_opportunities,
)

import streamlit as st

st.set_page_config(page_title="Market Edge AI V5", page_icon="📊", layout="wide")
from dual_agent.research import DEFAULT_UNIVERSE, latest_scan, run_research
from dual_agent.signal_engine import clean_symbols, stock_decision, sports_decision, GATES
from dual_agent.validated_model import VALIDATED_STOCK_MODEL as M

st.title("📊 Market Edge AI — V5")
st.caption("Persistent validated model • lightweight daily inference • research/paper mode")

with st.sidebar:
    page=st.radio("Navigation",["Command Center","Saved Model","Research Lab"])
    st.success("Persistence enabled")
    st.caption("Validated configuration loads automatically. Daily use does not require walk-forward retraining.")

default=clean_symbols(DEFAULT_UNIVERSE)

if page=="Command Center":
    st.success(f"VALIDATED MODEL LOADED — {M['target']} • {M['features']} • AUC {M['auc']:.3f}")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("📈 Best Stock Signal")
        txt=st.text_input("Symbols to scan", "AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AVGO,AMD,JPM,LLY,XOM")
        syms=clean_symbols(txt)
        if st.button("Scan Today's Market",type="primary",use_container_width=True):
            try:
                with st.spinner("Loading recent market data and scoring stocks — no retraining..."):
                    # Keep existing V4 model/data implementation, but do NOT run walk-forward research.
                    df=latest_scan(default,syms)
                    st.session_state["daily"]=stock_decision(df)
            except Exception as e:
                st.error(f"Daily scan failed: {e}")

        d=st.session_state.get("daily")
        if d:
            if d["status"]=="MAKE THIS TRADE":
                st.success("MAKE THIS TRADE")
                st.markdown(f"## {d['symbol']}")
                a,b=st.columns(2)
                a.metric("Model probability",f"{d['probability']:.1%}")
                b.metric("Edge vs base rate",f"+{d['edge']:.1%}")
                st.write(f"**Direction:** {d['direction']}")
                st.write(f"**Horizon:** {d['horizon']}")
                st.write(f"**Target:** {d['target']}")
            else:
                st.warning("NO QUALIFYING TRADE")
                st.caption(d["reason"])
                top=d.get("top",{})
                if top:
                    st.write(f"Top candidate: **{top.get('Symbol','—')}** • probability {float(top.get('P',0)):.1%} • required {GATES['min_probability']:.0%}")

            ranked=d.get("ranked")
            if ranked is not None and len(ranked):
                show=ranked.head(10).copy()
                cols=[x for x in ["Symbol","Close","P","edge"] if x in show.columns]
                show=show[cols].rename(columns={"P":"Model probability","edge":"Edge vs base"})
                st.markdown("#### Today's top candidates")
                st.dataframe(show,use_container_width=True,hide_index=True)

    with c2:
        st.subheader("🏀 Best Sports Signal")
        sd=sports_decision()
        st.warning(sd["status"])
        st.caption(sd["reason"])
        st.code("PICK THIS TEAM / PICK THIS PLAYER PROP\nor\nNO QUALIFYING SPORTS PICK")

elif page=="Saved Model":
    st.subheader("💾 Saved Validated Model")
    st.success("This configuration is bundled with the app and survives Streamlit restarts.")
    st.json(M)
    st.write("Daily Command Center scans load this approved configuration automatically; they do not rerun the 90-symbol walk-forward experiment.")

else:
    st.subheader("🧪 Research Lab — Optional Revalidation")

    st.markdown("### 🏀 NBA Data Connection Test")

    if st.button("Test NBA Data Sources"):
        with st.spinner("Testing SportsDataIO and The Odds API..."):
            nba_status = test_nba_connections()

        for provider, info in nba_status.items():
            if info["working"]:
                st.success(f"✅ {provider}: {info['message']}")
            elif info["configured"]:
                st.error(f"❌ {provider}: {info['message']}")
            else:
                st.warning(f"⚠️ {provider}: {info['message']}")

    st.divider()


    


    st.divider()

# ============================================================
# MULTI-SEASON NBA VALIDATION
# ============================================================

    st.markdown("### 🏀 BALLDONTLIE Historical Data Test")

    if st.button("Test BALLDONTLIE Historical Data"):
        try:
            with st.spinner("Downloading 2023 NBA games from BALLDONTLIE..."):
                bdl_test = test_balldontlie_connection()

            st.success(
                f"BALLDONTLIE connection successful — "
                f"{bdl_test['games_returned']} historical games returned."
            )

            st.write(
                "Season:",
                bdl_test["season"],
            )

            st.write(
                "Date range:",
                bdl_test["first_game"],
                "to",
                bdl_test["last_game"],
            )

            st.dataframe(
                bdl_test["sample"],
                use_container_width=True,
                hide_index=True,
            )

        except Exception as e:
            st.error(
                f"BALLDONTLIE historical data error: {e}"
            )
    if st.button("Download Full 2023 NBA Season"):
        try:
            with st.spinner(
                "Downloading the full 2023 NBA season. "
                "This may take a couple of minutes..."
            ):
                full_season_test = test_full_historical_season(2023)

            st.success(
                f"Full season downloaded — "
                f"{full_season_test['games_returned']} games."
            )

            st.write(
                "API requests used:",
                full_season_test["requests_used"],
            )

            st.write(
                "Date range:",
                full_season_test["first_game"],
                "to",
                full_season_test["last_game"],
            )

            st.write(
                "Home win rate:",
                f"{full_season_test['home_win_rate']:.1%}",
            )

            st.dataframe(
                full_season_test["sample"],
                use_container_width=True,
                hide_index=True,
            )

        except Exception as e:
            st.error(
                f"Full-season download error: {e}"
            )



    if st.button("Test 2022 + 2023 NBA Seasons"):
        try:
            with st.spinner(
                "Downloading 2022 and 2023 NBA seasons. "
                "This will take several minutes..."
            ):
                multi_bdl_test = test_multiple_historical_seasons(
                    [2022, 2023, 2024, 2025]
                )

            st.success(
                f"Multi-season download successful - "
                f"{multi_bdl_test['games_returned']} total games."
            )

            st.write(
                "API requests used:",
                multi_bdl_test["total_requests"],
            )

            st.write(
                "Overall date range:",
                multi_bdl_test["first_game"],
                "to",
                multi_bdl_test["last_game"],
            )

            st.write(
                "Overall home win rate:",
                f"{multi_bdl_test['home_win_rate']:.1%}",
            )

            st.markdown("#### Season Summary")

            st.dataframe(
                multi_bdl_test["season_summary"],
                use_container_width=True,
                hide_index=True,
            )

            # ==================================================
            # RESEARCH PIPELINE BRIDGE
            # ==================================================

            st.markdown("#### 🧠 Research Pipeline Bridge Test")

            research_games = prepare_balldontlie_games_for_research(
                multi_bdl_test["games"]
            )

            st.success(
                f"Research bridge successful - "
                f"{len(research_games)} games ready for modeling."
            )

            bridge_col1, bridge_col2, bridge_col3 = st.columns(3)

            bridge_col1.metric(
                "Model-Ready Games",
                len(research_games),
            )

            bridge_col2.metric(
                "First Game",
                research_games["game_date"]
                .min()
                .strftime("%Y-%m-%d"),
            )

            bridge_col3.metric(
                "Last Game",
                research_games["game_date"]
                .max()
                .strftime("%Y-%m-%d"),
            )

            st.write(
                "Verified home-win rate:",
                f"{research_games['home_win'].mean():.1%}",
            )

            # ==================================================
            # PRE-GAME FEATURE ENGINE
            # ==================================================

            st.markdown("#### 🧮 Pre-Game Feature Engine Test")

            feature_games = build_balldontlie_pregame_features(
                multi_bdl_test["games"],
                min_games=5,
            )

            st.success(
                f"Feature engine successful - "
                f"{len(feature_games)} model-ready rows created."
            )

            feature_col1, feature_col2, feature_col3 = st.columns(3)

            feature_col1.metric(
                "Feature Rows",
                len(feature_games),
            )

            feature_col2.metric(
                "First Feature Date",
                feature_games["game_date"]
                .min()
                .strftime("%Y-%m-%d"),
            )

            feature_col3.metric(
                "Last Feature Date",
                feature_games["game_date"]
                .max()
                .strftime("%Y-%m-%d"),
            )

            st.write(
                "Feature columns:",
                len(feature_games.columns),
            )

            st.dataframe(
                feature_games.head(10),
                use_container_width=True,
                hide_index=True,
            )

            # ==================================================
            # LEAKAGE AUDIT
            # ==================================================

            st.markdown("#### 🔒 Pre-Game Leakage Audit")

            leakage_audit = audit_balldontlie_pregame_features(
                feature_games
            )

            leak_col1, leak_col2, leak_col3 = st.columns(3)

            leak_col1.metric(
                "Rows Audited",
                leakage_audit["total_rows"],
            )

            leak_col2.metric(
                "Model Features",
                leakage_audit["model_feature_count"],
            )

            leak_col3.metric(
                "Suspicious Features",
                len(
                    leakage_audit[
                        "suspicious_model_features"
                    ]
                ),
            )

            st.write(
                "Model feature columns:",
                leakage_audit["model_features"],
            )

            if leakage_audit["suspicious_model_features"]:
                st.error(
                    "Possible leakage detected: "
                    + ", ".join(
                        leakage_audit[
                            "suspicious_model_features"
                        ]
                    )
                )
            else:
                st.success(
                    "No obvious current-game outcome leakage "
                    "detected in the proposed model features."
                )

            if len(
                leakage_audit["missing_feature_values"]
            ) > 0:
                st.warning(
                    "Missing values found in model features."
                )

                st.dataframe(
                    leakage_audit[
                        "missing_feature_values"
                    ]
                    .rename("missing_values")
                    .reset_index()
                    .rename(
                        columns={
                            "index": "feature"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success(
                    "No missing values found in model features."
                )

            # ==================================================
            # WALK-FORWARD MODEL
            # ==================================================

            st.markdown("#### 🧠 Walk-Forward Model Test")

            walkforward_results = (
                run_balldontlie_walkforward_model(
                    feature_games
                )
            )

            st.success(
                f"Walk-forward validation successful - "
                f"{walkforward_results['games_predicted']} "
                f"unseen games predicted."
            )
            st.markdown("#### 📊 Walk-Forward Model Performance")
    
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
            perf_col1.metric(
                "Accuracy",
                f"{walkforward_results['accuracy']:.1%}",
            )
    
            perf_col2.metric(
                "AUC",
                f"{walkforward_results['auc']:.3f}",
            )
    
            perf_col3.metric(
                "Brier Score",
                f"{walkforward_results['brier']:.4f}",
            )
    
            perf_col4.metric(
                "Log Loss",
                f"{walkforward_results['log_loss']:.4f}",
            )
    
            st.markdown("##### Baseline Comparison")
    
            base_col1, base_col2, base_col3 = st.columns(3)
    
            base_col1.metric(
                "Baseline Accuracy",
                f"{walkforward_results['baseline_accuracy']:.1%}",
            )
    
            base_col2.metric(
                "Baseline Brier",
                f"{walkforward_results['baseline_brier']:.4f}",
            )
    
            base_col3.metric(
                "Baseline Log Loss",
                f"{walkforward_results['baseline_log_loss']:.4f}",
            )
            st.markdown("##### 🎚️ Walk-Forward Confidence Calibration")
    
            calibration_predictions = (
                walkforward_results["predictions"].copy()
            )
    
            calibration_predictions["confidence"] = (
                calibration_predictions[
                    "probability"
                ]
                .where(
                    calibration_predictions["prediction"] == 1,
                    1.0
                    - calibration_predictions[
                        "probability"
                    ],
                )
            )
    
            calibration_predictions["correct"] = (
                calibration_predictions["prediction"]
                == calibration_predictions["actual"]
            ).astype(int)
    
            calibration_predictions["confidence_band"] = pd.cut(
                calibration_predictions["confidence"],
                bins=[
                    0.50,
                    0.55,
                    0.60,
                    0.65,
                    0.70,
                    0.75,
                    0.80,
                    0.85,
                    0.90,
                    0.95,
                    1.01,
                ],
                labels=[
                    "50–55%",
                    "55–60%",
                    "60–65%",
                    "65–70%",
                    "70–75%",
                    "75–80%",
                    "80–85%",
                    "85–90%",
                    "90–95%",
                    "95–100%",
                ],
                include_lowest=True,
            )
    
            calibration_table = (
                calibration_predictions
                .groupby(
                    "confidence_band",
                    observed=False,
                )
                .agg(
                    games=("correct", "size"),
                    actual_accuracy=("correct", "mean"),
                    average_confidence=("confidence", "mean"),
                )
                .reset_index()
            )
    
            calibration_table["actual_accuracy"] = (
                calibration_table["actual_accuracy"]
                .map(lambda value: f"{value:.1%}")
            )
    
            calibration_table["average_confidence"] = (
                calibration_table["average_confidence"]
                .map(lambda value: f"{value:.1%}")
            )
    
            st.dataframe(
                calibration_table,
                use_container_width=True,
                hide_index=True,
            )

            
            st.markdown("#### 🔮 Future Matchup Feature Test")
    
            historical_test_game = feature_games.iloc[-1]
    
            future_test_features = (
                build_balldontlie_future_matchup_features(
                    multi_bdl_test["games"],
                    historical_test_game["home_team"],
                    historical_test_game["away_team"],
                    historical_test_game["game_date"],
                )
            )
    
            st.success(
                "Future matchup feature builder successful."
            )
    
            future_col1, future_col2, future_col3 = st.columns(3)
    
            future_col1.metric(
                "Home Team",
                historical_test_game["home_team"],
            )
    
            future_col2.metric(
                "Away Team",
                historical_test_game["away_team"],
            )
    
            future_col3.metric(
                "Feature Count",
                len(future_test_features.columns),
            )
    
            st.write(
                "Prediction date:",
                historical_test_game["game_date"].strftime("%Y-%m-%d"),
            )
    
            st.dataframe(
                future_test_features,
                use_container_width=True,
                hide_index=True,
            )

            st.write(
                "DEBUG future columns:",
                list(future_test_features.columns),
            )

            st.markdown("#### 🎯 NBA Matchup Prediction")
    
            live_prediction = predict_balldontlie_matchup(
                feature_games,
                future_test_features,
            )
    
            pred_col1, pred_col2, pred_col3 = st.columns(3)
    
            pred_col1.metric(
                "Home Win Probability",
                f"{live_prediction['home_win_probability']:.1%}",
            )
    
            pred_col2.metric(
                "Away Win Probability",
                f"{live_prediction['away_win_probability']:.1%}",
            )
    
            pred_col3.metric(
                "Model Confidence",
                f"{live_prediction['confidence']:.1%}",
            )
    
            st.write(
                "Predicted side:",
                live_prediction["predicted_side"],
            )
    
            st.caption(
                f"Model trained on "
                f"{live_prediction['training_games']} historical games "
                f"using {live_prediction['feature_count']} features."
            )
            st.markdown("##### 🧪 No-Vig Edge Test")

            test_edge = calculate_no_vig_model_edge(
                home_model_probability=0.71,
                home_american_odds=-150,
                away_american_odds=130,
            )
            
            edge_col1, edge_col2, edge_col3 = st.columns(3)
            
            with edge_col1:
                st.metric(
                    "Model Home Probability",
                    f"{test_edge['home_model_probability']:.1%}",
                )
            
            with edge_col2:
                st.metric(
                    "No-Vig Market Probability",
                    f"{test_edge['home_market_probability']:.1%}",
                )
            
            with edge_col3:
                st.metric(
                    "Model Edge",
                    f"{test_edge['home_edge']:+.1%}",
                )
            
            st.write(
                f"Best model side: **{test_edge['best_side']}**"
            )
            
            st.write(
                f"Sportsbook hold: "
                f"**{test_edge['sportsbook_hold']:.2%}**"
            )
            
            # ==================================================
            # HISTORICAL DATASET AUDIT
            # ==================================================

            st.markdown("#### 🔎 Historical Dataset Audit")

            audit = audit_historical_games(
                multi_bdl_test["games"]
            )

            audit_col1, audit_col2, audit_col3 = st.columns(3)

            audit_col1.metric(
                "Total Rows",
                audit["total_rows"],
            )

            audit_col2.metric(
                "Unique Game IDs",
                audit["unique_game_ids"],
            )

            audit_col3.metric(
                "Duplicate Game IDs",
                audit["duplicate_game_ids"],
            )

            audit_col4, audit_col5, audit_col6 = st.columns(3)

            audit_col4.metric(
                "Missing Scores",
                audit["missing_scores"],
            )

            audit_col5.metric(
                "Tied Games",
                audit["tied_games"],
            )

            audit_col6.metric(
                "Teams Found",
                audit["teams_found"],
            )

            st.write(
                "Unusual teams:",
                audit["unusual_teams"]
                if audit["unusual_teams"]
                else "None",
            )

            st.markdown("##### Games by Month")

            st.dataframe(
                audit["games_by_month"],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("##### Game Status")

            st.dataframe(
                audit["status_summary"],
                use_container_width=True,
                hide_index=True,
            )

            if len(audit["suspicious_scores"]):
                st.warning(
                    f"{len(audit['suspicious_scores'])} "
                    f"games have suspicious scores."
                )
            else:
                st.success(
                    "No suspicious zero or missing scores found."
                )

        except Exception as e:
            st.error(
                f"Multi-season BALLDONTLIE error: {e}"
            )
    
    st.divider()
    


    
    st.markdown("### 🧪 NBA.com Historical Data Test")

    if st.button("Test NBA.com Historical Data"):
        try:
            with st.spinner("Downloading 2024-25 NBA games from NBA.com..."):
                history_test = test_nba_history("2024-25")

            st.success(
                f"NBA.com connection successful — "
                f"{history_test['games']} games loaded."
            )

            st.write(
                "Date range:",
                history_test["first_game"],
                "to",
                history_test["last_game"],
            )

            st.dataframe(
                history_test["sample"],
                use_container_width=True,
                hide_index=True,
            )

        except Exception as e:
            st.error(f"NBA.com historical data error: {e}")

    st.divider()



st.markdown("### 🏀 Multi-Season NBA Validation")

nba_seasons_text = st.text_input(
    "NBA seasons to validate",
    value="2022,2023,2024,2025",
    help="Enter SportsDataIO NBA seasons separated by commas.",
)

if st.button("Run Multi-Season NBA Validation"):

    seasons = [
        int(x.strip())
        for x in nba_seasons_text.split(",")
        if x.strip()
    ]

    try:
        with st.spinner(
            "Downloading multiple NBA seasons and running walk-forward validation..."
        ):
            multi_bdl_result = test_multiple_historical_seasons(seasons)

            historical_games = multi_bdl_result["games"]

            prepared_games = prepare_balldontlie_games_for_research(
                historical_games
            )

            feature_games = build_balldontlie_pregame_features(
                prepared_games
            )

            multi_nba_result = run_balldontlie_walkforward_model(
                feature_games
            )

            st.session_state["multi_nba_research_result"] = multi_nba_result

        st.success("Multi-season NBA validation complete.")

    except Exception as e:
        st.error(f"Multi-season NBA validation error: {e}")
if "multi_nba_research_result" in st.session_state:

    mr = st.session_state["multi_nba_research_result"]

    st.markdown("#### 🏀 Multi-Season Dataset")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Games predicted",
        mr["games_predicted"],
    )

    m2.metric(
        "Accuracy",
        f"{mr['accuracy']:.1%}",
    )

    m3.metric(
        "AUC",
        f"{mr['auc']:.3f}",
    )

    m4.metric(
        "Features",
        mr["feature_count"],
    )

    st.markdown("#### 📊 Walk-Forward Performance")

    p1, p2, p3, p4 = st.columns(4)

    p1.metric(
        "Accuracy",
        f"{mr['accuracy']:.1%}",
    )

    p2.metric(
        "AUC",
        f"{mr['auc']:.3f}",
    )

    p3.metric(
        "Brier Score",
        f"{mr['brier']:.4f}",
    )

    p4.metric(
        "Log Loss",
        f"{mr['log_loss']:.4f}",
    )

    st.markdown("#### ⚖️ Model vs Baseline")

    comparison_df = pd.DataFrame(
        [
            {
                "Model": "NBA Walk-Forward Model",
                "Accuracy": mr["accuracy"],
                "Brier Score": mr["brier"],
                "Log Loss": mr["log_loss"],
            },
            {
                "Model": "Home Win Baseline",
                "Accuracy": mr["baseline_accuracy"],
                "Brier Score": mr["baseline_brier"],
                "Log Loss": mr["baseline_log_loss"],
            },
        ]
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### 🎯 Probability Calibration")

    calibration_result = nba_calibration_summary(
        mr["predictions"]
    )

    calibration_table = calibration_result["calibration"]

    if len(calibration_table):

        calibration_display = calibration_table.copy()

        for column in [
            "avg_predicted_probability",
            "actual_win_rate",
            "calibration_gap",
            "absolute_calibration_error",
        ]:
            if column in calibration_display.columns:
                calibration_display[column] = (
                    calibration_display[column]
                    .map(lambda x: f"{x:.1%}")
                )

        st.dataframe(
            calibration_display,
            use_container_width=True,
            hide_index=True,
        )

        weighted_error = calibration_result.get(
            "weighted_calibration_error"
        )

        if weighted_error is not None:
            st.metric(
                "Weighted Calibration Error",
                f"{weighted_error:.1%}",
            )

    else:
        st.info(
            "No calibration results were generated."
        )

    st.markdown("#### 🔬 High-Confidence Threshold Analysis")

    high_confidence = calibration_result.get(
        "high_confidence"
    )

    if (
        high_confidence is not None
        and len(high_confidence)
    ):
        threshold_display = high_confidence.copy()

        for column in [
            "minimum_probability",
            "accuracy",
            "avg_model_probability",
            "calibration_gap",
        ]:
            if column in threshold_display.columns:
                threshold_display[column] = (
                    threshold_display[column]
                    .map(lambda x: f"{x:.1%}")
                )

        st.dataframe(
            threshold_display,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No high-confidence threshold results were generated."
        )

    st.divider()

    st.markdown("### 🏀 NBA Historical Validation")

    nba_season = st.text_input(
        "NBA season",
        value="2025",
        help="SportsDataIO season to use for historical NBA research.",
        key="single_nba_season",
    )

    if st.button("Run NBA Historical Validation", key="run_single_nba_validation"):
        try:
            with st.spinner(
                "Building historical NBA features and running walk-forward validation..."
            ):
                nba_result = run_nba_research(
                    nba_season,
                    minimum_training_games=250,
                    test_block_size=100,
                )

            st.session_state["nba_research_result"] = nba_result
            st.success("NBA historical validation complete.")

        except Exception as e:
            st.error(f"NBA validation error: {e}")

    if "nba_research_result" in st.session_state:
        r = st.session_state["nba_research_result"]

        st.markdown("#### Dataset")

        n1, n2, n3, n4 = st.columns(4)

        n1.metric("Raw games", r["raw_games"])
        n2.metric("Completed games", r["completed_games"])
        n3.metric("Feature rows", r["feature_rows"])
        n4.metric("Home win rate", f"{r['home_win_rate']:.1%}")

        st.markdown("#### Walk-Forward Performance")

        model = r["overall"]
        baseline = r["baseline"]

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("AUC", f"{model['auc']:.3f}")
        m2.metric("Accuracy", f"{model['accuracy']:.1%}")
        m3.metric("Brier Score", f"{model['brier']:.4f}")
        m4.metric("Log Loss", f"{model['logloss']:.4f}")

        st.markdown("#### Model vs Baseline")

        comparison = pd.DataFrame(
            [
                {
                    "Model": "NBA Model",
                    "Accuracy": model["accuracy"],
                    "Brier": model["brier"],
                    "Log Loss": model["logloss"],
                    "AUC": model["auc"],
                },
                {
                    "Model": "Home Win Base Rate",
                    "Accuracy": baseline["accuracy"],
                    "Brier": baseline["brier"],
                    "Log Loss": baseline["logloss"],
                    "AUC": baseline["auc"],
                },
            ]
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Confidence Analysis")

        confidence = r["confidence"]

        if len(confidence):
            st.dataframe(
                confidence,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No confidence-band results were generated.")

        with st.expander("Walk-Forward Folds"):
            st.dataframe(
                r["folds"],
                use_container_width=True,
                hide_index=True,
            )
st.markdown("### 🏀 Live NBA Moneylines")

try:
    live_moneylines = get_live_nba_moneylines()

    if live_moneylines:
        moneyline_df = pd.DataFrame(live_moneylines)
        moneyline_df["home_team_code"] = (
            moneyline_df["home_team"].apply(
                normalize_nba_team_name
            )
        )

        moneyline_df["away_team_code"] = (
            moneyline_df["away_team"].apply(
                normalize_nba_team_name
            )
        )

        unmapped_home = moneyline_df[
            moneyline_df["home_team_code"].isna()
        ]["home_team"].unique()

        unmapped_away = moneyline_df[
            moneyline_df["away_team_code"].isna()
        ]["away_team"].unique()

        unmapped_teams = sorted(
            set(unmapped_home) | set(unmapped_away)
        )

        if unmapped_teams:
            st.error(
                "Unmapped NBA teams: "
                + ", ".join(unmapped_teams)
            )
        else:
            st.success(
                "NBA team mapping successful - "
                "all live teams recognized."
            )
        st.success(
            f"Live NBA odds retrieved - "
            f"{len(moneyline_df)} sportsbook lines found."
        )

        st.dataframe(
            moneyline_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "The Odds API connection worked, "
            "but no NBA moneylines are currently available."
        )

except Exception as e:
    st.error(f"Live NBA odds test failed: {e}")
    st.divider()

st.markdown("---")
st.markdown("### 🏈 Football API Connection")

if st.button("Test Football API"):
    try:
        with st.spinner("Connecting to Sportradar..."):
            football_test = test_sportradar_connection()

        st.success(
            f"Sportradar connection successful — "
            f"{football_test['competition_count']} competitions found."
        )

        competitions_df = football_test["competitions"]

        if not competitions_df.empty:
            st.dataframe(
                competitions_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(
                "Connection succeeded, but no competitions were returned."
            )

    except Exception as e:
        st.error(f"Sportradar connection error: {e}")

st.markdown("### 🏈 NFL Historical Seasons")

if st.button("Get NFL Seasons"):
    try:
        with st.spinner("Checking available NFL seasons..."):
            nfl_season_test = get_nfl_seasons()

        st.success(
            f"NFL season lookup successful — "
            f"{nfl_season_test['season_count']} seasons found."
        )

        nfl_seasons_df = nfl_season_test["seasons"]

        if not nfl_seasons_df.empty:
            st.dataframe(
                nfl_seasons_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(
                "Connection succeeded, but no NFL seasons were returned."
            )

    except Exception as e:
        st.error(f"NFL season lookup error: {e}")

st.markdown("### 🏈 NFL Historical Games")

if st.button("Load 2024–25 NFL Games"):
    try:
        with st.spinner("Downloading 2024–25 NFL games..."):
            nfl_games_test = get_nfl_season_games(
                "sr:season:115087"
            )

        st.success(
            f"NFL historical game download successful — "
            f"{nfl_games_test['game_count']} games found."
        )

        nfl_games_df = nfl_games_test["games"]

        if not nfl_games_df.empty:
            st.dataframe(
                nfl_games_df,
                use_container_width=True,
                hide_index=True,
            )

            nfl_audit = audit_nfl_games(nfl_games_df)

            st.markdown("#### 🔍 NFL Dataset Audit")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Games",
                    nfl_audit["total_games"],
                )

            with col2:
                st.metric(
                    "Unique Games",
                    nfl_audit["unique_game_ids"],
                )

            with col3:
                st.metric(
                    "Duplicates",
                    nfl_audit["duplicate_games"],
                )

            with col4:
                st.metric(
                    "Teams",
                    nfl_audit["team_count"],
                )

            col5, col6, col7, col8 = st.columns(4)

            with col5:
                st.metric(
                    "Completed Games",
                    nfl_audit["completed_games"],
                )

            with col6:
                st.metric(
                    "Missing Scores",
                    nfl_audit["missing_home_scores"]
                    + nfl_audit["missing_away_scores"],
                )

            with col7:
                st.metric(
                    "Ties",
                    nfl_audit["ties"],
                )

            with col8:
                home_win_rate = nfl_audit["home_win_rate"]

                st.metric(
                    "Home Win Rate",
                    (
                        f"{home_win_rate:.1%}"
                        if home_win_rate is not None
                        else "N/A"
                    ),
                )

            st.write(
                "Date range:",
                nfl_audit["start_date"],
                "→",
                nfl_audit["end_date"],
            ) 
        else:
            st.warning(
                "Connection succeeded, but no NFL games were returned."
            )

    except Exception as e:
        st.error(f"NFL historical games error: {e}")

st.markdown("---")
st.markdown("### 🏈 Multi-Season NFL Dataset")

if st.button("Load Multi-Season NFL Dataset"):

    try:
        with st.spinner("Downloading multiple NFL seasons..."):

            season_ids = [
                "sr:season:115087",  # 2024-25
                "sr:season:127985",  # 2025-26
            ]

            multi_nfl = get_multiple_nfl_seasons(
                season_ids
            )

            st.session_state["multi_nfl_games"] = multi_nfl["games"]

        st.success(
            f"Multi-season NFL download successful — "
            f"{multi_nfl['game_count']} games found."
        )

        st.markdown("#### Season Summary")
        st.dataframe(
            multi_nfl["season_summary"],
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:
        st.error(f"Multi-season NFL error: {e}")


if "multi_nfl_games" in st.session_state:

    multi_games = st.session_state["multi_nfl_games"]

    st.markdown("#### Combined NFL Games")

    st.dataframe(
        multi_games,
        use_container_width=True,
        hide_index=True,
    )

    multi_audit = {
    "total_games": len(multi_games),
    "unique_games": multi_games["game_id"].nunique(),
    "duplicate_games": multi_games["game_id"].duplicated().sum(),
    "team_count": len(
        set(multi_games["home_team"].dropna())
        | set(multi_games["away_team"].dropna())
    ),
    "completed_games": (
        multi_games["status"].astype(str).str.lower() == "closed"
    ).sum(),
    "missing_scores": (
        multi_games["home_score"].isna()
        | multi_games["away_score"].isna()
    ).sum(),
    "ties": (
        multi_games["home_score"] == multi_games["away_score"]
    ).sum(),
    "home_win_rate": (
        multi_games["home_score"] > multi_games["away_score"]
    ).mean(),
}

    st.markdown("### 🔍 Multi-Season NFL Audit")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Games",
        multi_audit["total_games"],
    )

    c2.metric(
        "Unique Games",
        multi_audit["unique_games"],
    )

    c3.metric(
        "Duplicates",
        multi_audit["duplicate_games"],
    )

    c4.metric(
        "Teams",
        multi_audit["team_count"],
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Completed Games",
        multi_audit["completed_games"],
    )

    c2.metric(
        "Missing Scores",
        multi_audit["missing_scores"],
    )

    c3.metric(
        "Ties",
        multi_audit["ties"],
    )

    c4.metric(
        "Home Win Rate",
        f"{multi_audit['home_win_rate']:.1%}",
    )

# ============================================================
# NFL PREGAME FEATURE ENGINE
# ============================================================

if "multi_nfl_games" in st.session_state:

    st.markdown("---")
    st.markdown("### 🧠 NFL Pregame Feature Engine")

    if st.button("Build NFL Pregame Features"):

        try:
            with st.spinner(
                "Building leakage-safe NFL pregame features..."
            ):
                nfl_feature_games = build_nfl_pregame_features(
                    st.session_state["multi_nfl_games"]
                )

                st.session_state[
                    "nfl_feature_games"
                ] = nfl_feature_games

            st.success(
                f"NFL feature build complete — "
                f"{len(nfl_feature_games)} games processed."
            )

        except Exception as e:
            st.error(
                f"NFL feature engine error: {e}"
            )


if "nfl_feature_games" in st.session_state:

    nfl_features = st.session_state[
        "nfl_feature_games"
    ]

    st.markdown("#### 🧠 NFL Predictive Feature Dataset")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Games",
        len(nfl_features),
    )

    c2.metric(
        "Predictive Features",
        max(len(nfl_features.columns) - 6, 0),
    )

    c3.metric(
        "Training Games",
        nfl_features["home_win"].notna().sum(),
    )

    c4.metric(
        "Ties Excluded",
        nfl_features["home_win"].isna().sum(),
    )

    st.dataframe(
        nfl_features,
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# NFL WALK-FORWARD MODEL VALIDATION
# ============================================================

st.markdown("## 🧠 NFL Predictive Model Validation")

if "multi_nfl_games" not in st.session_state:
    st.info(
        "Load the multi-season NFL dataset first before "
        "running predictive validation."
    )

else:
    nfl_model_games = st.session_state["multi_nfl_games"]

    st.write(
        f"Historical games available for modeling: "
        f"{len(nfl_model_games):,}"
    )

    if st.button("Run NFL Walk-Forward Model"):

        try:
            with st.spinner(
                "Building leakage-safe features and "
                "running NFL walk-forward validation..."
            ):

                nfl_features = build_nfl_pregame_features(
                    nfl_model_games
                )

                nfl_model_result = run_nfl_walkforward_model(
                    nfl_features
                )

                st.session_state[
                    "nfl_walkforward_result"
                ] = nfl_model_result

                st.session_state[
                    "nfl_feature_games"
                ] = nfl_features

            st.success(
                "NFL walk-forward validation complete."
            )

        except Exception as e:
            st.error(
                f"NFL walk-forward validation error: {e}"
            )


if "nfl_walkforward_result" in st.session_state:

    result = st.session_state[
        "nfl_walkforward_result"
    ]

    st.markdown("### 🏈 NFL Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Predictions",
        f"{result['prediction_count']:,}",
    )

    c2.metric(
        "Accuracy",
        f"{result['accuracy']:.1%}",
    )

    c3.metric(
        "AUC",
        f"{result['auc']:.3f}",
    )

    c4.metric(
        "Brier Score",
        f"{result['brier']:.3f}",
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Log Loss",
        f"{result['log_loss']:.3f}",
    )

    c2.metric(
        "Home-Team Baseline",
        f"{result['baseline_home_accuracy']:.1%}",
    )

    c3.metric(
        "Accuracy vs Baseline",
        f"{result['accuracy_vs_baseline']:+.1%}",
    )

    st.markdown("### 🔎 Walk-Forward Predictions")

    prediction_table = result[
        "predictions"
    ].copy()

    prediction_table[
        "home_win_probability"
    ] = prediction_table[
        "probability"
    ].map(
        lambda x: f"{x:.1%}"
    )

    st.dataframe(
        prediction_table[
            [
                "start_time",
                "home_team",
                "away_team",
                "home_win_probability",
                "prediction",
                "actual",
                "correct",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# NFL CONFIDENCE CALIBRATION
# ============================================================

if "nfl_walkforward_result" in st.session_state:

    st.markdown("## 🎯 NFL Confidence Calibration")

    calibration_result = st.session_state[
        "nfl_walkforward_result"
    ]

    calibration_df = calibration_result[
        "predictions"
    ].copy()

    # Confidence = probability assigned to the predicted winner.
    calibration_df["confidence"] = (
        calibration_df["probability"].where(
            calibration_df["prediction"] == 1,
            1.0 - calibration_df["probability"],
        )
    )

    calibration_df["correct"] = (
        calibration_df["prediction"]
        == calibration_df["actual"]
    ).astype(int)

    # Confidence bands.
    calibration_df["confidence_band"] = pd.cut(
        calibration_df["confidence"],
        bins=[
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.01,
        ],
        labels=[
            "50–55%",
            "55–60%",
            "60–65%",
            "65–70%",
            "70–75%",
            "75–80%",
            "80–85%",
            "85–90%",
            "90–95%",
            "95–100%",
        ],
        include_lowest=True,
    )

    calibration_summary = (
        calibration_df
        .groupby(
            "confidence_band",
            observed=False,
        )
        .agg(
            predictions=("correct", "size"),
            correct=("correct", "sum"),
            actual_accuracy=("correct", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )

    calibration_summary["calibration_gap"] = (
        calibration_summary["actual_accuracy"]
        - calibration_summary["avg_confidence"]
    )

    display_calibration = calibration_summary.copy()

    display_calibration["Average Confidence"] = (
        display_calibration["avg_confidence"]
        .map(
            lambda x: f"{x:.1%}"
            if pd.notna(x)
            else "—"
        )
    )

    display_calibration["Actual Accuracy"] = (
        display_calibration["actual_accuracy"]
        .map(
            lambda x: f"{x:.1%}"
            if pd.notna(x)
            else "—"
        )
    )

    display_calibration["Calibration Gap"] = (
        display_calibration["calibration_gap"]
        .map(
            lambda x: f"{x:+.1%}"
            if pd.notna(x)
            else "—"
        )
    )

    display_calibration = display_calibration[
        [
            "confidence_band",
            "predictions",
            "correct",
            "Average Confidence",
            "Actual Accuracy",
            "Calibration Gap",
        ]
    ]

    display_calibration = display_calibration.rename(
        columns={
            "confidence_band": "Confidence Band",
            "predictions": "Predictions",
            "correct": "Correct",
        }
    )

    st.dataframe(
        display_calibration,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # HIGH-CONFIDENCE PERFORMANCE
    # --------------------------------------------------------

    st.markdown("### 🔥 High-Confidence Performance")

    confidence_thresholds = [
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    threshold_rows = []

    for threshold in confidence_thresholds:

        subset = calibration_df[
            calibration_df["confidence"] >= threshold
        ]

        if len(subset) == 0:
            continue

        threshold_rows.append(
            {
                "Minimum Confidence": threshold,
                "Predictions": len(subset),
                "Correct": int(
                    subset["correct"].sum()
                ),
                "Accuracy": subset["correct"].mean(),
                "Average Confidence": subset[
                    "confidence"
                ].mean(),
            }
        )
    
    threshold_df = pd.DataFrame(
        threshold_rows
    )

    if not threshold_df.empty:

        threshold_display = threshold_df.copy()

        threshold_display[
            "Minimum Confidence"
        ] = threshold_display[
            "Minimum Confidence"
        ].map(
            lambda x: f"{x:.0%}+"
        )

        threshold_display[
            "Accuracy"
        ] = threshold_display[
            "Accuracy"
        ].map(
            lambda x: f"{x:.1%}"
        )

        threshold_display[
            "Average Confidence"
        ] = threshold_display[
            "Average Confidence"
        ].map(
            lambda x: f"{x:.1%}"
        )

        st.dataframe(
            threshold_display,
            use_container_width=True,
            hide_index=True,
        )

# Save model predictions for later decision-engine/live-engine work.
calibration_result = st.session_state.get(
    "nfl_walkforward_result"
)

if calibration_result is not None:
    nfl_model_predictions = calibration_result[
        "predictions"
    ].copy()

    if (
        "home_win_probability" not in nfl_model_predictions.columns
        and "probability" in nfl_model_predictions.columns
    ):
        nfl_model_predictions["home_win_probability"] = (
            nfl_model_predictions["probability"]
        )

    st.session_state[
        "nfl_calibration_predictions"
    ] = nfl_model_predictions
    
# ---------------------------------------------------------
# NFL DECISION ENGINE
# ---------------------------------------------------------

st.markdown("### 🧠 NFL Decision Engine")

try:
    nfl_predictions = st.session_state.get(
        "nfl_calibration_predictions",
        pd.DataFrame()
    ).copy()

    if nfl_predictions.empty:
        raise ValueError(
            "Run the NFL historical model first to load NFL model predictions."
        )

    if "home_win_probability" not in nfl_predictions.columns and "probability" in nfl_predictions.columns:
        nfl_predictions["home_win_probability"] = nfl_predictions["probability"]
    
    if (
        "home_win_probability" not in nfl_predictions.columns
        and "probability" in nfl_predictions.columns
    ):
        nfl_predictions["home_win_probability"] = nfl_predictions["probability"]

    decision_profile = build_nfl_confidence_profile(
        nfl_predictions
    )

    decision_rows = []

    for _, game in nfl_predictions.iterrows():

        decision = get_nfl_decision(
            home_team=game["home_team"],
            away_team=game["away_team"],
            home_win_probability=game["home_win_probability"],
            confidence_profile=decision_profile,
        )

        decision_rows.append(
            {
                "start_time": game["start_time"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "predicted_team": decision["predicted_team"],
                "confidence": decision["confidence"],
                "decision": decision["decision"],
                "historical_accuracy": decision["historical_accuracy"],
                "historical_sample": decision["historical_sample"],
                "reason": decision["reason"],
                "actual": game["actual"],
                "correct": game["correct"],
            }
        )

    nfl_decisions = pd.DataFrame(decision_rows)

    st.session_state["nfl_decisions"] = nfl_decisions
    st.session_state["nfl_confidence_profile"] = decision_profile

    bet_count = (nfl_decisions["decision"] == "BET").sum()
    lean_count = (nfl_decisions["decision"] == "LEAN").sum()
    pass_count = (nfl_decisions["decision"] == "PASS").sum()

    c1, c2, c3 = st.columns(3)

    c1.metric("BET", int(bet_count))
    c2.metric("LEAN", int(lean_count))
    c3.metric("PASS", int(pass_count))

    bet_games = nfl_decisions[
        nfl_decisions["decision"] == "BET"
    ].copy()

    if not bet_games.empty:

        bet_accuracy = bet_games["correct"].mean()

        st.markdown("#### 🔥 Historical BET Performance")

        b1, b2, b3 = st.columns(3)

        b1.metric("Qualified Bets", len(bet_games))
        b2.metric("Correct", int(bet_games["correct"].sum()))
        b3.metric("Accuracy", f"{bet_accuracy:.1%}")

        display_bets = bet_games.copy()

        display_bets["confidence"] = (
            display_bets["confidence"]
            .map(lambda x: f"{x:.1%}")
        )

        display_bets["historical_accuracy"] = (
            display_bets["historical_accuracy"]
            .map(
                lambda x: f"{x:.1%}"
                if pd.notna(x)
                else "—"
            )
        )

        st.dataframe(
            display_bets,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No historical predictions currently "
            "qualify as BET decisions."
        )

except Exception as e:
    st.error(f"NFL Decision Engine error: {e}")

# --------------------------------------------------
# NFL MARKET EDGE TEST
# --------------------------------------------------

st.markdown("### 💰 NFL Market Edge Test")

test_home_probability = st.number_input(
    "Home model probability",
    min_value=0.01,
    max_value=0.99,
    value=0.70,
    step=0.01,
)

test_home_odds = st.number_input(
    "Home moneyline",
    value=-150,
    step=5,
)

test_away_odds = st.number_input(
    "Away moneyline",
    value=130,
    step=5,
)

if st.button("Test NFL Market Edge"):

    try:
        edge_result = calculate_nfl_market_edge(
            home_model_probability=test_home_probability,
            home_odds=test_home_odds,
            away_odds=test_away_odds,
        )

        if edge_result["best_side"] == "HOME":
            selected_probability = edge_result[
                "home_model_probability"
            ]
            selected_market_probability = edge_result[
                "home_no_vig_probability"
            ]
        else:
            selected_probability = edge_result[
                "away_model_probability"
            ]
            selected_market_probability = edge_result[
                "away_no_vig_probability"
            ]

        classification = classify_nfl_market_edge(
            model_probability=selected_probability,
            market_probability=selected_market_probability,
            american_odds=edge_result["best_odds"],
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Best Side",
            edge_result["best_side"],
        )

        col2.metric(
            "Model Probability",
            f"{selected_probability:.1%}",
        )

        col3.metric(
            "Market No-Vig Probability",
            f"{selected_market_probability:.1%}",
        )

        col4.metric(
            "Model Edge",
            f"{edge_result['best_edge']:+.1%}",
        )

        st.metric(
            "Expected Value / $1",
            f"{classification['expected_value']:+.3f}",
        )

        decision = classification["decision"]

        if decision == "BET":
            st.success(
                f"BET — {classification['reason']}"
            )
        elif decision == "LEAN":
            st.warning(
                f"LEAN — {classification['reason']}"
            )
        else:
            st.info(
                f"PASS — {classification['reason']}"
            )

    except Exception as e:
        st.error(
            f"NFL Market Edge Test error: {e}"
        )

# --------------------------------------------------
# LIVE NFL MONEYLINES
# --------------------------------------------------

st.markdown("### 📡 Live NFL Moneylines")

if st.button("Load Live NFL Moneylines"):
    try:
        with st.spinner("Loading current NFL moneylines..."):
            live_nfl_odds = get_live_nfl_moneylines()

        if not live_nfl_odds:
            st.info(
                "No current NFL moneyline markets were returned."
            )
        else:
            live_nfl_odds_df = pd.DataFrame(live_nfl_odds)

            st.success(
                f"Loaded {len(live_nfl_odds_df)} "
                "NFL sportsbook moneyline markets."
            )

            st.dataframe(
                live_nfl_odds_df,
                use_container_width=True,
                hide_index=True,
            )

            st.session_state["live_nfl_odds"] = (
                live_nfl_odds_df
            )

    except Exception as e:
        st.error(f"Live NFL odds error: {e}")

# --------------------------------------------------
# NFL BEST-LINE SHOPPER
# --------------------------------------------------

st.markdown("### 🛒 NFL Best-Line Shopper")

if "live_nfl_odds" not in st.session_state:
    st.info(
        "Load Live NFL Moneylines first."
    )

else:
    if st.button("Find Best NFL Moneylines"):
        try:
            odds_rows = (
                st.session_state["live_nfl_odds"]
                .to_dict("records")
            )

            best_lines = get_best_nfl_moneylines(
                odds_rows
            )

            best_lines_df = pd.DataFrame(best_lines)

            st.session_state["nfl_best_lines"] = (
                best_lines_df

            )

            st.success(
                f"Found best available lines for "
                f"{len(best_lines_df)} NFL games."
            )

            st.dataframe(
                best_lines_df,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as e:
            st.error(
                f"NFL best-line error: {e}"
            )

# ------------------------------------------------------------
# LIVE NFL OPPORTUNITY ENGINE
# ------------------------------------------------------------

st.markdown("### 🧠 Live NFL Opportunity Engine")

if (
    "nfl_best_lines" in st.session_state
    and "nfl_calibration_predictions" in st.session_state
):
    if st.button("Analyze Live NFL Opportunities"):

        try:
            live_opportunities = build_live_nfl_opportunities(
                model_predictions=st.session_state[
                    "nfl_calibration_predictions"
                ],
                best_lines=st.session_state[
                    "nfl_best_lines"
                ],
            )

            st.session_state[
                "nfl_live_opportunities"
            ] = live_opportunities

            if live_opportunities.empty:
                st.warning(
                    "No live NFL games matched the available "
                    "model predictions."
                )
            else:
                st.success(
                    f"Analyzed {len(live_opportunities)} "
                    "live NFL opportunities."
                )

        except Exception as e:
            st.error(
                f"Live NFL Opportunity Engine error: {e}"
            )

else:
    st.info(
        "Load NFL model predictions and run the "
        "Best-Line Shopper first."
    )


if "nfl_live_opportunities" in st.session_state:

    live_display = st.session_state[
        "nfl_live_opportunities"
    ].copy()

    if not live_display.empty:

        for column in [
            "home_win_probability",
            "model_probability",
            "market_no_vig_probability",
            "model_edge",
        ]:
            if column in live_display.columns:
                live_display[column] = (
                    live_display[column]
                    .map(lambda x: f"{x:.1%}")
                )

        if "expected_value" in live_display.columns:
            live_display["expected_value"] = (
                live_display["expected_value"]
                .map(lambda x: f"{x:+.3f}")
            )

        st.dataframe(
            live_display,
            use_container_width=True,
            hide_index=True,
        )

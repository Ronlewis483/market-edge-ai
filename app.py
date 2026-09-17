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
                    [2022, 2023]
                )

            st.success(
                f"Multi-season download successful — "
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
            st.markdown("#### 🧠 Research Pipeline Bridge Test")
    
            research_games = prepare_balldontlie_games_for_research(
                multi_bdl_test["games"]
            )
    
            st.success(
                f"Research bridge successful — "
                f"{len(research_games)} games ready for modeling."
            )
    
            bridge_col1, bridge_col2, bridge_col3 = st.columns(3)
    
            bridge_col1.metric(
                "Model-Ready Games",
                len(research_games),
            )
    
            bridge_col2.metric(
                "First Game",
                research_games["game_date"].min().strftime("%Y-%m-%d"),
            )
    
            bridge_col3.metric(
                "Last Game",
                research_games["game_date"].max().strftime("%Y-%m-%d"),
            )
    
            st.write(
                "Verified home-win rate:",
                f"{research_games['home_win'].mean():.1%}",
        )
            st.markdown("#### 🧮 Pre-Game Feature Engine Test")
    
    
    
            
            feature_games = build_balldontlie_pregame_features(
                multi_bdl_test["games"],
                min_games=5,
            )
    
            st.success(
                f"Feature engine successful — "
                f"{len(feature_games)} model-ready rows created."
            )
    
            feature_col1, feature_col2, feature_col3 = st.columns(3)
    
            feature_col1.metric(
                "Feature Rows",
                len(feature_games),
            )
    
            feature_col2.metric(
                "First Feature Date",
                feature_games["game_date"].min().strftime("%Y-%m-%d"),
            )
    
            feature_col3.metric(
                "Last Feature Date",
                feature_games["game_date"].max().strftime("%Y-%m-%d"),
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
                len(leakage_audit["suspicious_model_features"]),
            )
    
            st.write(
                "Model feature columns:",
                leakage_audit["model_features"],
            )
    
            if leakage_audit["suspicious_model_features"]:
                st.error(
                    "Possible leakage detected: "
                    + ", ".join(
                        leakage_audit["suspicious_model_features"]
                    )
                )
            else:
                st.success(
                    "No obvious current-game outcome leakage "
                    "detected in the proposed model features."
                )
    
            if len(leakage_audit["missing_feature_values"]) > 0:
                st.warning("Missing values found in model features.")
                st.dataframe(
                    leakage_audit["missing_feature_values"]
                    .rename("missing_values")
                    .reset_index()
                    .rename(columns={"index": "feature"}),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success(
                    "No missing values found in model features."
                )



            st.markdown("#### 🧠 Walk-Forward Model Test")

            walkforward_results = run_balldontlie_walkforward_model(
                feature_games
            )

           st.success(
                f"Walk-forward validation successful — "
                f"{walkforward_results['games_predicted']} "
                f"unseen games predicted."
           )
    
                
                
                
                audit = audit_historical_games(
                    multi_bdl_test["games"]
                )
        
                st.markdown("#### 🔎 Historical Dataset Audit")
        
                col1, col2, col3 = st.columns(3)
        
                col1.metric(
                    "Total Rows",
                    audit["total_rows"],
                )
        
                col2.metric(
                    "Unique Game IDs",
                    audit["unique_game_ids"],
                )
        
                col3.metric(
                    "Duplicate Game IDs",
                    audit["duplicate_game_ids"],
                )
        
                col4, col5, col6 = st.columns(3)
        
                col4.metric(
                    "Missing Scores",
                    audit["missing_scores"],
                )
        
                col5.metric(
                    "Tied Games",
                    audit["tied_games"],
                )
        
                col6.metric(
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
                        "games have suspicious scores."
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
        x.strip()
        for x in nba_seasons_text.split(",")
        if x.strip()
    ]

    try:
        with st.spinner(
            "Downloading multiple NBA seasons and running walk-forward validation..."
        ):
            multi_nba_result = run_multi_season_nba_research(
                seasons,
                minimum_training_games=250,
                test_block_size=100,
            )

            st.session_state["multi_nba_research_result"] = multi_nba_result

        st.success("Multi-season NBA validation complete.")

    except Exception as e:
        st.error(f"Multi-season NBA validation error: {e}")

if "multi_nba_research_result" in st.session_state:

    mr = st.session_state["multi_nba_research_result"]

    st.markdown("#### 🏀 Multi-Season Dataset")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Raw games", mr["raw_games"])
    m2.metric("Completed games", mr["completed_games"])
    m3.metric("Feature rows", mr["feature_rows"])
    m4.metric("Home win rate", f"{mr['home_win_rate']:.1%}")

    st.markdown("#### Walk-Forward Performance")

    p1, p2, p3, p4 = st.columns(4)

    overall = mr["overall"]

    p1.metric("AUC", f"{overall['auc']:.3f}")
    p2.metric("Accuracy", f"{overall['accuracy']:.1%}")
    p3.metric("Brier Score", f"{overall['brier']:.4f}")
    p4.metric("Log Loss", f"{overall['logloss']:.4f}")

    st.markdown("#### Model vs Baseline")

    baseline = mr["baseline"]

    multi_comparison = pd.DataFrame(
        [
            {
                "Model": "NBA Multi-Season Model",
                "Accuracy": overall["accuracy"],
                "Brier": overall["brier"],
                "Log Loss": overall["logloss"],
                "AUC": overall["auc"],
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
        multi_comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Multi-Season Confidence Analysis")

    if len(mr["confidence"]):
        st.dataframe(
            mr["confidence"],
            use_container_width=True,
            hide_index=True,
        )
  
    else:
        st.info("No confidence-band results were generated.")

    st.markdown("#### 🎯 Probability Calibration")

    calibration_result = nba_calibration_summary(
        mr["predictions"]
    )

    calibration_table = calibration_result["calibration"]

    if len(calibration_table):
        calibration_display = calibration_table.copy()

        calibration_display["avg_predicted_probability"] = (
            calibration_display["avg_predicted_probability"]
            .map(lambda x: f"{x:.1%}")
        )

        calibration_display["actual_win_rate"] = (
            calibration_display["actual_win_rate"]
            .map(lambda x: f"{x:.1%}")
        )

        calibration_display["calibration_gap"] = (
            calibration_display["calibration_gap"]
            .map(lambda x: f"{x:+.1%}")
        )

        calibration_display["absolute_calibration_error"] = (
            calibration_display["absolute_calibration_error"]
            .map(lambda x: f"{x:.1%}")
        )

        st.dataframe(
            calibration_display,
            use_container_width=True,
            hide_index=True,
        )

        weighted_error = calibration_result[
            "weighted_calibration_error"
        ]

        if weighted_error is not None:
            st.metric(
                "Weighted Calibration Error",
                f"{weighted_error:.1%}",
            )

    else:
        st.info("No calibration results were generated.")

    st.markdown("#### 🔬 High-Confidence Threshold Analysis")

    high_confidence = calibration_result["high_confidence"]

    if len(high_confidence):
        threshold_display = high_confidence.copy()

        threshold_display["minimum_probability"] = (
            threshold_display["minimum_probability"]
            .map(lambda x: f"{x:.0%}")
        )

        threshold_display["accuracy"] = (
            threshold_display["accuracy"]
            .map(lambda x: f"{x:.1%}")
        )

        threshold_display["avg_model_probability"] = (
            threshold_display["avg_model_probability"]
            .map(lambda x: f"{x:.1%}")
        )

        threshold_display["calibration_gap"] = (
            threshold_display["calibration_gap"]
            .map(lambda x: f"{x:+.1%}")
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

    st.markdown("#### Season Breakdown")
    st.markdown("#### Season Breakdown")

    st.dataframe(
        mr["season_summary"],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Multi-Season Walk-Forward Folds"):
        st.dataframe(
            mr["folds"],
            use_container_width=True,
            hide_index=True,
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

    st.divider()

"""Game-status normalization; no elapsed-time settlement or invented outcomes."""
import re
import pandas as pd
import requests


def team_identity(value):
    name = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return name.replace("losangeles", "la")


def public_statuses(league, now=None):
    routes = {"NFL": "football/nfl", "NBA": "basketball/nba", "MLB": "baseball/mlb"}
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    results = []
    for offset in range(3):
        day = (current.tz_convert("America/Chicago") - pd.Timedelta(days=offset)).strftime("%Y%m%d")
        try:
            response = requests.get("https://site.api.espn.com/apis/site/v2/sports/" + routes[league] + "/scoreboard",
                                    params={"dates": day, "limit": 100}, timeout=(3, 5))
            response.raise_for_status()
            for event in response.json().get("events", []):
                for game in event.get("competitions", []):
                    teams = {t.get("homeAway"): t.get("team", {}).get("displayName") for t in game.get("competitors", [])}
                    status = game.get("status", event.get("status", {})).get("type", {})
                    if not teams.get("home") or not teams.get("away"):
                        continue
                    name = str(status.get("name", "")).upper()
                    results.append({"home_team": teams["home"], "away_team": teams["away"],
                        "commence_time": game.get("date", event.get("date")),
                        "completed": status.get("completed") is True and not any(word in name for word in ("POSTPON", "CANCEL", "SUSPEND")),
                        "in_progress": status.get("state") == "in",
                        "unavailable": any(word in name for word in ("POSTPON", "CANCEL", "SUSPEND"))})
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            continue
    return results


def display_rows(frame):
    if frame is None or frame.empty or "pick_display_state" not in frame:
        return frame
    return frame.loc[~frame["pick_display_state"].isin(["archived_unverified", "unavailable"])].copy()

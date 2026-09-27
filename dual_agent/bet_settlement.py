import re
import requests
import streamlit as st

from dual_agent.supabase_db import update_bet_result


# ==========================================
# MARKET EDGE AI — AUTOMATIC BET SETTLEMENT
# VERSION 1
#
# Supported:
# - NFL
# - NBA
# - MLB
# - College Football
#
# Markets:
# - Game Winner / Moneyline
# - Spread
# - Game Total
#
# Player props intentionally NOT settled yet.
# ==========================================


SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "College Football": "americanfootball_ncaaf",
}


def _clean_text(value):
    return str(value or "").strip()


def _normalize(value):
    """
    Normalize team names for safer matching.
    """
    value = _clean_text(value).lower()

    value = re.sub(
        r"[^a-z0-9 ]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def _team_matches(saved_team, api_team):
    """
    Conservative team-name matching.

    Examples:
        Seahawks
        Seattle Seahawks

    Both should match.
    """

    saved = _normalize(saved_team)
    api = _normalize(api_team)

    if not saved or not api:
        return False

    if saved == api:
        return True

    if saved in api:
        return True

    if api in saved:
        return True

    saved_words = saved.split()
    api_words = api.split()

    if saved_words and api_words:
        if saved_words[-1] == api_words[-1]:
            return True

    return False


@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def get_recent_scores(sport_key):
    """
    Fetch live and recently completed games.

    Cache for 15 minutes so Streamlit reruns do
    not repeatedly consume API credits.
    """

    api_key = st.secrets["ODDS_API_KEY"]

    url = (
        "https://api.the-odds-api.com/v4/"
        f"sports/{sport_key}/scores"
    )

    response = requests.get(
        url,
        params={
            "apiKey": api_key,
            "daysFrom": 3,
            "dateFormat": "iso",
        },
        timeout=20,
    )

    response.raise_for_status()

    games = response.json()

    quota = {
        "remaining": response.headers.get(
            "x-requests-remaining"
        ),
        "used": response.headers.get(
            "x-requests-used"
        ),
        "last": response.headers.get(
            "x-requests-last"
        ),
    }

    return games, quota


def _extract_score(game, team_name):
    """
    Return the score for a specific team.
    """

    scores = game.get("scores") or []

    for score_data in scores:

        api_team = score_data.get("name")

        if _team_matches(
            team_name,
            api_team,
        ):

            try:
                return float(
                    score_data.get("score")
                )

            except (
                TypeError,
                ValueError,
            ):
                return None

    return None


def _find_game(
    games,
    team_name=None,
    description=None,
):
    """
    Find the completed game associated with
    a saved bet.

    Version 1 uses conservative team-name matching.
    """

    team_name = _clean_text(team_name)
    description = _clean_text(description)

    completed_games = [
        game
        for game in games
        if game.get("completed") is True
    ]

    # --------------------------------------
    # PRIMARY MATCH:
    # saved team_name
    # --------------------------------------

    if team_name:

        matches = []

        for game in completed_games:

            home_team = game.get(
                "home_team",
                "",
            )

            away_team = game.get(
                "away_team",
                "",
            )

            if (
                _team_matches(
                    team_name,
                    home_team,
                )
                or _team_matches(
                    team_name,
                    away_team,
                )
            ):
                matches.append(game)

        if len(matches) == 1:
            return matches[0]

    # --------------------------------------
    # FALLBACK:
    # look for either API team name inside
    # the saved description
    # --------------------------------------

    normalized_description = _normalize(
        description
    )

    if normalized_description:

        matches = []

        for game in completed_games:

            home_team = game.get(
                "home_team",
                "",
            )

            away_team = game.get(
                "away_team",
                "",
            )

            home_normalized = _normalize(
                home_team
            )

            away_normalized = _normalize(
                away_team
            )

            home_nickname = (
                home_normalized.split()[-1]
                if home_normalized
                else ""
            )

            away_nickname = (
                away_normalized.split()[-1]
                if away_normalized
                else ""
            )

            home_found = (
                home_normalized
                in normalized_description
                or (
                    home_nickname
                    and home_nickname
                    in normalized_description
                )
            )

            away_found = (
                away_normalized
                in normalized_description
                or (
                    away_nickname
                    and away_nickname
                    in normalized_description
                )
            )

            if home_found or away_found:
                matches.append(game)

        if len(matches) == 1:
            return matches[0]

    return None


def _american_profit(
    wager,
    odds,
):
    """
    Profit only — stake excluded.
    """

    wager = float(wager)
    odds = float(odds)

    if odds > 0:

        return (
            wager
            * odds
            / 100.0
        )

    return (
        wager
        * 100.0
        / abs(odds)
    )


def _settle_moneyline(
    bet,
    game,
):
    """
    Settle a moneyline/game-winner bet.
    """

    selected_team = _clean_text(
        bet.get("team_name")
    )

    if not selected_team:

        selected_team = _clean_text(
            bet.get("bet_description")
        )

    home_team = game.get(
        "home_team",
        "",
    )

    away_team = game.get(
        "away_team",
        "",
    )

    home_score = _extract_score(
        game,
        home_team,
    )

    away_score = _extract_score(
        game,
        away_team,
    )

    if (
        home_score is None
        or away_score is None
    ):
        return None

    if home_score == away_score:
        return "Push"

    winner = (
        home_team
        if home_score > away_score
        else away_team
    )

    if _team_matches(
        selected_team,
        winner,
    ):
        return "Won"

    if (
        _team_matches(
            selected_team,
            home_team,
        )
        or _team_matches(
            selected_team,
            away_team,
        )
    ):
        return "Lost"

    return None


def _settle_spread(
    bet,
    game,
):
    """
    Settle a standard point-spread bet.

    Example:
        49ers -3.5

    adjusted score:
        selected team score + spread
    """

    selected_team = _clean_text(
        bet.get("team_name")
    )

    if not selected_team:
        return None

    try:
        spread = float(
            bet.get("betting_line")
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    home_team = game.get(
        "home_team",
        "",
    )

    away_team = game.get(
        "away_team",
        "",
    )

    home_score = _extract_score(
        game,
        home_team,
    )

    away_score = _extract_score(
        game,
        away_team,
    )

    if (
        home_score is None
        or away_score is None
    ):
        return None

    if _team_matches(
        selected_team,
        home_team,
    ):

        selected_score = home_score
        opponent_score = away_score

    elif _team_matches(
        selected_team,
        away_team,
    ):

        selected_score = away_score
        opponent_score = home_score

    else:
        return None

    adjusted_score = (
        selected_score
        + spread
    )

    if adjusted_score > opponent_score:
        return "Won"

    if adjusted_score < opponent_score:
        return "Lost"

    return "Push"


def _settle_total(
    bet,
    game,
):
    """
    Settle full-game Over/Under.
    """

    try:
        line = float(
            bet.get("betting_line")
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    direction = _clean_text(
        bet.get("bet_type")
    ).lower()

    if direction not in [
        "over",
        "under",
    ]:
        return None

    home_team = game.get(
        "home_team",
        "",
    )

    away_team = game.get(
        "away_team",
        "",
    )

    home_score = _extract_score(
        game,
        home_team,
    )

    away_score = _extract_score(
        game,
        away_team,
    )

    if (
        home_score is None
        or away_score is None
    ):
        return None

    total_score = (
        home_score
        + away_score
    )

    if total_score == line:
        return "Push"

    if direction == "over":

        return (
            "Won"
            if total_score > line
            else "Lost"
        )

    return (
        "Won"
        if total_score < line
        else "Lost"
    )


def _calculate_profit_loss(
    bet,
    result,
):
    """
    Convert settlement result into P/L.
    """

    try:
        wager = float(
            bet.get("wager") or 0
        )

        odds = float(
            bet.get("odds") or 0
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if result == "Won":

        if (
            wager <= 0
            or abs(odds) < 100
        ):
            return None

        return round(
            _american_profit(
                wager,
                odds,
            ),
            2,
        )

    if result == "Lost":
        return round(
            -wager,
            2,
        )

    if result == "Push":
        return 0.0

    return None


def _settle_single_bet(
    bet,
    games,
):
    """
    Attempt to settle one saved bet.

    Returns None if Market Edge cannot safely
    determine the result.
    """

    market = _clean_text(
        bet.get("betting_market")
    ).lower()

    game = _find_game(
        games=games,
        team_name=bet.get(
            "team_name"
        ),
        description=bet.get(
            "bet_description"
        ),
    )

    if game is None:
        return None

    if market in [
        "game winner",
        "moneyline",
    ]:

        result = _settle_moneyline(
            bet,
            game,
        )

    elif market == "spread":

        result = _settle_spread(
            bet,
            game,
        )

    elif market in [
        "game total",
        "total",
    ]:

        result = _settle_total(
            bet,
            game,
        )

    else:
        return None

    if result is None:
        return None

    profit_loss = _calculate_profit_loss(
        bet,
        result,
    )

    if profit_loss is None:
        return None

    return {
        "result": result,
        "profit_loss": profit_loss,
        "game": game,
    }


def auto_settle_bets(bets):
    """
    Check pending game-level bets and automatically
    update completed bets.

    One scores request is made per sport, not per bet.

    Player props and parlays are intentionally skipped
    in Version 1.
    """

    if not bets:

        return {
            "checked": 0,
            "settled": 0,
            "updated": [],
            "skipped": 0,
            "quota": {},
            "errors": [],
        }

    pending_bets = [
        bet
        for bet in bets
        if _clean_text(
            bet.get("status")
        ).lower()
        in [
            "pending",
            "open",
            "live",
        ]
    ]

    supported_bets = []

    for bet in pending_bets:

        sport = _clean_text(
            bet.get("sport")
        )

        market = _clean_text(
            bet.get("betting_market")
        ).lower()

        bet_type = _clean_text(
            bet.get("bet_type")
        ).lower()

        # Skip parlays in Version 1.
        if bet_type == "parlay":
            continue

        if sport not in SPORT_KEYS:
            continue

        if market not in [
            "game winner",
            "moneyline",
            "spread",
            "game total",
            "total",
        ]:
            continue

        supported_bets.append(bet)

    if not supported_bets:

        return {
            "checked": 0,
            "settled": 0,
            "updated": [],
            "skipped": len(
                pending_bets
            ),
            "quota": {},
            "errors": [],
        }

    # --------------------------------------
    # Group bets by sport.
    #
    # This is the important credit-saving
    # behavior: ONE scores request can settle
    # multiple tickets.
    # --------------------------------------

    bets_by_sport = {}

    for bet in supported_bets:

        sport = _clean_text(
            bet.get("sport")
        )

        bets_by_sport.setdefault(
            sport,
            [],
        ).append(bet)

    updated = []
    errors = []
    quota = {}
    checked = 0

    for sport, sport_bets in bets_by_sport.items():

        sport_key = SPORT_KEYS[
            sport
        ]

        try:

            games, sport_quota = (
                get_recent_scores(
                    sport_key
                )
            )

            quota[sport] = sport_quota

        except Exception as error:

            errors.append(
                f"{sport}: {error}"
            )

            continue

        for bet in sport_bets:

            checked += 1

            settlement = (
                _settle_single_bet(
                    bet,
                    games,
                )
            )

            if settlement is None:
                continue

            success, response = (
                update_bet_result(
                    bet_id=bet["id"],
                    status=settlement[
                        "result"
                    ],
                    profit_loss=settlement[
                        "profit_loss"
                    ],
                )
            )

            if success:

                game = settlement[
                    "game"
                ]

                updated.append(
                    {
                        "bet_id": bet[
                            "id"
                        ],
                        "description": bet.get(
                            "bet_description"
                        ),
                        "result": settlement[
                            "result"
                        ],
                        "profit_loss": settlement[
                            "profit_loss"
                        ],
                        "home_team": game.get(
                            "home_team"
                        ),
                        "away_team": game.get(
                            "away_team"
                        ),
                    }
                )

            else:

                errors.append(
                    f"Bet {bet.get('id')}: "
                    f"{response}"
                )

    return {
        "checked": checked,
        "settled": len(updated),
        "updated": updated,
        "skipped": (
            len(pending_bets)
            - len(supported_bets)
        ),
        "quota": quota,
        "errors": errors,
    }

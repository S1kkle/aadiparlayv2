from __future__ import annotations

from datetime import datetime
from typing import Any

from app.models.core import Prop, SportId


def _parse_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def _parse_int(x: Any) -> int | None:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        try:
            return int(x)
        except ValueError:
            return None
    return None


def american_to_implied_prob(american: int) -> float:
    if american < 0:
        return (-american) / ((-american) + 100.0)
    return 100.0 / (american + 100.0)


def _sport_from_any(player: dict[str, Any] | None, game: dict[str, Any] | None) -> SportId:
    raw = None
    if player:
        raw = player.get("sport_id")
    if not raw and game:
        raw = game.get("sport_id")
    if raw in ("NBA", "NFL", "NHL"):
        return raw  # type: ignore[return-value]
    if raw in ("SOCCER", "Soccer", "soccer"):
        return "SOCCER"
    if raw in ("MMA", "UFC"):
        return "MMA"
    return "UNKNOWN"


def normalize_underdog_over_under_lines(payload: dict[str, Any]) -> list[Prop]:
    """
    Normalizes Underdog `over_under_lines` payloads into per-option props.

    Works with the v1 shape you pasted (top-level keys: players/appearances/games/over_under_lines)
    and the commonly-seen beta/v5 shape used by the web app.
    """
    players = {p.get("id"): p for p in (payload.get("players") or []) if isinstance(p, dict)}
    appearances = {a.get("id"): a for a in (payload.get("appearances") or []) if isinstance(a, dict)}
    games = {g.get("id"): g for g in (payload.get("games") or []) if isinstance(g, dict)}

    out: list[Prop] = []
    for line in payload.get("over_under_lines") or []:
        if not isinstance(line, dict):
            continue

        ou = line.get("over_under") or {}
        appearance_stat = (ou.get("appearance_stat") or {}) if isinstance(ou, dict) else {}

        appearance_id = appearance_stat.get("appearance_id")
        stat = appearance_stat.get("stat")
        display_stat = appearance_stat.get("display_stat")

        line_value = _parse_float(line.get("stat_value"))
        if line_value is None or not stat:
            continue

        player: dict[str, Any] | None = None
        game: dict[str, Any] | None = None
        game_title: str | None = None
        scheduled_at: datetime | None = None
        team_abbr: str | None = None
        opp_abbr: str | None = None

        if appearance_id and appearance_id in appearances:
            app = appearances[appearance_id]
            player_id = app.get("player_id")
            match_id = app.get("match_id")
            if player_id and player_id in players:
                player = players[player_id]
            if match_id and match_id in games:
                game = games[match_id]
                game_title = game.get("title") or game.get("short_title") or game.get("abbreviated_title")
                abbr_title = game.get("abbreviated_title") or game.get("title") or ""
                if isinstance(abbr_title, str) and "@" in abbr_title:
                    away, home = [p.strip() for p in abbr_title.split("@", 1)]
                    home_team_id = game.get("home_team_id")
                    away_team_id = game.get("away_team_id")
                    if player and player.get("team_id") == home_team_id:
                        team_abbr, opp_abbr = home, away
                    elif player and player.get("team_id") == away_team_id:
                        team_abbr, opp_abbr = away, home
                    else:
                        team_abbr, opp_abbr = None, None
                elif isinstance(abbr_title, str) and " vs" in abbr_title.lower():
                    # MMA / 1v1 sport: "Fighter A vs. Fighter B" or "Fighter A vs Fighter B"
                    import re as _re
                    parts = _re.split(r"\s+vs\.?\s+", abbr_title, maxsplit=1, flags=_re.IGNORECASE)
                    if len(parts) == 2:
                        a_name, b_name = parts[0].strip(), parts[1].strip()
                        player_full = ""
                        if player:
                            first = player.get("first_name") or ""
                            last = player.get("last_name") or ""
                            player_full = (first + " " + last).strip().lower()
                        if player_full and player_full in a_name.lower():
                            opp_abbr = b_name
                        elif player_full and player_full in b_name.lower():
                            opp_abbr = a_name
                        else:
                            opp_abbr = None
                dt_raw = game.get("scheduled_at")
                if isinstance(dt_raw, str):
                    try:
                        scheduled_at = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
                    except ValueError:
                        scheduled_at = None

        player_name: str | None = None
        underdog_player_id: str | None = None
        if player:
            underdog_player_id = player.get("id")
            first = player.get("first_name") or ""
            last = player.get("last_name") or ""
            full = (first + " " + last).strip()
            player_name = full or player.get("full_name")

        for opt in line.get("options") or []:
            if not isinstance(opt, dict):
                continue
            if (opt.get("status") or "active") != "active":
                continue

            choice = (opt.get("choice") or "").lower()
            if choice == "higher":
                side = "over"
            elif choice == "lower":
                side = "under"
            else:
                continue

            american = _parse_int(opt.get("american_price"))
            dec = _parse_float(opt.get("decimal_price"))
            implied = american_to_implied_prob(american) if american is not None else None

            resolved_player_name = player_name
            if not resolved_player_name:
                sel = opt.get("selection_header")
                if isinstance(sel, str) and sel.strip():
                    resolved_player_name = sel.strip()
                else:
                    resolved_player_name = "Unknown Player"

            out.append(
                Prop(
                    sport=_sport_from_any(player, game),
                    league=None,
                    player_name=resolved_player_name,
                    underdog_player_id=underdog_player_id,
                    underdog_option_id=str(opt.get("id")),
                    game_title=game_title,
                    scheduled_at=scheduled_at,
                    team_abbr=team_abbr,
                    opponent_abbr=opp_abbr,
                    stat=str(stat),
                    display_stat=str(display_stat) if display_stat else None,
                    line=float(line_value),
                    side=side,  # type: ignore[arg-type]
                    american_price=american,
                    decimal_price=dec,
                    implied_prob=implied,
                    notes=[],
                )
            )

    return out


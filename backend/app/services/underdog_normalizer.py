from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any

from app.models.core import MarketType, Prop, SportId, SubjectKind


# Pattern for splitting MMA / 1v1 sport game titles. Handles:
#   "Fighter A vs Fighter B"
#   "Fighter A vs. Fighter B"
#   "Fighter A VS Fighter B"
#   "Fighter A vs.  Fighter B"  (double space)
_VS_SPLIT_RE = re.compile(r"\s+vs\.?\s+", re.IGNORECASE)


def _canon_name(name: str) -> str:
    """Lower-case, strip diacritics, drop non-alphanumeric. Use ONLY for
    name comparison — preserves the original for display elsewhere.

    Mirrors the canonicalizer in `clients/espn.py` so MMA name-matching
    behaves consistently across modules. Example:
      "Jiří Procházka, Jr." → "jiriprochazkajr"
    """
    if not name:
        return ""
    # NFKD decomposes accents → base char + combining mark; filter combining.
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", stripped.lower())


def _parse_vs_title(title: str) -> tuple[str | None, str | None]:
    """Split an MMA/1v1 game title on the 'vs' separator."""
    parts = _VS_SPLIT_RE.split(title, maxsplit=1)
    if len(parts) != 2:
        return None, None
    a, b = parts[0].strip(), parts[1].strip()
    if not a or not b:
        return None, None
    return a, b


def _matches_fighter(player: dict[str, Any] | None, candidate: str) -> bool:
    """True if `candidate` (a side of the 'A vs B' title) refers to the
    given Underdog player record. Tries, in order:

      1. Full name substring (preserves the legacy fast path).
      2. Last-name word-boundary match (handles 'J. Aldo' / 'C. Sandhagen'
         where the title abbreviates the first name).
      3. Canonicalized substring (strips diacritics, punctuation, case).

    Returns False when player is None or no strategy hits.
    """
    if not player or not candidate:
        return False
    first = (player.get("first_name") or "").strip()
    last = (player.get("last_name") or "").strip()
    full = f"{first} {last}".strip()
    if not full:
        full = (player.get("full_name") or "").strip()
    if not full:
        return False

    cand_lower = candidate.lower()
    if full.lower() in cand_lower:
        return True

    # Last-name word match — survives first-name abbreviations.
    if last:
        last_lower = last.lower()
        if re.search(rf"\b{re.escape(last_lower)}\b", cand_lower):
            return True

    # Canonicalized comparison — survives diacritics and punctuation.
    return _canon_name(full) and _canon_name(full) in _canon_name(candidate)


# Regex heuristics for detecting non-player markets purely from the stat
# label. Underdog uses human-readable strings here ("Total Points",
# "Lakers Total Points", "Run Line", etc.) so a small set of patterns
# covers virtually all flavours.
_GAME_TOTAL_RE = re.compile(
    r"^\s*(game\s+)?(total|points?\s+total|combined|over\/under|o\/u)\s+(points?|runs?|goals?)?\s*$",
    re.IGNORECASE,
)
_TEAM_TOTAL_RE = re.compile(
    r"\bteam\s+(total|points?|runs?|goals?)\b",
    re.IGNORECASE,
)
_SPREAD_RE = re.compile(
    r"\b(spread|run\s+line|puck\s+line|handicap|line)\b",
    re.IGNORECASE,
)
_MONEYLINE_RE = re.compile(
    r"\bmoneyline\b|\bwin(ner)?\b|\bto\s+win\b",
    re.IGNORECASE,
)


def _classify_market(stat: str, has_player: bool) -> MarketType:
    """Heuristic classifier — falls back to player_prop when nothing matches.

    `has_player` is the strongest signal: if the appearance dereferences to
    a real player record, this is virtually always a player prop, even if
    the stat string contains words like "total".
    """
    s = (stat or "").strip()
    if has_player:
        return "player_prop"
    if _MONEYLINE_RE.search(s):
        return "moneyline"
    if _SPREAD_RE.search(s):
        return "spread"
    if _TEAM_TOTAL_RE.search(s):
        return "team_total"
    if _GAME_TOTAL_RE.search(s):
        return "game_total"
    if "total" in s.lower():
        # Generic "Total <stat>" without team / game qualifier defaults to game total.
        return "game_total"
    return "player_prop"


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


# Underdog Pick'em per-leg break-even by entry size (research-backed):
#   2-pick: -136 (~57.7%), 3-pick: -122 (~55.0%), 4-pick: -128 (~56.2%), 5-pick: -122 (~54.9%)
# Default to 3-pick economics — most common entry size, also one of the best.
UD_DEFAULT_LEG_DECIMAL = 1.82  # corresponds to ~55% break-even (-122 American)


def decimal_to_implied_prob(decimal_price: float) -> float:
    if decimal_price <= 1.0:
        return 1.0
    return 1.0 / decimal_price


def decimal_to_american(decimal_price: float) -> int | None:
    if decimal_price <= 1.0:
        return None
    if decimal_price >= 2.0:
        return int(round((decimal_price - 1.0) * 100.0))
    return int(round(-100.0 / (decimal_price - 1.0)))


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
    # Some payload shapes also include a flat `teams` collection and/or a
    # `team_appearances` collection used by team-level markets. We index
    # both so we can dereference team/game-level appearances cleanly.
    teams = {t.get("id"): t for t in (payload.get("teams") or []) if isinstance(t, dict)}
    team_appearances = {
        a.get("id"): a
        for a in (
            (payload.get("team_appearances") or [])
            + (payload.get("game_appearances") or [])
        )
        if isinstance(a, dict)
    }

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
        team: dict[str, Any] | None = None  # for team-level markets
        game_title: str | None = None
        scheduled_at: datetime | None = None
        team_abbr: str | None = None
        opp_abbr: str | None = None

        # Resolve the appearance — it may live in either `appearances`
        # (player-keyed) or `team_appearances` (team-keyed). Some payloads
        # use a single appearances list with mixed kinds.
        app: dict[str, Any] | None = None
        if appearance_id:
            if appearance_id in appearances:
                app = appearances[appearance_id]
            elif appearance_id in team_appearances:
                app = team_appearances[appearance_id]

        if app is not None:
            player_id = app.get("player_id")
            team_id = app.get("team_id")
            match_id = app.get("match_id")
            if player_id and player_id in players:
                player = players[player_id]
            if not player and team_id and team_id in teams:
                team = teams[team_id]
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
                    # MMA / 1v1 sport: "Fighter A vs. Fighter B" or "Fighter A vs Fighter B".
                    # Robust name-matching: exact substring → last-name word
                    # match → diacritic-stripped substring. Without this, fighters
                    # with accented names (Procházka, Błachowicz), abbreviated
                    # title strings (J. Aldo vs C. Sandhagen), or nicknames in
                    # the title fall through to opp_abbr=None and the UI
                    # renders "? vs ?". We also populate team_abbr with the
                    # fighter's own name so the matchup displays correctly —
                    # MMA has no team concept, but the display layer expects
                    # both sides.
                    a_name, b_name = _parse_vs_title(abbr_title)
                    if a_name and b_name:
                        player_is_a = _matches_fighter(player, a_name)
                        player_is_b = _matches_fighter(player, b_name)
                        if player_is_a:
                            team_abbr, opp_abbr = a_name, b_name
                        elif player_is_b:
                            team_abbr, opp_abbr = b_name, a_name
                        else:
                            # Last resort: when we can't determine which fighter
                            # is the prop subject, use the resolved player name
                            # itself (built later from first/last fields) as
                            # team_abbr and assume the OTHER name is opponent.
                            # This is the right call when the player record is
                            # present but the title uses a different
                            # transliteration. Setting both side names ensures
                            # the UI / AI prompt still shows the matchup.
                            team_abbr = a_name
                            opp_abbr = b_name
                dt_raw = game.get("scheduled_at")
                if isinstance(dt_raw, str):
                    try:
                        scheduled_at = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
                    except ValueError:
                        scheduled_at = None

        # Determine home/away from the title split
        is_home: bool | None = None
        if team_abbr and isinstance(abbr_title, str) and "@" in abbr_title:
            _away, _home = [p.strip() for p in abbr_title.split("@", 1)]
            is_home = (team_abbr == _home)

        player_name: str | None = None
        underdog_player_id: str | None = None
        if player:
            underdog_player_id = player.get("id")
            first = player.get("first_name") or ""
            last = player.get("last_name") or ""
            full = (first + " " + last).strip()
            player_name = full or player.get("full_name")

        # Team-level markets: derive a display "team name" from the team
        # record + the home/away split. Falls back to the abbreviation
        # captured from the game title when needed.
        team_display: str | None = None
        if team:
            team_display = (
                team.get("display_name")
                or team.get("name")
                or team.get("abbr")
                or team.get("abbreviation")
            )
            if not team_abbr:
                team_abbr = team.get("abbr") or team.get("abbreviation")
            # Compute is_home / opp_abbr for team-level markets too.
            if game and isinstance(abbr_title := (game.get("abbreviated_title") or game.get("title")), str) and "@" in abbr_title:
                _away, _home = [p.strip() for p in abbr_title.split("@", 1)]
                if team_abbr:
                    if team_abbr == _home:
                        is_home, opp_abbr = True, _away
                    elif team_abbr == _away:
                        is_home, opp_abbr = False, _home

        # Classify the market — player record present is the strongest
        # signal it's a player_prop. Otherwise inspect the stat label.
        market_type = _classify_market(str(stat), has_player=bool(player))
        subject_kind: SubjectKind = (
            "player" if market_type == "player_prop" else ("team" if team else "game")
        )

        # Build the display name. For team / game markets the existing
        # downstream code paths key off `player_name`, so we put a
        # human-readable subject there and also stash the canonical name
        # in `subject_name`.
        subject_name: str | None = None
        if market_type == "player_prop":
            pass  # player_name already set above
        elif market_type == "team_total":
            subject_name = team_display or team_abbr or game_title
            player_name = subject_name
        elif market_type in ("game_total", "moneyline", "spread"):
            subject_name = game_title or (
                f"{team_display} vs {opp_abbr}" if team_display and opp_abbr else None
            )
            player_name = subject_name or "Game Line"

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
            elif market_type == "moneyline" and choice in ("yes", "win", "team_a", "home", "favorite"):
                # Binary moneyline: encode "win" as the over side so the
                # rest of the pipeline (which assumes binary over/under)
                # keeps working without special cases.
                side = "over"
            elif market_type == "moneyline" and choice in ("no", "loss", "team_b", "away", "underdog"):
                side = "under"
            else:
                continue

            american = _parse_int(opt.get("american_price"))
            dec = _parse_float(opt.get("decimal_price"))
            payout_mult = _parse_float(opt.get("payout_multiplier"))
            subheader = opt.get("selection_subheader")
            subheader_str = subheader.strip() if isinstance(subheader, str) else None
            is_boosted = False
            if subheader_str:
                lower_sh = subheader_str.lower()
                is_boosted = ("boost" in lower_sh) or ("special" in lower_sh) or ("rescue" in lower_sh)

            # Truth hierarchy: payout_multiplier > decimal_price > american_price > Pick'em default.
            # Pick'em legs typically arrive with NO price fields — we fall back to the documented
            # 3-pick break-even (~55%, decimal 1.82). This is much closer to reality than -110.
            if isinstance(payout_mult, (int, float)) and payout_mult > 1.0:
                effective_decimal = float(payout_mult)
            elif isinstance(dec, (int, float)) and dec > 1.0:
                effective_decimal = float(dec)
            elif isinstance(american, int):
                effective_decimal = (
                    1.0 + (100.0 / (-american)) if american < 0 else 1.0 + (american / 100.0)
                )
            else:
                effective_decimal = UD_DEFAULT_LEG_DECIMAL

            implied = decimal_to_implied_prob(effective_decimal)
            breakeven_prob = implied  # for Pick'em legs, break-even == implied
            if american is None:
                american = decimal_to_american(effective_decimal)

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
                    is_home=is_home,
                    market_type=market_type,
                    subject_kind=subject_kind,
                    subject_name=subject_name,
                    stat=str(stat),
                    display_stat=str(display_stat) if display_stat else None,
                    line=float(line_value),
                    side=side,  # type: ignore[arg-type]
                    american_price=american,
                    decimal_price=effective_decimal,
                    payout_multiplier=payout_mult if isinstance(payout_mult, (int, float)) else None,
                    selection_subheader=subheader_str,
                    is_boosted=is_boosted,
                    breakeven_prob=breakeven_prob,
                    implied_prob=implied,
                    notes=[],
                )
            )

    return out


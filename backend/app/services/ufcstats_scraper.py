"""UFCStats.com scraper.

Why scrape? ESPN's MMA API returns fight-total stats only — no per-round
breakdown, no strike-by-target (head/body/leg), no control time. These
are the highest-information features for MMA prop modeling per the
published research (Cage Calculus, Diving Into Data, Combat Press
advanced prop strategy series). UFCStats.com publishes all of this in
HTML; no public API exists.

This module:
  1. Lists completed events from /statistics/events/completed?page=all
  2. For each new event (not already in the DB), parses event-details
  3. For each new fight in that event, parses fight-details
  4. Persists per-round stats to the SQLite tables in ufcstats_db.py

Ethics / politeness:
  - Default 2-second delay between requests.
  - Only personal/research use. Don't redistribute scraped data.
  - Realistic User-Agent header. No headless browser, no JS execution.
  - Resumable: skips events already in the DB.

The scraper is invoked manually via the /admin/ufcstats/scrape endpoint
or via a Render Cron Job (weekly is plenty).

Dependencies: beautifulsoup4 (added to requirements.txt). Lazy-imported
so an environment without bs4 can still run the rest of the app — the
scraper just silently no-ops with a log warning.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import httpx

from app.db import SqliteTTLCache
from app.services.ufcstats_db import (
    upsert_event,
    upsert_fight,
    upsert_fight_round,
    _ensure_tables,
)

log = logging.getLogger(__name__)

UFCSTATS_BASE = "http://ufcstats.com"
EVENTS_URL = f"{UFCSTATS_BASE}/statistics/events/completed?page=all"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; aadiparlay-research/1.0; "
    "+https://github.com/S1kkle/aadiparlayv2)"
)


class _ScraperDisabled(RuntimeError):
    """Raised when BeautifulSoup is not installed — scraper silently
    no-ops in that case to avoid breaking other startup work."""


def _bs4_or_disabled():
    """Lazy-import BeautifulSoup. Returns the BeautifulSoup class or
    raises _ScraperDisabled."""
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]
        return BeautifulSoup
    except ImportError:
        raise _ScraperDisabled("beautifulsoup4 not installed")


def _http_client(*, timeout: float = 20.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout,
        headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/html"},
        follow_redirects=True,
    )


async def _fetch(client: httpx.AsyncClient, url: str, *, retries: int = 2) -> str | None:
    """Single GET with retries. Returns text or None on persistent failure."""
    for attempt in range(retries + 1):
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (429, 503):
                # Rate-limited / temporarily unavailable — back off.
                await asyncio.sleep(min(2 ** attempt * 5, 30))
                continue
            log.warning("UFCStats fetch %s returned %s", url, resp.status_code)
            return None
        except httpx.HTTPError:
            log.warning("UFCStats fetch %s failed (attempt %d)", url, attempt + 1)
            await asyncio.sleep(2 * (attempt + 1))
    return None


def _id_from_url(url: str) -> str | None:
    """UFCStats URLs end in a 16-char hex ID. Extract and return it."""
    m = re.search(r"/(event|fight|fighter)-details/([0-9a-f]+)", url)
    return m.group(2) if m else None


def _parse_int(text: str | None) -> int | None:
    if text is None:
        return None
    s = text.strip()
    if not s or s in ("--", "—"):
        return None
    try:
        return int(s.split()[0].split("/")[0])
    except (ValueError, IndexError):
        return None


def _parse_mmss_to_sec(text: str | None) -> int | None:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s or s in ("--", "—"):
        return None
    if ":" not in s:
        return _parse_int(s)
    try:
        m, sec = s.split(":")
        return int(m) * 60 + int(sec)
    except (ValueError, TypeError):
        return None


def _split_x_of_y(text: str | None) -> tuple[int | None, int | None]:
    """UFCStats reports many stats as 'landed of attempted', e.g. '45 of 102'."""
    if not isinstance(text, str):
        return None, None
    parts = text.strip().split(" of ")
    if len(parts) != 2:
        return _parse_int(text), None
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, IndexError):
        return None, None


async def _parse_events_index(client: httpx.AsyncClient, BeautifulSoup) -> list[dict[str, Any]]:
    """List all completed events. Returns [{id, name, date, location}, ...]
    newest-first.
    """
    html = await _fetch(client, EVENTS_URL)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict[str, Any]] = []
    # Rows live in <tr class="b-statistics__table-row"> with three <td>s.
    for tr in soup.select("tr.b-statistics__table-row"):
        a = tr.find("a", href=re.compile(r"event-details"))
        if not a:
            continue
        href = a.get("href", "")
        eid = _id_from_url(href)
        if not eid:
            continue
        name = a.get_text(strip=True)
        # Date is in a sibling <span> with class b-statistics__date.
        date_span = tr.find("span", class_="b-statistics__date")
        event_date = date_span.get_text(strip=True) if date_span else None
        # Location is the second <td>.
        tds = tr.find_all("td")
        location = tds[1].get_text(strip=True) if len(tds) >= 2 else None
        out.append({
            "id": eid,
            "name": name,
            "event_date": event_date,
            "location": location,
            "url": href,
        })
    return out


async def _parse_event_detail(
    client: httpx.AsyncClient, BeautifulSoup, event_url: str
) -> list[dict[str, Any]]:
    """Parse a single event-details page; return list of fight stubs:
    [{id, fighter_a, fighter_b, weight_class, method, round, time}, ...]
    """
    html = await _fetch(client, event_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict[str, Any]] = []
    for tr in soup.select("tr.b-fight-details__table-row__hover"):
        href = tr.get("data-link") or ""
        fid = _id_from_url(href)
        if not fid:
            a = tr.find("a", href=re.compile(r"fight-details"))
            if a:
                href = a.get("href") or ""
                fid = _id_from_url(href)
        if not fid:
            continue
        tds = tr.find_all("td")
        if len(tds) < 7:
            continue
        # Fighter names live in the second <td>'s two <p> elements.
        name_ps = tds[1].find_all("p")
        if len(name_ps) < 2:
            continue
        fighter_a = name_ps[0].get_text(strip=True)
        fighter_b = name_ps[1].get_text(strip=True)
        # Weight class in the 6th td.
        wc_p = tds[6].find("p") if len(tds) >= 7 else None
        weight_class = wc_p.get_text(strip=True) if wc_p else None
        # Method, round, time in tds 7, 8, 9.
        method = tds[7].get_text(strip=True) if len(tds) >= 8 else None
        round_p = tds[8].get_text(strip=True) if len(tds) >= 9 else None
        time_finished = tds[9].get_text(strip=True) if len(tds) >= 10 else None
        out.append({
            "id": fid,
            "url": href,
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "weight_class": weight_class,
            "method": method,
            "round_finished": _parse_int(round_p),
            "time_finished": time_finished,
        })
    return out


async def _parse_fight_detail(
    client: httpx.AsyncClient, BeautifulSoup, fight_url: str,
) -> dict[str, Any] | None:
    """Parse the totals + per-round tables on a fight-details page.

    Returns a dict with:
      fighter_a, fighter_b, winner, scheduled_rounds,
      per_round: list of {fighter, round_num, sig_strikes_landed, ...}
    """
    html = await _fetch(client, fight_url)
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")

    # Header has fighter names + winner indicator
    fighter_names: list[str] = []
    win_flags: list[bool] = []
    for div in soup.select("div.b-fight-details__person"):
        name_a = div.find("a", class_="b-link")
        if name_a:
            fighter_names.append(name_a.get_text(strip=True))
        status = div.find("i", class_="b-fight-details__person-status")
        win_flags.append(bool(status and "_green" in (status.get("class") or [])))
    if len(fighter_names) < 2:
        return None

    winner = None
    for name, won in zip(fighter_names, win_flags):
        if won:
            winner = name
            break

    # Per-round stats live in two tables:
    #   "Significant Strikes" totals + per-round breakdown
    # We parse the per-round tables: each section has class
    # "b-fight-details__table" and is preceded by a <thead> labeled
    # "Per round" / "Significant Strikes (per round)".
    per_round: list[dict[str, Any]] = []

    tables = soup.select("table.b-fight-details__table")
    # The general totals table is the FIRST, followed by per-round.
    # We want the per-round breakdown for both totals + sig strikes.
    # Per-round tables have class containing "b-fight-details__table_style"
    # plus the parent section's <p> with text "Per round".
    # Easier: identify tables by checking if they have multiple <tr> rows
    # with round numbering, OR look at preceding section text.

    # Strategy: walk through all section <p> headers; when we hit "Per round",
    # take the next table.
    section_round_tables: list[Any] = []
    for section in soup.select("section.b-fight-details__section"):
        h = section.find("p", class_="b-fight-details__collapse-link")
        if h and "per round" in h.get_text(strip=True).lower():
            tbl = section.find_next("table", class_="b-fight-details__table")
            if tbl is not None:
                section_round_tables.append(tbl)
    # Fallback: tables with multiple thead rounds.
    if not section_round_tables:
        section_round_tables = list(tables[1:3])  # heuristic

    # We expect 2 per-round tables:
    #   Index 0: Totals per round (TD, KD, sub_att, control_time, sig+total str)
    #   Index 1: Significant Strikes per round (by target / position)
    totals_table = section_round_tables[0] if section_round_tables else None
    strikes_table = section_round_tables[1] if len(section_round_tables) >= 2 else None

    def _parse_two_p_cells(td) -> tuple[str | None, str | None]:
        """Many UFCStats cells contain two <p> children (one per fighter)."""
        ps = td.find_all("p", recursive=True)
        if len(ps) >= 2:
            return ps[0].get_text(strip=True), ps[1].get_text(strip=True)
        if len(ps) == 1:
            return ps[0].get_text(strip=True), None
        return None, None

    def _parse_round_section(thead_text: str) -> int | None:
        m = re.search(r"Round\s+(\d+)", thead_text)
        return int(m.group(1)) if m else None

    if totals_table is not None:
        current_round: int | None = None
        for el in totals_table.find_all(["thead", "tbody"]):
            if el.name == "thead":
                # Round-marker thead: "Round N"
                text = el.get_text(" ", strip=True)
                rnd = _parse_round_section(text)
                if rnd is not None:
                    current_round = rnd
            elif el.name == "tbody" and current_round is not None:
                tr = el.find("tr")
                if tr is None:
                    continue
                tds = tr.find_all("td")
                # Columns (UFCStats totals-per-round):
                # 0=Fighter, 1=KD, 2=Sig.Str, 3=Sig.Str.%, 4=Total Str,
                # 5=TD, 6=TD.%, 7=Sub.Att, 8=Pass, 9=Ctrl
                if len(tds) < 10:
                    continue
                name_p0, name_p1 = _parse_two_p_cells(tds[0])
                kd_a, kd_b = _parse_two_p_cells(tds[1])
                sigstr_a, sigstr_b = _parse_two_p_cells(tds[2])
                totalstr_a, totalstr_b = _parse_two_p_cells(tds[4])
                td_a, td_b = _parse_two_p_cells(tds[5])
                sub_a, sub_b = _parse_two_p_cells(tds[7])
                ctrl_a, ctrl_b = _parse_two_p_cells(tds[9])
                for nm, kd, sigstr, totalstr, tdv, subv, ctrl in [
                    (name_p0, kd_a, sigstr_a, totalstr_a, td_a, sub_a, ctrl_a),
                    (name_p1, kd_b, sigstr_b, totalstr_b, td_b, sub_b, ctrl_b),
                ]:
                    if not nm:
                        continue
                    sig_land, sig_att = _split_x_of_y(sigstr)
                    total_land, _ = _split_x_of_y(totalstr)
                    td_land, td_att = _split_x_of_y(tdv)
                    per_round.append({
                        "fighter": nm,
                        "round_num": current_round,
                        "sig_strikes_landed": sig_land,
                        "sig_strikes_attempted": sig_att,
                        "total_strikes_landed": total_land,
                        "takedowns_landed": td_land,
                        "takedowns_attempted": td_att,
                        "sub_attempts": _parse_int(subv),
                        "knockdowns": _parse_int(kd),
                        "control_time_sec": _parse_mmss_to_sec(ctrl),
                    })

    # Merge significant-strikes per-round (head/body/leg) into per_round rows.
    if strikes_table is not None:
        by_key = {(r["fighter"], r["round_num"]): r for r in per_round}
        current_round = None
        for el in strikes_table.find_all(["thead", "tbody"]):
            if el.name == "thead":
                text = el.get_text(" ", strip=True)
                rnd = _parse_round_section(text)
                if rnd is not None:
                    current_round = rnd
            elif el.name == "tbody" and current_round is not None:
                tr = el.find("tr")
                if tr is None:
                    continue
                tds = tr.find_all("td")
                # Columns:
                # 0=Fighter, 1=Sig.Str, 2=Sig.Str.%, 3=Head, 4=Body, 5=Leg,
                # 6=Distance, 7=Clinch, 8=Ground
                if len(tds) < 6:
                    continue
                name_p0, name_p1 = _parse_two_p_cells(tds[0])
                head_a, head_b = _parse_two_p_cells(tds[3])
                body_a, body_b = _parse_two_p_cells(tds[4])
                leg_a, leg_b = _parse_two_p_cells(tds[5])
                for nm, h, b, lg in [
                    (name_p0, head_a, body_a, leg_a),
                    (name_p1, head_b, body_b, leg_b),
                ]:
                    if not nm:
                        continue
                    entry = by_key.get((nm, current_round))
                    if entry is None:
                        continue
                    h_land, _ = _split_x_of_y(h)
                    b_land, _ = _split_x_of_y(b)
                    l_land, _ = _split_x_of_y(lg)
                    entry["head_strikes"] = h_land
                    entry["body_strikes"] = b_land
                    entry["leg_strikes"] = l_land

    # Scheduled rounds — title fights are 5, regular non-title are 3. We
    # infer from the highest round we saw with stats; cap at 5.
    scheduled = max((r["round_num"] for r in per_round), default=3)

    return {
        "fighter_a": fighter_names[0],
        "fighter_b": fighter_names[1] if len(fighter_names) > 1 else None,
        "winner": winner,
        "scheduled_rounds": min(5, max(3, scheduled)),
        "per_round": per_round,
    }


async def scrape_ufcstats(
    cache: SqliteTTLCache,
    *,
    max_events: int = 20,
    delay_seconds: float = 2.0,
) -> dict[str, Any]:
    """Top-level scrape entry point. Walks the events index newest-first
    and scrapes any not already in the DB, up to `max_events`.

    Returns a structured dict for logging / API:
      {events_scraped, fights_scraped, rounds_scraped, skipped_existing}
    """
    try:
        BeautifulSoup = _bs4_or_disabled()
    except _ScraperDisabled:
        return {"status": "disabled", "reason": "beautifulsoup4 not installed"}

    _ensure_tables(cache)
    with cache._connection() as conn:
        existing = {
            r["id"] for r in conn.execute("SELECT id FROM ufcstats_events").fetchall()
        }

    events_added = 0
    fights_added = 0
    rounds_added = 0
    skipped = 0

    async with _http_client() as client:
        index = await _parse_events_index(client, BeautifulSoup)
        log.info("UFCStats index returned %d events; %d already in DB", len(index), len(existing))
        for evt in index:
            if events_added >= max_events:
                break
            if evt["id"] in existing:
                skipped += 1
                continue
            await asyncio.sleep(delay_seconds)
            fights = await _parse_event_detail(client, BeautifulSoup, evt["url"])
            if not fights:
                continue
            upsert_event(
                cache,
                event_id=evt["id"], name=evt["name"],
                event_date=evt["event_date"], location=evt.get("location"),
            )
            events_added += 1
            for fight in fights:
                await asyncio.sleep(delay_seconds)
                detail = await _parse_fight_detail(client, BeautifulSoup, fight["url"])
                if detail is None:
                    continue
                upsert_fight(
                    cache,
                    fight_id=fight["id"], event_id=evt["id"],
                    event_date=evt["event_date"],
                    weight_class=fight.get("weight_class"),
                    method=fight.get("method"),
                    round_finished=fight.get("round_finished"),
                    time_finished=fight.get("time_finished"),
                    scheduled_rounds=detail.get("scheduled_rounds"),
                    fighter_a=detail["fighter_a"],
                    fighter_b=detail.get("fighter_b") or fight.get("fighter_b") or "",
                    winner=detail.get("winner"),
                )
                fights_added += 1
                for r in detail.get("per_round", []):
                    upsert_fight_round(cache, fight_id=fight["id"], **r)
                    rounds_added += 1

    return {
        "status": "ok",
        "events_scraped": events_added,
        "fights_scraped": fights_added,
        "rounds_scraped": rounds_added,
        "skipped_existing": skipped,
    }

"use client";

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  fetchHistory,
  fetchLearningEntries,
  fetchLearningLog,
  fetchLearningReports,
  fetchPropsJobResult,
  fetchRankedProps,
  getBackendUrl,
  learningAnalyzeMisses,
  learningResolve,
  learningWeeklyReport,
  saveHistory,
  seedHistory,
  startPropsJob,
} from "@/lib/api";
import type { HistoryEntry, LearningEntry, LearningLogEntry, LearningLogKind, LearningReport, Prop, RankedPropsResponse, SportId } from "@/lib/types";
import {
  type EntryType,
  availableEntryTypes,
  underdogStandardPayout,
} from "@/lib/underdog";
import {
  allEntryEvs,
  bestEntryType,
  entryEv,
  kellyFullFraction,
  parlayCorrelationFactor,
} from "@/lib/parlay-math";
import { ReliabilityDiagram } from "@/components/ReliabilityDiagram";
import { BankrollGrowth } from "@/components/BankrollGrowth";

// ── localStorage history persistence ──────────────────────────────────
const LS_HISTORY_KEY = "parlay_prediction_history";
const LS_BANKROLL_KEY = "parlay_bankroll";
const LS_KELLY_DIVISOR_KEY = "parlay_kelly_divisor";
const LS_ENTRY_TYPE_KEY = "parlay_entry_type";
const LS_LOCKED_IDS_KEY = "parlay_locked_ids";

function lsGetHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(LS_HISTORY_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function lsSaveHistoryEntry(entry: HistoryEntry) {
  try {
    const existing = lsGetHistory();
    const merged = [entry, ...existing.filter((e) => e.id !== entry.id)].slice(0, 100);
    localStorage.setItem(LS_HISTORY_KEY, JSON.stringify(merged));
  } catch {}
}

// Underdog payout tables and Kelly math are now in @/lib/underdog and
// @/lib/parlay-math (shared with backend service contract).

function fmtMoney(x: number | null | undefined) {
  if (x === null || x === undefined || !isFinite(x)) return "—";
  return `$${x.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function fmtAmericanFromDecimal(decimal: number): string {
  if (!isFinite(decimal) || decimal <= 1) return "—";
  if (decimal >= 2) return `+${Math.round((decimal - 1) * 100)}`;
  return `${Math.round(-100 / (decimal - 1))}`;
}

const SPORT_OPTIONS: { id: SportId; label: string }[] = [
  { id: "UNKNOWN", label: "All sports" },
  { id: "NBA", label: "NBA" },
  { id: "NFL", label: "NFL" },
  { id: "NHL", label: "NHL" },
  { id: "SOCCER", label: "Soccer" },
  { id: "MMA", label: "MMA" },
];

type SortKey =
  | "score"
  | "edge"
  | "ev"
  | "volatility"
  | "model_prob"
  | "implied_prob"
  | "hit_rate";

function fmtPct(x: number | null | undefined) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}

function fmtNum(x: number | null | undefined, digits = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}

/** Compact "5 min ago" style relative timestamp for the learning log. */
function fmtTimeAgo(iso: string | undefined | null): string {
  if (!iso) return "—";
  const t = new Date(iso).getTime();
  if (!isFinite(t)) return "—";
  const sec = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (sec < 45) return "just now";
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} min ago`;
  const hr = Math.floor(min / 60);
  if (hr < 36) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day < 14) return `${day}d ago`;
  const wk = Math.floor(day / 7);
  if (wk < 9) return `${wk}w ago`;
  const mo = Math.floor(day / 30);
  return `${mo}mo ago`;
}

/** Pretty pick description that adapts to market type so game-line legs
 * read naturally instead of "OVER 0 Moneyline". */
function fmtPick(p: Prop): string {
  const mt = p.market_type;
  if (mt === "moneyline") return "Win"; // player_name already encodes the team
  if (mt === "spread") {
    const sign = p.line >= 0 ? `+${p.line}` : `${p.line}`;
    return `Spread ${sign}`;
  }
  if (mt === "game_total" || mt === "team_total") {
    return `${(p.side || "").toUpperCase()} ${p.line}`;
  }
  return `${(p.side || "").toUpperCase()} ${p.line} ${p.display_stat ?? p.stat}`;
}

function fmtVal(x: unknown) {
  if (typeof x !== "number" || Number.isNaN(x)) return "—";
  return Number.isInteger(x) ? String(x) : x.toFixed(1);
}

function biasLabel(bias: number | null | undefined) {
  if (bias === 1) return "Favors pick";
  if (bias === -1) return "Against pick";
  if (bias === 0) return "Neutral";
  return "—";
}

function shortText(s: string | null | undefined, maxLen = 120) {
  if (!s) return null;
  const t = s.trim();
  if (t.length <= maxLen) return t;
  return t.slice(0, maxLen - 1).trimEnd() + "…";
}

function formatDate(d: string | null | undefined) {
  if (!d) return "—";
  const dt = new Date(d);
  if (Number.isNaN(dt.getTime())) return "—";
  return dt.toLocaleDateString(undefined, { month: "2-digit", day: "2-digit" });
}

function hitResult(value: unknown, line: number, side: "over" | "under") {
  if (typeof value !== "number" || Number.isNaN(value)) return null;
  if (side === "over") return value > line;
  return value < line;
}

function edgeColor(edge: number | null | undefined) {
  if (edge === null || edge === undefined) return "";
  if (edge > 0.03) return "text-emerald-600 dark:text-emerald-400";
  if (edge < -0.03) return "text-rose-600 dark:text-rose-400";
  return "";
}

function biasColor(bias: number | null | undefined) {
  if (bias === 1) return "text-emerald-600 dark:text-emerald-400";
  if (bias === -1) return "text-rose-600 dark:text-rose-400";
  return "text-zinc-500";
}

function tierBadge(tier: string | null | undefined) {
  if (tier === "high")
    return (
      <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300">
        HIGH
      </span>
    );
  if (tier === "medium")
    return (
      <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-semibold text-amber-800 dark:bg-amber-900/40 dark:text-amber-300">
        MED
      </span>
    );
  if (tier === "low")
    return (
      <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-[10px] font-semibold text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
        LOW
      </span>
    );
  return null;
}

function trendArrow(dir: string | null | undefined) {
  if (dir === "up") return <span className="text-emerald-500" title="Trending up">▲</span>;
  if (dir === "down") return <span className="text-rose-500" title="Trending down">▼</span>;
  if (dir === "flat") return <span className="text-zinc-400" title="Flat">—</span>;
  return null;
}

function sortProps(props: Prop[], key: SortKey, asc: boolean): Prop[] {
  const sorted = [...props].sort((a, b) => {
    let va: number | null = null;
    let vb: number | null = null;
    switch (key) {
      case "score":
        va = a.score ?? null;
        vb = b.score ?? null;
        break;
      case "edge":
        va = a.edge ?? null;
        vb = b.edge ?? null;
        break;
      case "ev":
        va = a.ev ?? null;
        vb = b.ev ?? null;
        break;
      case "volatility":
        va = a.volatility ?? null;
        vb = b.volatility ?? null;
        break;
      case "model_prob":
        va = a.model_prob ?? null;
        vb = b.model_prob ?? null;
        break;
      case "implied_prob":
        va = a.implied_prob ?? null;
        vb = b.implied_prob ?? null;
        break;
      case "hit_rate":
        va = a.hit_rate_last10 ?? null;
        vb = b.hit_rate_last10 ?? null;
        break;
    }
    if (va === null && vb === null) return 0;
    if (va === null) return 1;
    if (vb === null) return -1;
    return asc ? va - vb : vb - va;
  });
  return sorted;
}

function getUniqueStats(props: Prop[]): string[] {
  const s = new Set<string>();
  for (const p of props) s.add(p.stat);
  return Array.from(s).sort();
}

export default function Home() {
  const backendUrl = useMemo(() => getBackendUrl(), []);
  const [sport, setSport] = useState<SportId>("NBA");
  const [loading, setLoading] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<RankedPropsResponse | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [jobProgress, setJobProgress] = useState<{
    stage: string;
    detail: string;
    ai_succeeded: number;
    ai_attempted: number;
    ai_target: number;
    analyzed: number;
  } | null>(null);
  const esRef = useRef<EventSource | null>(null);

  // Sorting
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [sortAsc, setSortAsc] = useState(false);

  // Stat filter
  const [statFilter, setStatFilter] = useState<string>("all");

  // Market filter — separates player props from game-level lines (game/team
  // total, spread, moneyline). All / props / game-lines.
  const [marketFilter, setMarketFilter] = useState<"all" | "player_prop" | "game_line">("all");

  // Parlay builder
  const [parlayIds, setParlayIds] = useState<Set<string>>(new Set());

  // Bankroll & Kelly sizing (persisted in localStorage)
  // Default to quarter Kelly per the literature: ~44% of full-Kelly growth
  // for ~6% of the variance — the sweet spot that elite bettors use.
  const [bankroll, setBankroll] = useState<number>(1000);
  const [bankrollInput, setBankrollInput] = useState<string>("1000");
  const [kellyDivisor, setKellyDivisor] = useState<number>(4);

  // User-supplied parlay payout override (e.g. "13" when Underdog's slip
  // shows 13x). Defaults to null → use the Underdog standard payout table.
  const [parlayPayoutOverride, setParlayPayoutOverride] = useState<number | null>(null);
  const [parlayPayoutInput, setParlayPayoutInput] = useState<string>("");

  // Underdog entry-type selector (Standard, Insurance, Flex). Persisted.
  const [entryType, setEntryType] = useState<EntryType>("standard");

  // "Locked" picks — protected from accidental removal (the 'X' button on a
  // parlay leg is hidden for locked picks). Stored as a Set of underdog_option_id.
  const [lockedIds, setLockedIds] = useState<Set<string>>(new Set());

  // Model-only props (shown before AI finishes)
  const [modelProps, setModelProps] = useState<Prop[]>([]);

  // History
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  // Learning Mode (now: unified time-ordered log of model events)
  const [showLearning, setShowLearning] = useState(false);
  const [learningLoading, setLearningLoading] = useState(false);
  const [learningEntries, setLearningEntries] = useState<LearningEntry[]>([]);
  const [learningReports, setLearningReports] = useState<LearningReport[]>([]);
  const [learningError, setLearningError] = useState<string | null>(null);
  const [learningStatus, setLearningStatus] = useState<string | null>(null);
  const [learningLog, setLearningLog] = useState<LearningLogEntry[]>([]);
  const [learningLogTotals, setLearningLogTotals] = useState<Partial<Record<LearningLogKind, number>>>({});
  const [logFilter, setLogFilter] = useState<"all" | LearningLogKind>("all");
  const [expandedLogIds, setExpandedLogIds] = useState<Set<string>>(new Set());

  const toggleParlay = useCallback((id: string) => {
    setParlayIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const toggleLock = useCallback((id: string) => {
    setLockedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  async function load(refresh = false) {
    setLoading(true);
    setError(null);
    setJobProgress(null);
    setModelProps([]);
    if (esRef.current) {
      try { esRef.current.close(); } catch {}
      esRef.current = null;
    }
    try {
      setData(null);
      const { job_id } = await startPropsJob({
        sport,
        scope: "all",
        refresh,
        maxProps: 200,
        aiLimit: 10,
        requireAiCount: 15,
      });

      const es = new EventSource(
        `${backendUrl.replace(/\/$/, "")}/props/job/${job_id}/events`
      );
      esRef.current = es;

      es.onmessage = async (msg) => {
        try {
          const ev = JSON.parse(msg.data) as any;
          if (ev?.type === "progress") {
            setJobProgress({
              stage: String(ev.stage ?? "ai"),
              detail: String(ev.detail ?? ""),
              ai_succeeded: Number(ev.ai_succeeded ?? 0),
              ai_attempted: Number(ev.ai_attempted ?? 0),
              ai_target: Number(ev.ai_target ?? 10),
              analyzed: Number(ev.analyzed ?? 0),
            });
          } else if (ev?.type === "model_done") {
            const props = (ev.props ?? []) as Prop[];
            setModelProps(props);
          } else if (ev?.type === "ai_update") {
            const optId = ev.option_id as string;
            const aiData = ev.ai as Record<string, unknown>;
            if (optId && aiData) {
              setModelProps((prev) =>
                prev.map((p) =>
                  p.underdog_option_id === optId
                    ? {
                        ...p,
                        ai_summary: (aiData.ai_summary as string) ?? p.ai_summary,
                        ai_bias: (aiData.ai_bias as number) ?? p.ai_bias,
                        ai_confidence: (aiData.ai_confidence as number) ?? p.ai_confidence,
                        ai_tailwinds: (aiData.ai_tailwinds as string[]) ?? p.ai_tailwinds,
                        ai_risk_factors: (aiData.ai_risk_factors as string[]) ?? p.ai_risk_factors,
                        ai_prob_adjustment: (aiData.ai_prob_adjustment as number) ?? p.ai_prob_adjustment,
                      }
                    : p
                )
              );
            }
          } else if (ev?.type === "error") {
            setError(String(ev.error ?? "Job failed"));
            setLoading(false);
            try { es.close(); } catch {}
            esRef.current = null;
          } else if (ev?.type === "done") {
            try { es.close(); } catch {}
            esRef.current = null;
            const res = await fetchPropsJobResult(job_id);
            setData(res);
            setModelProps([]);
            setLoading(false);
            setJobProgress(null);
            // Save to localStorage (survives browser close + Render restarts)
            const historyProps = res.props.slice(0, 15);
            const historyEntry: HistoryEntry = {
              id: `local-${Date.now()}`,
              timestamp: new Date().toISOString(),
              sport,
              props: historyProps as any,
            };
            lsSaveHistoryEntry(historyEntry);
            // Also save to backend (best-effort)
            try {
              await saveHistory({ sport, props: historyProps });
            } catch {}
          }
        } catch {}
      };

      es.onerror = () => {
        setError("Lost connection to backend progress stream.");
        setLoading(false);
        try { es.close(); } catch {}
        esRef.current = null;
      };
    } catch (e) {
      setData(null);
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function clearCacheAndReload() {
    setClearing(true);
    setError(null);
    try {
      await fetch(`${backendUrl.replace(/\/$/, "")}/cache/clear`, { method: "POST" });
      setExpanded({});
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setClearing(false);
    }
    void load(true);
  }

  async function loadHistory() {
    const local = lsGetHistory();
    let remote: HistoryEntry[] = [];
    try {
      remote = await fetchHistory();
    } catch {}
    // Merge: combine both, deduplicate by id, sort newest first
    const byId = new Map<string, HistoryEntry>();
    for (const e of [...remote, ...local]) byId.set(e.id, e);
    const merged = [...byId.values()].sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
    setHistory(merged);
    setShowHistory(true);
  }

  async function openLearning() {
    setShowLearning(true);
    setLearningError(null);
    try {
      const [entries, reports, log] = await Promise.all([
        fetchLearningEntries({ limit: 200 }),
        fetchLearningReports(5),
        fetchLearningLog({ limit: 80 }),
      ]);
      setLearningEntries(entries);
      setLearningReports(reports);
      setLearningLog(log.entries || []);
      setLearningLogTotals(log.totals || {});
    } catch (e: any) {
      setLearningError(e?.message ?? "Failed to load learning data");
    }
  }

  async function refreshLearningLog() {
    try {
      const log = await fetchLearningLog({ limit: 80 });
      setLearningLog(log.entries || []);
      setLearningLogTotals(log.totals || {});
    } catch {
      // non-fatal — feed will simply show stale data
    }
  }

  async function runLearningPipeline() {
    setLearningLoading(true);
    setLearningError(null);
    const summaryParts: string[] = [];

    try {
      // Step 0: Re-seed backend with localStorage history (survives Render restarts)
      setLearningStatus("Syncing prediction history to backend...");
      const localHistory = lsGetHistory();
      if (localHistory.length > 0) {
        try {
          await seedHistory(localHistory);
        } catch {}
      }

      // Step 1: Resolve outcomes
      setLearningStatus("Step 1/3 — Resolving outcomes from ESPN game logs...");
      const r = await learningResolve();
      const resolveParts: string[] = [];
      if (r.resolved > 0) resolveParts.push(`${r.resolved} new picks resolved`);
      if (r.already_done > 0) resolveParts.push(`${r.already_done} already done`);
      if (r.failed_lookup > 0) resolveParts.push(`${r.failed_lookup} couldn't look up`);
      if (r.skipped_future > 0) resolveParts.push(`${r.skipped_future} games not finished yet`);
      const resolveMsg = resolveParts.length > 0 ? resolveParts.join(", ") : "no new picks to resolve";
      summaryParts.push(`Resolve: ${resolveMsg}`);

      // Refresh entries after resolve so we see hits/misses
      const entriesAfterResolve = await fetchLearningEntries({ limit: 200 });
      setLearningEntries(entriesAfterResolve);

      const missCount = entriesAfterResolve.filter(e => e.hit === 0 && e.resolved === 0).length;

      // Step 2: Analyze misses
      if (missCount > 0) {
        setLearningStatus(`Step 2/3 — AI analyzing ${missCount} missed pick${missCount === 1 ? "" : "s"}...`);
      } else {
        setLearningStatus("Step 2/3 — Checking for unanalyzed misses...");
      }
      const a = await learningAnalyzeMisses();
      if (a.analyzed > 0) {
        summaryParts.push(`Misses: ${a.analyzed} analyzed by AI`);
      } else {
        summaryParts.push(`Misses: ${a.message || "none to analyze"}`);
      }

      // Refresh entries after miss analysis
      const entriesAfterAnalysis = await fetchLearningEntries({ limit: 200 });
      setLearningEntries(entriesAfterAnalysis);

      // Step 3: Weekly report
      setLearningStatus("Step 3/3 — Generating weekly improvement report...");
      await learningWeeklyReport();
      summaryParts.push("Report: generated");

      // Final refresh
      const [finalEntries, reports, log] = await Promise.all([
        fetchLearningEntries({ limit: 200 }),
        fetchLearningReports(5),
        fetchLearningLog({ limit: 80 }),
      ]);
      setLearningEntries(finalEntries);
      setLearningReports(reports);
      setLearningLog(log.entries || []);
      setLearningLogTotals(log.totals || {});

      setLearningStatus(summaryParts.join(" · "));
    } catch (e: any) {
      setLearningError(e?.message ?? "Failed to run learning pipeline");
      setLearningStatus(summaryParts.length > 0 ? summaryParts.join(" · ") + " (stopped due to error)" : null);
    } finally {
      setLearningLoading(false);
    }
  }

  useEffect(() => {
    void load(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sport]);

  // Hydrate bankroll / Kelly fraction from localStorage on mount
  useEffect(() => {
    try {
      const rawBank = localStorage.getItem(LS_BANKROLL_KEY);
      if (rawBank !== null) {
        const v = Number(rawBank);
        if (isFinite(v) && v > 0) {
          setBankroll(v);
          setBankrollInput(String(v));
        }
      }
      const rawDiv = localStorage.getItem(LS_KELLY_DIVISOR_KEY);
      if (rawDiv !== null) {
        const v = Number(rawDiv);
        if (isFinite(v) && [1, 2, 4, 8].includes(v)) setKellyDivisor(v);
      }
      const rawEt = localStorage.getItem(LS_ENTRY_TYPE_KEY);
      if (rawEt === "standard" || rawEt === "insurance" || rawEt === "flex") {
        setEntryType(rawEt);
      }
      const rawLocked = localStorage.getItem(LS_LOCKED_IDS_KEY);
      if (rawLocked) {
        try {
          const arr = JSON.parse(rawLocked);
          if (Array.isArray(arr)) setLockedIds(new Set(arr.filter((x): x is string => typeof x === "string")));
        } catch {}
      }
    } catch {}
  }, []);

  useEffect(() => {
    try { localStorage.setItem(LS_BANKROLL_KEY, String(bankroll)); } catch {}
  }, [bankroll]);

  useEffect(() => {
    try { localStorage.setItem(LS_KELLY_DIVISOR_KEY, String(kellyDivisor)); } catch {}
  }, [kellyDivisor]);

  useEffect(() => {
    try { localStorage.setItem(LS_ENTRY_TYPE_KEY, entryType); } catch {}
  }, [entryType]);

  useEffect(() => {
    try { localStorage.setItem(LS_LOCKED_IDS_KEY, JSON.stringify(Array.from(lockedIds))); } catch {}
  }, [lockedIds]);

  function commitBankroll(v: string) {
    const n = Number(v.replace(/[^0-9.]/g, ""));
    if (isFinite(n) && n >= 0) {
      setBankroll(n);
      setBankrollInput(String(n));
    } else {
      setBankrollInput(String(bankroll));
    }
  }

  // Reset the manual payout override whenever the leg count changes — a
  // different parlay means a different actual Underdog multiplier.
  useEffect(() => {
    setParlayPayoutOverride(null);
    setParlayPayoutInput("");
  }, [parlayIds.size]);

  const allProps = data?.props ?? [];
  const displayProps = allProps.length ? allProps : modelProps;
  const aiFinished = !!data;
  const availableStats = useMemo(() => getUniqueStats(displayProps), [displayProps]);

  const hasAi = (p: Prop) => typeof p.ai_summary === "string" && p.ai_summary.trim().length > 0;

  // Helper: detect game-line props. The backend marks them with
  // market_type != "player_prop"; we tolerate older props missing the
  // field by assuming they're player_props.
  const isGameLine = (p: Prop) =>
    p.market_type !== undefined && p.market_type !== null && p.market_type !== "player_prop";

  const topProps = useMemo(() => {
    let list = displayProps;
    if (marketFilter === "player_prop") list = list.filter((p) => !isGameLine(p));
    else if (marketFilter === "game_line") list = list.filter((p) => isGameLine(p));
    if (statFilter !== "all") list = list.filter((p) => p.stat === statFilter);
    const withAiAgree = sortProps(
      list.filter((p) => hasAi(p) && p.model_ai_agree === true),
      sortKey,
      sortAsc,
    );
    const seenPlayers = new Set<string>();
    const deduped: Prop[] = [];
    for (const p of withAiAgree) {
      if (seenPlayers.has(p.player_name)) continue;
      seenPlayers.add(p.player_name);
      deduped.push(p);
      if (deduped.length >= 10) break;
    }
    return deduped;
  }, [displayProps, statFilter, marketFilter, sortKey, sortAsc]);

  const remainingProps = useMemo(() => {
    let list = displayProps;
    if (marketFilter === "player_prop") list = list.filter((p) => !isGameLine(p));
    else if (marketFilter === "game_line") list = list.filter((p) => isGameLine(p));
    if (statFilter !== "all") list = list.filter((p) => p.stat === statFilter);
    const topIds = new Set(topProps.map((p) => p.underdog_option_id));
    return sortProps(list.filter((p) => !topIds.has(p.underdog_option_id)), sortKey, sortAsc);
  }, [displayProps, statFilter, marketFilter, sortKey, sortAsc, topProps]);

  const filteredProps = topProps;

  // Parlay computations
  const parlayProps = useMemo(
    () => displayProps.filter((p) => parlayIds.has(p.underdog_option_id)),
    [displayProps, parlayIds]
  );

  const parlayCorrelationWarnings = useMemo(() => {
    const warnings: string[] = [];
    const byPlayer = new Map<string, string[]>();
    const byGame = new Map<string, string[]>();
    for (const p of parlayProps) {
      const pKey = p.player_name;
      byPlayer.set(pKey, [...(byPlayer.get(pKey) ?? []), p.stat]);
      if (p.game_title) byGame.set(p.game_title, [...(byGame.get(p.game_title) ?? []), p.player_name]);
    }
    for (const [name, stats] of byPlayer) {
      if (stats.length > 1) warnings.push(`${name} has ${stats.length} correlated props (${stats.join(", ")})`);
    }
    for (const [game, players] of byGame) {
      const unique = [...new Set(players)];
      if (unique.length > 1) warnings.push(`${unique.length} picks from same game: ${game}`);
    }
    return warnings;
  }, [parlayProps]);

  // Comprehensive parlay analytics — joint probability, no-vig estimate,
  // edge, EV, full / fractional Kelly stake, expected profit.
  //
  // Underdog payout source-of-truth hierarchy:
  //   1. User override (parlayPayoutOverride) — what their actual slip shows
  //   2. Underdog standard payout table by leg count (2=3x, 3=6x, 4=10x,
  //      5=20x, 6=37.5x, 7=75x, 8=150x)
  // The legacy "product of per-leg decimal_price" is kept only as a
  // diagnostic for users who want to compare.
  const parlayAnalytics = useMemo(() => {
    const n = parlayProps.length;
    if (!n) {
      return {
        n: 0,
        decimalOdds: 0,
        americanOdds: "—",
        legProductOdds: 0,
        standardPayout: 0,
        usingOverride: false,
        jointModelProb: 0,
        jointNoVigProb: 0,
        impliedFromOdds: 0,
        edge: 0,
        evPerDollar: 0,
        kellyFull: 0,
        kellyUsed: 0,
        recommendedStake: 0,
        potentialPayout: 0,
        potentialProfit: 0,
        expectedProfit: 0,
        hasNegativeEdge: false,
        availableTypes: [] as EntryType[],
        evByType: [] as ReturnType<typeof allEntryEvs>,
        bestType: null as ReturnType<typeof bestEntryType>,
        boostMultiplier: 1,
      };
    }
    const legProductOdds = parlayProps.reduce((acc, p) => acc * (p.decimal_price ?? 1), 1);
    const standardPayout = underdogStandardPayout(n);

    // Per-leg booster multipliers from Underdog (Pick'em Specials, 2x boosts).
    // We treat any per-leg payout_multiplier > 1.81 as a boost above the
    // implicit base price.
    const boostMultiplier = parlayProps.reduce((acc, p) => {
      const m = p.payout_multiplier ?? 0;
      return acc * (m > 1.81 ? m / 1.81 : 1);
    }, 1);

    const usingOverride = parlayPayoutOverride !== null && parlayPayoutOverride > 1;
    // Smart payout detection: if any leg is a non-Pick'em market priced at
    // true sportsbook decimal odds (clearly above the ~1.81 Pick'em base),
    // we can no longer use Underdog's flat payout tables — the whole
    // parlay must be priced as the product of per-leg decimals.
    const hasSportsbookPricing = parlayProps.some(
      (p) => isGameLine(p) && (p.decimal_price ?? 0) > 1.95,
    );
    const decimalOdds = usingOverride
      ? (parlayPayoutOverride as number)
      : hasSportsbookPricing
      ? legProductOdds
      : standardPayout * boostMultiplier;

    const legProbs = parlayProps.map((p) => p.model_prob ?? 0.5);
    const jointModelProb = legProbs.reduce((acc, p) => acc * p, 1);
    const jointNoVigProb = parlayProps.reduce(
      (acc, p) => acc * (p.no_vig_prob ?? p.implied_prob ?? 1 / Math.max(1.01, p.decimal_price ?? 2)),
      1,
    );
    const impliedFromOdds = decimalOdds > 0 ? 1 / decimalOdds : 0;
    const b = decimalOdds - 1;
    const edge = jointModelProb - impliedFromOdds;
    const evPerDollar = jointModelProb * b - (1 - jointModelProb);

    // Compute EV under every supported entry type so the user can pick the
    // optimal one for these legs.
    const evByType = allEntryEvs(n, legProbs);
    const bestType = bestEntryType(n, legProbs);
    const availableTypes = availableEntryTypes(n);

    const kellyFull = kellyFullFraction(jointModelProb, decimalOdds);
    const kellyUsed = Math.max(0, kellyFull / Math.max(1, kellyDivisor));
    const recommendedStake = Math.max(0, bankroll * kellyUsed);
    const potentialPayout = recommendedStake * decimalOdds;
    const potentialProfit = recommendedStake * b;
    const expectedProfit = recommendedStake * evPerDollar;
    return {
      n,
      decimalOdds,
      americanOdds: fmtAmericanFromDecimal(decimalOdds),
      legProductOdds,
      standardPayout,
      usingOverride,
      jointModelProb,
      jointNoVigProb,
      impliedFromOdds,
      edge,
      evPerDollar,
      kellyFull,
      kellyUsed,
      recommendedStake,
      potentialPayout,
      potentialProfit,
      expectedProfit,
      hasNegativeEdge: kellyFull <= 0,
      availableTypes,
      evByType,
      bestType,
      boostMultiplier,
      hasSportsbookPricing,
    };
  }, [parlayProps, bankroll, kellyDivisor, parlayPayoutOverride]);

  // Currently-selected entry-type EV (for display in slip header).
  const selectedEntryEv = useMemo(() => {
    if (!parlayAnalytics.n) return null;
    return entryEv(
      entryType,
      parlayAnalytics.n,
      parlayProps.map((p) => p.model_prob ?? 0.5),
    );
  }, [entryType, parlayAnalytics.n, parlayProps]);

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  }

  const sortIcon = (key: SortKey) => {
    if (sortKey !== key) return <span className="ml-1 text-zinc-300 dark:text-zinc-700">⇅</span>;
    return <span className="ml-1">{sortAsc ? "↑" : "↓"}</span>;
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-50 to-zinc-100 text-zinc-950 dark:from-zinc-950 dark:to-black dark:text-zinc-50">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        <header className="flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <h1 className="text-2xl font-semibold tracking-tight">
              Underdog Prop Predictor
            </h1>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">
              Backend: <span className="font-mono text-xs">{backendUrl}</span>
            </p>
          </div>

          <div className="rounded-xl border border-zinc-200 bg-white/80 p-4 shadow-sm backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/70">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex flex-wrap items-center gap-3">
                <label className="flex items-center gap-2 text-sm">
                  <span className="text-zinc-600 dark:text-zinc-400">Sport</span>
                  <select
                    className="h-9 rounded-md border border-zinc-200 bg-white px-3 text-sm shadow-sm outline-none focus:ring-2 focus:ring-zinc-300 dark:border-zinc-800 dark:bg-zinc-950 dark:focus:ring-zinc-700"
                    value={sport}
                    onChange={(e) => setSport(e.target.value as SportId)}
                  >
                    {SPORT_OPTIONS.map((o) => (
                      <option key={o.id} value={o.id}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </label>

                {availableStats.length > 1 && (
                  <label className="flex items-center gap-2 text-sm">
                    <span className="text-zinc-600 dark:text-zinc-400">Stat</span>
                    <select
                      className="h-9 rounded-md border border-zinc-200 bg-white px-3 text-sm shadow-sm outline-none focus:ring-2 focus:ring-zinc-300 dark:border-zinc-800 dark:bg-zinc-950 dark:focus:ring-zinc-700"
                      value={statFilter}
                      onChange={(e) => setStatFilter(e.target.value)}
                    >
                      <option value="all">All stats</option>
                      {availableStats.map((s) => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </label>
                )}

                <div className="inline-flex rounded-md border border-zinc-200 dark:border-zinc-800" role="tablist" aria-label="Market filter">
                  {([
                    { id: "all", label: "All" },
                    { id: "player_prop", label: "Player Props" },
                    { id: "game_line", label: "Game Lines" },
                  ] as const).map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => setMarketFilter(opt.id)}
                      className={
                        "h-9 px-3 text-xs font-medium first:rounded-l-md last:rounded-r-md " +
                        (marketFilter === opt.id
                          ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                          : "bg-white text-zinc-600 hover:bg-zinc-100 dark:bg-zinc-950 dark:text-zinc-400 dark:hover:bg-zinc-900")
                      }
                      title={
                        opt.id === "game_line"
                          ? "Show only game-level markets (game total, team total, spread, moneyline)"
                          : opt.id === "player_prop"
                          ? "Show only individual player O/U props"
                          : "Show all markets"
                      }
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex flex-col gap-2 w-full sm:w-auto">
                {data ? (
                  <div className="text-xs text-zinc-600 dark:text-zinc-400">
                    Updated{" "}
                    <span className="font-mono text-[11px]">{data.updated_at}</span>
                  </div>
                ) : null}

                <div className="grid grid-cols-2 gap-2 sm:flex sm:flex-wrap sm:items-center">
                  <button
                    className="h-9 rounded-md bg-zinc-900 px-4 text-sm font-medium text-white shadow-sm hover:bg-zinc-800 disabled:opacity-50 dark:bg-zinc-50 dark:text-black dark:hover:bg-zinc-200"
                    onClick={() => void load(true)}
                    disabled={loading}
                  >
                    {loading ? "Loading…" : "Refresh"}
                  </button>

                  <button
                    className="h-9 rounded-md border border-zinc-200 bg-white px-3 text-sm font-medium text-zinc-900 shadow-sm hover:bg-zinc-50 disabled:opacity-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-100 dark:hover:bg-zinc-900"
                    onClick={() => void clearCacheAndReload()}
                    disabled={loading || clearing}
                    title="Clears backend cache and reloads"
                  >
                    {clearing ? "Clearing…" : "Clear cache"}
                  </button>

                  <button
                    className="h-9 rounded-md border border-zinc-200 bg-white px-3 text-sm font-medium text-zinc-900 shadow-sm hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-100 dark:hover:bg-zinc-900"
                    onClick={() => void loadHistory()}
                  >
                    History
                  </button>

                  <button
                    className="h-9 rounded-md border border-violet-300 bg-violet-50 px-3 text-sm font-medium text-violet-900 shadow-sm hover:bg-violet-100 dark:border-violet-800 dark:bg-violet-950/50 dark:text-violet-200 dark:hover:bg-violet-900/50"
                    onClick={() => void openLearning()}
                  >
                    Learning
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* --- Bankroll & Kelly sizing --- */}
          <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 p-4 shadow-sm dark:border-emerald-800/50 dark:bg-emerald-950/20">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
              <div className="flex flex-wrap items-end gap-4">
                <label className="flex flex-col gap-1 text-xs">
                  <span className="font-semibold text-emerald-800 dark:text-emerald-200">Current bankroll</span>
                  <div className="flex items-center gap-1">
                    <span className="text-zinc-500">$</span>
                    <input
                      type="text"
                      inputMode="decimal"
                      className="h-9 w-32 rounded-md border border-emerald-200 bg-white px-2 font-mono text-sm shadow-sm outline-none focus:ring-2 focus:ring-emerald-300 dark:border-emerald-800 dark:bg-zinc-950 dark:focus:ring-emerald-700"
                      value={bankrollInput}
                      onChange={(e) => setBankrollInput(e.target.value)}
                      onBlur={(e) => commitBankroll(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                      placeholder="1000"
                    />
                  </div>
                </label>

                <label className="flex flex-col gap-1 text-xs">
                  <span className="font-semibold text-emerald-800 dark:text-emerald-200">Kelly fraction</span>
                  <select
                    className="h-9 rounded-md border border-emerald-200 bg-white px-2 text-sm shadow-sm outline-none focus:ring-2 focus:ring-emerald-300 dark:border-emerald-800 dark:bg-zinc-950 dark:focus:ring-emerald-700"
                    value={kellyDivisor}
                    onChange={(e) => setKellyDivisor(Number(e.target.value))}
                    title="Lower = safer. Quarter Kelly is the practitioner standard."
                  >
                    <option value={1}>Full Kelly (max growth, max variance)</option>
                    <option value={2}>Half Kelly (75% growth, 50% variance)</option>
                    <option value={4}>Quarter Kelly (44% growth, 6% variance) — recommended</option>
                    <option value={8}>Eighth Kelly (very conservative)</option>
                  </select>
                </label>
              </div>

              <div className="text-xs text-zinc-600 dark:text-zinc-400 sm:text-right">
                <div>
                  Stake sizing follows the <span className="font-semibold">Kelly Criterion</span>:{" "}
                  <span className="font-mono">f* = (b·p − q) / b</span>.
                </div>
                <div className="text-[10px] text-zinc-500">
                  Settings persist locally. Lower fractions trade growth for drawdown protection.
                </div>
              </div>
            </div>
          </div>
        </header>

        <section className="mt-6">
          {jobProgress ? (
            <div className="mb-4 rounded-xl border border-zinc-200 bg-white/80 p-4 shadow-sm backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/70">
              {/* Stage indicator */}
              <div className="flex items-center gap-3 text-sm">
                <div className="flex items-center gap-2 text-zinc-700 dark:text-zinc-200">
                  <span className="relative flex h-2.5 w-2.5">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-zinc-600 opacity-75 dark:bg-zinc-300" />
                    <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-zinc-700 dark:bg-zinc-200" />
                  </span>
                  {jobProgress.stage === "fetch" && (jobProgress.detail || "Fetching props...")}
                  {jobProgress.stage === "espn" && (jobProgress.detail || "Loading ESPN player data...")}
                  {jobProgress.stage === "rank" && (jobProgress.detail || "Computing statistical model...")}
                  {jobProgress.stage === "starting" && "Initializing..."}
                  {jobProgress.stage === "ai_select" && (jobProgress.detail || "AI selecting best picks...")}
                  {jobProgress.stage === "ai" && (
                    <>
                      Generating AI summaries:{" "}
                      <span className="font-mono text-xs">
                        {jobProgress.ai_succeeded}/{jobProgress.ai_target}
                      </span>
                    </>
                  )}
                  {!["fetch", "espn", "rank", "starting", "ai_select", "ai"].includes(jobProgress.stage) && (
                    jobProgress.detail || jobProgress.stage
                  )}
                </div>
              </div>

              {/* Stage steps */}
              <div className="mt-3 flex items-center gap-1">
                {["fetch", "espn", "rank", "ai_select", "ai"].map((s, i) => {
                  const stages = ["fetch", "espn", "rank", "ai_select", "ai"];
                  const currentIdx = stages.indexOf(jobProgress.stage);
                  const isDone = i < currentIdx;
                  const isCurrent = i === currentIdx;
                  return (
                    <div key={s} className="flex items-center gap-1 flex-1">
                      <div className={`h-1.5 flex-1 rounded-full transition-all ${
                        isDone ? "bg-emerald-500 dark:bg-emerald-400"
                        : isCurrent ? "bg-zinc-700 dark:bg-zinc-200"
                        : "bg-zinc-200 dark:bg-zinc-800"
                      }`} />
                    </div>
                  );
                })}
              </div>
              <div className="mt-1.5 flex justify-between text-[10px] text-zinc-400">
                <span>Fetch</span>
                <span>ESPN</span>
                <span>Model</span>
                <span>AI Pick</span>
                <span>AI Analyze</span>
              </div>

              {/* AI progress bar (only show when in AI stage) */}
              {jobProgress.stage === "ai" && (
                <>
                  <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
                    <div
                      className="h-2 rounded-full bg-zinc-900 transition-all dark:bg-zinc-100"
                      style={{
                        width: `${Math.min(100, Math.round((100 * jobProgress.ai_succeeded) / Math.max(1, jobProgress.ai_target)))}%`,
                      }}
                    />
                  </div>
                  <div className="mt-1.5 text-xs text-zinc-500">
                    {jobProgress.ai_attempted > 0 && `${jobProgress.ai_attempted} attempted, `}
                    {jobProgress.analyzed > 0 && `${jobProgress.analyzed} analyzed`}
                  </div>
                </>
              )}
            </div>
          ) : null}

          {error ? (
            <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-900 dark:border-red-900/40 dark:bg-red-950/40 dark:text-red-200">
              <div className="font-medium">Request failed</div>
              <div className="mt-1 font-mono text-xs whitespace-pre-wrap">{error}</div>
              <div className="mt-3 text-xs text-red-800/80 dark:text-red-200/80">
                Make sure the FastAPI backend is running on{" "}
                <span className="font-mono">{backendUrl}</span>.
              </div>
            </div>
          ) : null}

          {(data || modelProps.length > 0) ? (
            <div className="mt-4 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-4 text-sm text-zinc-600 dark:text-zinc-400">
                  <div>
                    Top <span className="font-mono text-xs">{filteredProps.length}</span> picks
                    {!aiFinished && modelProps.length > 0 && (
                      <span className="ml-2 text-xs text-amber-600 dark:text-amber-400">(stat model — AI loading...)</span>
                    )}
                  </div>
                  {remainingProps.length > 0 && (
                    <div className="text-xs text-zinc-400">
                      +{remainingProps.length} more below
                    </div>
                  )}
                </div>
              </div>
              <div className="rounded-md border border-emerald-200 bg-emerald-50/60 px-3 py-2 text-xs text-emerald-900 dark:border-emerald-800/60 dark:bg-emerald-950/30 dark:text-emerald-200">
                Tap the <span className="font-mono font-semibold">+</span> next to any pick to add it to your custom parlay. Combined odds, model probability, edge, and your Kelly-sized recommended stake update live in the slip below.
              </div>
            </div>
          ) : null}

          {/* --- Main table (desktop) --- */}
          <div className="mt-4 hidden overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-sm md:block dark:border-zinc-800 dark:bg-zinc-950">
            <div className="overflow-auto">
              <table className="min-w-[1150px] w-full text-sm">
                <thead className="sticky top-0 z-10 bg-white/95 backdrop-blur dark:bg-zinc-950/95">
                  <tr className="border-b border-zinc-200 text-left dark:border-zinc-800">
                    <th className="px-2 py-3 w-8"></th>
                    <th className="px-2 py-3 w-8">#</th>
                    <th className="px-3 py-3">Player</th>
                    <th className="px-3 py-3">Pick</th>
                    <th className="px-3 py-3 max-w-[220px]">AI Summary</th>
                    <th className="cursor-pointer px-3 py-3 select-none" onClick={() => handleSort("hit_rate")}>
                      Hit{sortIcon("hit_rate")}
                    </th>
                    <th className="cursor-pointer px-3 py-3 select-none" onClick={() => handleSort("implied_prob")}>
                      Impl{sortIcon("implied_prob")}
                    </th>
                    <th className="cursor-pointer px-3 py-3 select-none" onClick={() => handleSort("model_prob")}>
                      Model{sortIcon("model_prob")}
                    </th>
                    <th className="cursor-pointer px-3 py-3 select-none" onClick={() => handleSort("edge")}>
                      Edge{sortIcon("edge")}
                    </th>
                    <th className="cursor-pointer px-3 py-3 select-none" onClick={() => handleSort("ev")}>
                      EV{sortIcon("ev")}
                    </th>
                    <th className="cursor-pointer px-3 py-3 select-none" onClick={() => handleSort("volatility")}>
                      Vol{sortIcon("volatility")}
                    </th>
                    <th className="cursor-pointer px-3 py-3 select-none" onClick={() => handleSort("score")}>
                      Score{sortIcon("score")}
                    </th>
                    <th className="px-2 py-3 w-8" title="Add to parlay">+</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredProps.map((p, idx) => {
                    const isOpen = !!expanded[p.underdog_option_id];
                    const inParlay = parlayIds.has(p.underdog_option_id);
                    return (
                      <Fragment key={p.underdog_option_id}>
                        <tr
                          className={`border-b border-zinc-100 align-top hover:bg-zinc-50 dark:border-zinc-900 dark:hover:bg-zinc-900/30 ${
                            idx % 2 === 0
                              ? "bg-white dark:bg-zinc-950"
                              : "bg-zinc-50/40 dark:bg-zinc-950"
                          } ${inParlay ? "ring-1 ring-inset ring-emerald-400/40" : ""}`}
                        >
                          <td className="px-2 py-3">
                            <button
                              className="inline-flex h-7 w-7 items-center justify-center rounded-md border border-zinc-200 bg-white text-zinc-700 shadow-sm hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200 dark:hover:bg-zinc-900"
                              onClick={() =>
                                setExpanded((prev) => ({
                                  ...prev,
                                  [p.underdog_option_id]: !prev[p.underdog_option_id],
                                }))
                              }
                              aria-label={isOpen ? "Collapse" : "Expand"}
                            >
                              <span className={`text-xs transition-transform ${isOpen ? "rotate-90" : ""}`}>▶</span>
                            </button>
                          </td>
                          <td className="px-2 py-3 text-zinc-500">{idx + 1}</td>
                          <td className="px-3 py-3">
                            <div className="flex items-center gap-2">
                              <div>
                                <div className="flex items-center gap-1.5 font-medium">
                                  {p.player_name}
                                  {trendArrow(p.trend_direction)}
                                  {tierBadge(p.confidence_tier)}
                                  {p.model_ai_agree && (
                                    <span className="rounded-full bg-blue-100 px-1.5 py-0.5 text-[9px] font-semibold text-blue-700 dark:bg-blue-900/40 dark:text-blue-300" title="Model & AI agree">
                                      AGREE
                                    </span>
                                  )}
                                  {isGameLine(p) && (
                                    <span
                                      className="rounded-full bg-violet-100 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-violet-700 dark:bg-violet-900/40 dark:text-violet-300"
                                      title={`Game-level market (${p.market_type})`}
                                    >
                                      {p.market_type === "game_total"
                                        ? "GAME TOTAL"
                                        : p.market_type === "team_total"
                                        ? "TEAM TOTAL"
                                        : p.market_type === "spread"
                                        ? "SPREAD"
                                        : "MONEYLINE"}
                                    </span>
                                  )}
                                </div>
                                <div className="mt-0.5 flex items-center gap-2 text-xs text-zinc-500 dark:text-zinc-400">
                                  <span className="font-mono">{p.sport}</span>
                                  <span>{p.stat}</span>
                                  {p.is_home !== null && p.is_home !== undefined && (
                                    <span className={`font-mono text-[10px] ${p.is_home ? "text-emerald-600 dark:text-emerald-400" : "text-zinc-400"}`}>
                                      {p.is_home ? "HOME" : "AWAY"}
                                    </span>
                                  )}
                                  {p.is_b2b && <span className="font-mono text-[10px] text-amber-600 dark:text-amber-400">B2B</span>}
                                </div>
                              </div>
                            </div>
                          </td>
                          <td className="px-3 py-3 font-mono text-xs">
                            <div>{fmtPick(p)}</div>
                            <div className="mt-0.5 text-[11px] text-zinc-500">
                              {p.team_abbr || "?"} vs {p.opponent_abbr || "?"}
                            </div>
                          </td>
                          <td className="px-3 py-3 text-xs text-zinc-700 dark:text-zinc-300 max-w-[220px]">
                            <div className="leading-5 break-words">
                              {shortText(p.ai_summary, 140) ?? (
                                <span className="text-zinc-400">(no AI summary)</span>
                              )}
                            </div>
                            <div className="mt-1.5 flex flex-wrap gap-1.5">
                              <span className={`rounded-full border border-zinc-200 bg-white px-2 py-0.5 text-[10px] dark:border-zinc-800 dark:bg-zinc-950 ${biasColor(p.ai_bias)}`}>
                                {biasLabel(p.ai_bias)}
                                {p.ai_confidence != null ? ` (${fmtNum(p.ai_confidence, 2)})` : ""}
                              </span>
                              {p.ai_prob_adjustment != null && p.ai_prob_adjustment !== 0 && (
                                <span className={`rounded-full border px-2 py-0.5 text-[10px] font-mono ${
                                  p.ai_prob_adjustment > 0
                                    ? "border-emerald-200 text-emerald-700 dark:border-emerald-800 dark:text-emerald-400"
                                    : "border-rose-200 text-rose-700 dark:border-rose-800 dark:text-rose-400"
                                }`}>
                                  AI adj: {p.ai_prob_adjustment > 0 ? "+" : ""}{(p.ai_prob_adjustment * 100).toFixed(1)}%
                                </span>
                              )}
                            </div>
                          </td>
                          <td className="px-3 py-3 font-mono text-xs">
                            {p.hit_rate_str ?? "—"}
                          </td>
                          <td className="px-3 py-3 font-mono text-xs">{fmtPct(p.implied_prob)}</td>
                          <td className="px-3 py-3 font-mono text-xs">{fmtPct(p.model_prob)}</td>
                          <td className={`px-3 py-3 font-mono text-xs ${edgeColor(p.edge)}`}>{fmtPct(p.edge)}</td>
                          <td className="px-3 py-3 font-mono text-xs">{fmtNum(p.ev, 4)}</td>
                          <td className="px-3 py-3 font-mono text-xs">{fmtNum(p.volatility, 2)}</td>
                          <td className="px-3 py-3 font-mono text-xs">{fmtNum(p.score, 3)}</td>
                          <td className="px-2 py-3">
                            <button
                              className={`inline-flex h-7 w-7 items-center justify-center rounded-md border text-xs shadow-sm ${
                                inParlay
                                  ? "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                                  : "border-zinc-200 bg-white text-zinc-600 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-300 dark:hover:bg-zinc-900"
                              }`}
                              onClick={() => toggleParlay(p.underdog_option_id)}
                              title={inParlay ? "Remove from parlay" : "Add to parlay"}
                            >
                              {inParlay ? "✓" : "+"}
                            </button>
                          </td>
                        </tr>

                        {isOpen && (
                          <tr key={`${p.underdog_option_id}__expanded`} className="border-b border-zinc-100 dark:border-zinc-900">
                            <td colSpan={13} className="px-4 py-4">
                              <div className="grid gap-4 rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900/20">
                                <div className="grid gap-1">
                                  <div className="text-xs font-semibold text-zinc-600 dark:text-zinc-300">AI Summary</div>
                                  <div className="max-w-full overflow-x-auto whitespace-pre-wrap break-words text-sm leading-6 text-zinc-800 dark:text-zinc-200">
                                    {p.ai_summary ?? <span className="text-zinc-500">No AI summary available.</span>}
                                  </div>
                                  <div className="mt-2 flex flex-wrap gap-2 text-xs">
                                    <span className={`rounded-full border border-zinc-200 bg-white px-3 py-1 dark:border-zinc-800 dark:bg-zinc-950 ${biasColor(p.ai_bias)}`}>
                                      {biasLabel(p.ai_bias)} {p.ai_confidence != null ? `(${fmtNum(p.ai_confidence, 2)})` : ""}
                                    </span>
                                    <span className="rounded-full border border-zinc-200 bg-white px-3 py-1 font-mono text-zinc-700 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200">
                                      model {fmtPct(p.model_prob)} | implied {fmtPct(p.implied_prob)} | edge {fmtPct(p.edge)}
                                    </span>
                                    {p.avg_minutes != null && (
                                      <span className="rounded-full border border-zinc-200 bg-white px-3 py-1 font-mono text-zinc-700 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200">
                                        ~{p.avg_minutes} min/game
                                      </span>
                                    )}
                                    {p.trend_short_avg != null && (
                                      <span className="rounded-full border border-zinc-200 bg-white px-3 py-1 font-mono text-zinc-700 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200">
                                        last3 avg: {p.trend_short_avg} {trendArrow(p.trend_direction)}
                                      </span>
                                    )}
                                  </div>
                                </div>

                                <div className="grid gap-4 md:grid-cols-3">
                                  <div className="md:col-span-3 grid gap-4 md:grid-cols-2">
                                    <div>
                                      <div className="text-xs font-semibold text-zinc-600 dark:text-zinc-300">
                                        Last 10 games ({p.display_stat ?? p.stat})
                                      </div>
                                      <div className="mt-2 overflow-hidden rounded-md border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
                                        <table className="w-full text-xs">
                                          <thead className="bg-zinc-50 text-zinc-600 dark:bg-zinc-900/40 dark:text-zinc-300">
                                            <tr>
                                              <th className="px-2 py-2 text-left">Date</th>
                                              <th className="px-2 py-2 text-left">Opp</th>
                                              <th className="px-2 py-2 text-right">Value</th>
                                              <th className="px-2 py-2 text-right">Hit</th>
                                            </tr>
                                          </thead>
                                          <tbody>
                                            {(p.recent_games ?? []).length ? (
                                              (p.recent_games ?? []).slice(0, 10).map((g, i) => (
                                                <tr key={i} className="border-t border-zinc-100 dark:border-zinc-900">
                                                  <td className="px-2 py-1 font-mono text-[11px]">{formatDate(g.game_date)}</td>
                                                  <td className="px-2 py-1 font-mono text-[11px]">{g.opponent_abbr ?? "—"}</td>
                                                  <td className="px-2 py-1 text-right font-mono text-[11px] text-zinc-950 dark:text-zinc-50">{fmtVal(g.value)}</td>
                                                  <td className="px-2 py-1 text-right font-mono text-[11px]">
                                                    {(() => {
                                                      const hr = hitResult(g.value, p.line, p.side);
                                                      if (hr === null) return <span className="text-zinc-500">—</span>;
                                                      return (
                                                        <span className={hr ? "text-emerald-600 dark:text-emerald-300" : "text-rose-600 dark:text-rose-300"}>
                                                          {hr ? "✓" : "✕"}
                                                        </span>
                                                      );
                                                    })()}
                                                  </td>
                                                </tr>
                                              ))
                                            ) : (
                                              <tr>
                                                <td colSpan={4} className="px-2 py-3 text-zinc-500">No game log available.</td>
                                              </tr>
                                            )}
                                          </tbody>
                                        </table>
                                      </div>
                                    </div>

                                    <div>
                                      <div className="text-xs font-semibold text-zinc-600 dark:text-zinc-300">
                                        Vs {p.opponent_abbr ?? "opponent"} (this season)
                                      </div>
                                      <div className="mt-2 overflow-hidden rounded-md border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
                                        <table className="w-full text-xs">
                                          <thead className="bg-zinc-50 text-zinc-600 dark:bg-zinc-900/40 dark:text-zinc-300">
                                            <tr>
                                              <th className="px-2 py-2 text-left">Date</th>
                                              <th className="px-2 py-2 text-right">Value</th>
                                              <th className="px-2 py-2 text-right">Hit</th>
                                            </tr>
                                          </thead>
                                          <tbody>
                                            {(p.vs_opponent_games ?? []).length ? (
                                              (p.vs_opponent_games ?? []).slice(0, 10).map((g, i) => (
                                                <tr key={i} className="border-t border-zinc-100 dark:border-zinc-900">
                                                  <td className="px-2 py-1 font-mono text-[11px]">{formatDate(g.game_date)}</td>
                                                  <td className="px-2 py-1 text-right font-mono text-[11px] text-zinc-950 dark:text-zinc-50">{fmtVal(g.value)}</td>
                                                  <td className="px-2 py-1 text-right font-mono text-[11px]">
                                                    {(() => {
                                                      const hr = hitResult(g.value, p.line, p.side);
                                                      if (hr === null) return <span className="text-zinc-500">—</span>;
                                                      return (
                                                        <span className={hr ? "text-emerald-600 dark:text-emerald-300" : "text-rose-600 dark:text-rose-300"}>
                                                          {hr ? "✓" : "✕"}
                                                        </span>
                                                      );
                                                    })()}
                                                  </td>
                                                </tr>
                                              ))
                                            ) : (
                                              <tr>
                                                <td colSpan={3} className="px-2 py-3 text-zinc-500">No meetings logged yet.</td>
                                              </tr>
                                            )}
                                          </tbody>
                                        </table>
                                      </div>
                                    </div>
                                  </div>

                                  {/* Statistical profile */}
                                  <div className="md:col-span-3">
                                    <div className="text-xs font-semibold text-zinc-600 dark:text-zinc-300 mb-2">Statistical Profile</div>
                                    <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs sm:grid-cols-3 md:grid-cols-6">
                                      {p.stat_median != null && (
                                        <div>
                                          <span className="text-zinc-500">Median:</span>{" "}
                                          <span className="font-mono text-zinc-800 dark:text-zinc-100">{p.stat_median}</span>
                                        </div>
                                      )}
                                      {p.stat_floor != null && p.stat_ceiling != null && (
                                        <div>
                                          <span className="text-zinc-500">Range:</span>{" "}
                                          <span className="font-mono text-zinc-800 dark:text-zinc-100">{p.stat_floor}–{p.stat_ceiling}</span>
                                        </div>
                                      )}
                                      {p.stat_consistency != null && (
                                        <div>
                                          <span className="text-zinc-500">Consistency:</span>{" "}
                                          <span className="font-mono text-zinc-800 dark:text-zinc-100">{(p.stat_consistency * 100).toFixed(0)}%</span>
                                        </div>
                                      )}
                                      {p.current_streak != null && p.current_streak !== 0 && (
                                        <div>
                                          <span className="text-zinc-500">Streak:</span>{" "}
                                          <span className={`font-mono ${p.current_streak > 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                                            {p.current_streak > 0 ? `${p.current_streak} overs` : `${Math.abs(p.current_streak)} unders`}
                                          </span>
                                        </div>
                                      )}
                                      {p.line_percentile != null && (
                                        <div>
                                          <span className="text-zinc-500">Line pctl:</span>{" "}
                                          <span className={`font-mono ${
                                            p.line_percentile > 0.6 ? "text-rose-600 dark:text-rose-400"
                                            : p.line_percentile < 0.4 ? "text-emerald-600 dark:text-emerald-400"
                                            : "text-zinc-800 dark:text-zinc-100"
                                          }`}>
                                            {(p.line_percentile * 100).toFixed(0)}%
                                          </span>
                                        </div>
                                      )}
                                      {p.no_vig_prob != null && (
                                        <div>
                                          <span className="text-zinc-500">No-vig:</span>{" "}
                                          <span className="font-mono text-zinc-800 dark:text-zinc-100">{(p.no_vig_prob * 100).toFixed(1)}%</span>
                                        </div>
                                      )}
                                      {p.kelly_fraction != null && p.kelly_fraction > 0 && (
                                        <div>
                                          <span className="text-zinc-500">Kelly:</span>{" "}
                                          <span className="font-mono text-emerald-600 dark:text-emerald-400">{(p.kelly_fraction * 100).toFixed(1)}%</span>
                                        </div>
                                      )}
                                      {p.edge_confidence != null && (
                                        <div>
                                          <span className="text-zinc-500">Edge conf:</span>{" "}
                                          <span className={`font-mono ${
                                            p.edge_confidence > 0.5 ? "text-emerald-600 dark:text-emerald-400"
                                            : p.edge_confidence < 0.2 ? "text-rose-600 dark:text-rose-400"
                                            : "text-zinc-800 dark:text-zinc-100"
                                          }`}>
                                            {(p.edge_confidence * 100).toFixed(0)}%
                                          </span>
                                        </div>
                                      )}
                                      {p.per_minute_rate != null && (
                                        <div>
                                          <span className="text-zinc-500">Per-min:</span>{" "}
                                          <span className="font-mono text-zinc-800 dark:text-zinc-100">{p.per_minute_rate.toFixed(3)}</span>
                                        </div>
                                      )}
                                    </div>
                                  </div>

                                  <div>
                                    <div className="text-xs font-semibold text-emerald-700 dark:text-emerald-300">Tailwinds</div>
                                    {p.ai_tailwinds?.length ? (
                                      <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-zinc-700 dark:text-zinc-300 break-words">
                                        {p.ai_tailwinds.slice(0, 8).map((t, i) => <li key={i}>{t}</li>)}
                                      </ul>
                                    ) : <div className="mt-2 text-xs text-zinc-500">—</div>}
                                  </div>
                                  <div>
                                    <div className="text-xs font-semibold text-rose-700 dark:text-rose-300">Risk factors</div>
                                    {p.ai_risk_factors?.length ? (
                                      <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-zinc-700 dark:text-zinc-300 break-words">
                                        {p.ai_risk_factors.slice(0, 8).map((t, i) => <li key={i}>{t}</li>)}
                                      </ul>
                                    ) : <div className="mt-2 text-xs text-zinc-500">—</div>}
                                  </div>
                                  <div>
                                    <div className="text-xs font-semibold text-zinc-600 dark:text-zinc-300">Notes</div>
                                    {p.notes?.length ? (
                                      <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-zinc-700 dark:text-zinc-300 break-words">
                                        {p.notes.slice(0, 8).map((t, i) => <li key={i}>{t}</li>)}
                                      </ul>
                                    ) : <div className="mt-2 text-xs text-zinc-500">—</div>}
                                  </div>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    );
                  })}
                  {!filteredProps.length && (
                    <tr>
                      <td colSpan={13} className="px-4 py-8 text-center text-sm text-zinc-500">
                        {loading ? "Loading props…" : "No props returned yet."}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="border-t border-zinc-200 px-4 py-3 text-xs text-zinc-600 dark:border-zinc-800 dark:text-zinc-400">
              Showing top {filteredProps.length} picks. Click column headers to sort.
            </div>
          </div>

          {/* --- Mobile card layout --- */}
          <div className="mt-4 space-y-3 md:hidden">
            {filteredProps.map((p, idx) => {
              const isOpen = !!expanded[p.underdog_option_id];
              const inParlay = parlayIds.has(p.underdog_option_id);
              return (
                <div
                  key={p.underdog_option_id}
                  className={`rounded-xl border bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-950 ${
                    inParlay ? "border-emerald-300 dark:border-emerald-700" : "border-zinc-200"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="flex items-center gap-1.5 font-medium">
                        <span className="text-zinc-400 text-xs">#{idx + 1}</span>
                        {p.player_name}
                        {trendArrow(p.trend_direction)}
                        {tierBadge(p.confidence_tier)}
                        {isGameLine(p) && (
                          <span
                            className="rounded-full bg-violet-100 px-1.5 py-0.5 text-[9px] font-semibold uppercase text-violet-700 dark:bg-violet-900/40 dark:text-violet-300"
                            title={`Game-level market (${p.market_type})`}
                          >
                            {p.market_type === "game_total"
                              ? "GAME"
                              : p.market_type === "team_total"
                              ? "TEAM"
                              : p.market_type === "spread"
                              ? "SPREAD"
                              : "ML"}
                          </span>
                        )}
                      </div>
                      <div className="mt-1 flex flex-wrap gap-2 text-xs text-zinc-500">
                        <span className="font-mono">{p.sport}</span>
                        <span>{fmtPick(p)}</span>
                        {p.hit_rate_str && <span className="font-mono">{p.hit_rate_str}</span>}
                        {p.is_home != null && <span className="font-mono text-[10px]">{p.is_home ? "HOME" : "AWAY"}</span>}
                        {p.is_b2b && <span className="font-mono text-[10px] text-amber-600">B2B</span>}
                      </div>
                    </div>
                    <div className="flex gap-1.5">
                      <button
                        className={`h-7 w-7 rounded-md border text-xs ${
                          inParlay ? "border-emerald-300 bg-emerald-50 text-emerald-700" : "border-zinc-200 bg-white text-zinc-600"
                        }`}
                        onClick={() => toggleParlay(p.underdog_option_id)}
                      >
                        {inParlay ? "✓" : "+"}
                      </button>
                      <button
                        className="h-7 w-7 rounded-md border border-zinc-200 bg-white text-xs text-zinc-600"
                        onClick={() => setExpanded((prev) => ({ ...prev, [p.underdog_option_id]: !prev[p.underdog_option_id] }))}
                      >
                        <span className={`transition-transform inline-block ${isOpen ? "rotate-90" : ""}`}>▶</span>
                      </button>
                    </div>
                  </div>

                  <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
                    <div>
                      <div className="text-zinc-400">Edge</div>
                      <div className={`font-mono ${edgeColor(p.edge)}`}>{fmtPct(p.edge)}</div>
                    </div>
                    <div>
                      <div className="text-zinc-400">Model</div>
                      <div className="font-mono">{fmtPct(p.model_prob)}</div>
                    </div>
                    <div>
                      <div className="text-zinc-400">Score</div>
                      <div className="font-mono">{fmtNum(p.score, 2)}</div>
                    </div>
                  </div>

                  {p.ai_summary && (
                    <div className="mt-3 text-xs text-zinc-700 dark:text-zinc-300 break-words leading-5">
                      {shortText(p.ai_summary, 200)}
                    </div>
                  )}

                  {isOpen && (
                    <div className="mt-3 border-t border-zinc-100 pt-3 dark:border-zinc-800">
                      <div className="text-xs text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap break-words leading-5">
                        {p.ai_summary}
                      </div>
                      <div className="mt-2 flex flex-wrap gap-1.5 text-[10px]">
                        <span className={`rounded-full border px-2 py-0.5 ${biasColor(p.ai_bias)}`}>
                          {biasLabel(p.ai_bias)}
                        </span>
                        <span className="rounded-full border border-zinc-200 px-2 py-0.5 font-mono">
                          impl {fmtPct(p.implied_prob)} | EV {fmtNum(p.ev, 3)} | vol {fmtNum(p.volatility, 2)}
                        </span>
                      </div>
                      {(p.recent_games ?? []).length > 0 && (
                        <div className="mt-3">
                          <div className="text-[11px] font-semibold text-zinc-600 dark:text-zinc-300">Recent games</div>
                          <div className="mt-1 flex flex-wrap gap-1">
                            {(p.recent_games ?? []).slice(0, 10).map((g, i) => {
                              const hr = hitResult(g.value, p.line, p.side);
                              return (
                                <span key={i} className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-mono ${
                                  hr === true ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
                                  : hr === false ? "bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-300"
                                  : "bg-zinc-100 text-zinc-500"
                                }`}>
                                  {fmtVal(g.value)}
                                </span>
                              );
                            })}
                          </div>
                        </div>
                      )}
                      {/* Stat profile (mobile) */}
                      {(p.stat_median != null || p.stat_consistency != null || p.current_streak) && (
                        <div className="mt-3">
                          <div className="text-[11px] font-semibold text-zinc-600 dark:text-zinc-300">Stat Profile</div>
                          <div className="mt-1 flex flex-wrap gap-1.5 text-[10px]">
                            {p.stat_median != null && (
                              <span className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono dark:bg-zinc-800">
                                Med: {p.stat_median}
                              </span>
                            )}
                            {p.stat_floor != null && p.stat_ceiling != null && (
                              <span className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono dark:bg-zinc-800">
                                {p.stat_floor}–{p.stat_ceiling}
                              </span>
                            )}
                            {p.stat_consistency != null && (
                              <span className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono dark:bg-zinc-800">
                                {(p.stat_consistency * 100).toFixed(0)}% consistent
                              </span>
                            )}
                            {p.current_streak != null && p.current_streak !== 0 && (
                              <span className={`rounded px-1.5 py-0.5 font-mono ${
                                p.current_streak > 0
                                  ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
                                  : "bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-300"
                              }`}>
                                {p.current_streak > 0 ? `${p.current_streak} overs` : `${Math.abs(p.current_streak)} unders`}
                              </span>
                            )}
                            {p.line_percentile != null && (
                              <span className={`rounded px-1.5 py-0.5 font-mono ${
                                p.line_percentile > 0.6 ? "bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-300"
                                : p.line_percentile < 0.4 ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
                                : "bg-zinc-100 dark:bg-zinc-800"
                              }`}>
                                Line at {(p.line_percentile * 100).toFixed(0)}%
                              </span>
                            )}
                          </div>
                        </div>
                      )}
                      {p.ai_tailwinds?.length ? (
                        <div className="mt-2">
                          <div className="text-[11px] font-semibold text-emerald-700 dark:text-emerald-300">Tailwinds</div>
                          <ul className="mt-1 list-disc pl-4 text-[11px] text-zinc-700 dark:text-zinc-300 space-y-0.5">
                            {p.ai_tailwinds.slice(0, 4).map((t, i) => <li key={i}>{t}</li>)}
                          </ul>
                        </div>
                      ) : null}
                      {p.ai_risk_factors?.length ? (
                        <div className="mt-2">
                          <div className="text-[11px] font-semibold text-rose-700 dark:text-rose-300">Risks</div>
                          <ul className="mt-1 list-disc pl-4 text-[11px] text-zinc-700 dark:text-zinc-300 space-y-0.5">
                            {p.ai_risk_factors.slice(0, 4).map((t, i) => <li key={i}>{t}</li>)}
                          </ul>
                        </div>
                      ) : null}
                    </div>
                  )}
                </div>
              );
            })}
            {!filteredProps.length && !loading && (
              <div className="text-center text-sm text-zinc-500 py-8">No props returned yet.</div>
            )}
            {loading && !filteredProps.length && (
              <div className="text-center text-sm text-zinc-500 py-8">Loading props…</div>
            )}
          </div>

          {/* --- Remaining props (condensed) --- */}
          {remainingProps.length > 0 && (
            <div className="mt-6">
              <details>
                <summary className="cursor-pointer text-sm font-medium text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100">
                  All other props ({remainingProps.length})
                </summary>
                <div className="mt-3 overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
                  <div className="overflow-auto max-h-[400px]">
                    <table className="w-full text-xs">
                      <thead className="sticky top-0 bg-zinc-50/95 backdrop-blur dark:bg-zinc-900/95">
                        <tr className="border-b border-zinc-200 text-left dark:border-zinc-800">
                          <th className="px-2 py-2 w-6">#</th>
                          <th className="px-2 py-2">Player</th>
                          <th className="px-2 py-2">Pick</th>
                          <th className="px-2 py-2">Hit</th>
                          <th className="px-2 py-2">Model</th>
                          <th className="px-2 py-2">Edge</th>
                          <th className="px-2 py-2">Score</th>
                          <th className="px-2 py-2 w-6">+</th>
                        </tr>
                      </thead>
                      <tbody>
                        {remainingProps.map((p, idx) => {
                          const inParlay = parlayIds.has(p.underdog_option_id);
                          return (
                            <tr
                              key={p.underdog_option_id}
                              className="border-b border-zinc-100 hover:bg-zinc-50 dark:border-zinc-900 dark:hover:bg-zinc-900/30"
                            >
                              <td className="px-2 py-1.5 text-zinc-400 font-mono">{idx + 11}</td>
                              <td className="px-2 py-1.5">
                                <div className="font-medium text-zinc-800 dark:text-zinc-100">{p.player_name}</div>
                                <div className="text-[10px] text-zinc-400">{p.sport} {p.team_abbr ? `(${p.team_abbr})` : ""}</div>
                              </td>
                              <td className="px-2 py-1.5 font-mono">
                                {fmtPick(p)}
                              </td>
                              <td className="px-2 py-1.5 font-mono">{p.hit_rate_str ?? "—"}</td>
                              <td className="px-2 py-1.5 font-mono">{fmtPct(p.model_prob)}</td>
                              <td className={`px-2 py-1.5 font-mono ${edgeColor(p.edge)}`}>{fmtPct(p.edge)}</td>
                              <td className="px-2 py-1.5 font-mono">{fmtNum(p.score, 2)}</td>
                              <td className="px-2 py-1.5">
                                <button
                                  className={`inline-flex h-6 w-6 items-center justify-center rounded text-[10px] border ${
                                    inParlay
                                      ? "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                                      : "border-zinc-200 bg-white text-zinc-500 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-400"
                                  }`}
                                  onClick={() => toggleParlay(p.underdog_option_id)}
                                >
                                  {inParlay ? "✓" : "+"}
                                </button>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </details>
            </div>
          )}

          {/* --- Custom Parlay Slip + Kelly-sized stake --- */}
          {parlayProps.length > 0 && (
            <div className="mt-6 rounded-xl border border-emerald-200 bg-emerald-50/50 p-4 shadow-sm dark:border-emerald-800 dark:bg-emerald-950/30">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-emerald-800 dark:text-emerald-200">
                  Custom Parlay ({parlayAnalytics.n} leg{parlayAnalytics.n > 1 ? "s" : ""})
                </h3>
                <button
                  className="text-xs text-emerald-700 hover:underline dark:text-emerald-400"
                  onClick={() => setParlayIds(new Set())}
                >
                  Clear all
                </button>
              </div>

              {/* Per-leg list */}
              <div className="mt-3 space-y-1.5">
                {parlayProps.map((p, i) => (
                  <div key={p.underdog_option_id} className="flex items-center justify-between rounded-md border border-emerald-100 bg-white px-3 py-2 text-xs dark:border-emerald-900/40 dark:bg-zinc-950">
                    <div className="flex items-center gap-2">
                      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-100 text-[10px] font-bold text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300">
                        {i + 1}
                      </span>
                      <div>
                        <span className="font-medium">{p.player_name}</span>{" "}
                        <span className="text-zinc-500">{fmtPick(p)}</span>
                        {p.game_title ? (
                          <span className="ml-1 text-[10px] text-zinc-400">— {p.game_title}</span>
                        ) : null}
                      </div>
                    </div>
                    <div className="flex items-center gap-3 font-mono text-[11px]">
                      <span className="text-zinc-600 dark:text-zinc-400" title="Model probability of this leg hitting">
                        p {fmtPct(p.model_prob)}
                      </span>
                      <span className="text-zinc-500" title="Decimal odds of this leg">
                        {(p.decimal_price ?? 0).toFixed(2)}x
                      </span>
                      <span className={edgeColor(p.edge)} title="Edge vs no-vig market price">
                        {fmtPct(p.edge)}
                      </span>
                      <button
                        onClick={() => toggleLock(p.underdog_option_id)}
                        className={
                          lockedIds.has(p.underdog_option_id)
                            ? "text-amber-500 hover:text-amber-600"
                            : "text-zinc-300 hover:text-zinc-500 dark:text-zinc-700 dark:hover:text-zinc-500"
                        }
                        title={lockedIds.has(p.underdog_option_id) ? "Unlock pick" : "Lock pick (protect from removal)"}
                      >
                        {lockedIds.has(p.underdog_option_id) ? "🔒" : "🔓"}
                      </button>
                      <button
                        className={
                          lockedIds.has(p.underdog_option_id)
                            ? "cursor-not-allowed text-zinc-300 dark:text-zinc-700"
                            : "text-rose-500 hover:text-rose-700"
                        }
                        disabled={lockedIds.has(p.underdog_option_id)}
                        onClick={() => {
                          if (!lockedIds.has(p.underdog_option_id)) toggleParlay(p.underdog_option_id);
                        }}
                        title={lockedIds.has(p.underdog_option_id) ? "Pick is locked — unlock to remove" : "Remove from parlay"}
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {/* Entry-type selector (Standard, Insurance, Flex) with EV per type */}
              {parlayAnalytics.availableTypes.length > 1 && (
                <div className="mt-4 rounded-md border border-emerald-200 bg-white p-3 dark:border-emerald-800 dark:bg-zinc-950">
                  <div className="flex items-center justify-between">
                    <div className="text-[11px] uppercase tracking-wide text-emerald-700 dark:text-emerald-300">
                      Entry type
                    </div>
                    {parlayAnalytics.bestType && (
                      <div className="text-[10px] font-mono text-emerald-700 dark:text-emerald-300">
                        Best:{" "}
                        <span className="font-semibold uppercase">
                          {parlayAnalytics.bestType.entryType}
                        </span>{" "}
                        ({(parlayAnalytics.bestType.evPerDollar * 100).toFixed(1)}% EV/$)
                      </div>
                    )}
                  </div>
                  {parlayAnalytics.hasSportsbookPricing && (
                    <div className="mt-2 rounded-md border border-amber-200 bg-amber-50 px-2 py-1.5 text-[10px] text-amber-800 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-300">
                      Slip contains a non-Pick&apos;em market — Underdog Flex/Insurance entry types don&apos;t apply. Using true decimal-odds payout.
                    </div>
                  )}
                  <div className="mt-2 grid grid-cols-3 gap-2">
                    {(["standard", "insurance", "flex"] as EntryType[]).map((et) => {
                      const ev = parlayAnalytics.evByType.find((e) => e.entryType === et);
                      const enabled =
                        parlayAnalytics.availableTypes.includes(et) &&
                        !(parlayAnalytics.hasSportsbookPricing && et !== "standard");
                      const isSelected = entryType === et;
                      const isBest =
                        parlayAnalytics.bestType?.entryType === et;
                      return (
                        <button
                          key={et}
                          disabled={!enabled}
                          onClick={() => setEntryType(et)}
                          className={[
                            "flex flex-col items-start rounded-md border px-2 py-1.5 text-left text-xs transition-colors",
                            !enabled
                              ? "cursor-not-allowed border-zinc-200 bg-zinc-50 text-zinc-400 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-600"
                              : isSelected
                              ? "border-emerald-500 bg-emerald-50 text-emerald-900 dark:border-emerald-400 dark:bg-emerald-950/40 dark:text-emerald-100"
                              : "border-emerald-200 bg-white hover:bg-emerald-50 dark:border-emerald-800 dark:bg-zinc-950 dark:hover:bg-emerald-950/20",
                          ].join(" ")}
                          title={!enabled ? `${et} not offered for ${parlayAnalytics.n}-pick entries` : undefined}
                        >
                          <div className="flex w-full items-center justify-between">
                            <span className="font-semibold uppercase">{et}</span>
                            {isBest && enabled && (
                              <span className="rounded-full bg-emerald-200 px-1 text-[9px] font-bold uppercase text-emerald-800 dark:bg-emerald-700 dark:text-emerald-100">
                                top
                              </span>
                            )}
                          </div>
                          <span className="font-mono text-[10px] text-zinc-500">
                            {enabled && ev
                              ? `EV ${ev.evPerDollar >= 0 ? "+" : ""}${(ev.evPerDollar * 100).toFixed(1)}%`
                              : "—"}
                          </span>
                          <span className="font-mono text-[10px] text-zinc-500">
                            {enabled && ev ? `payout ${ev.expectedPayoutMultiplier.toFixed(2)}x` : ""}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                  {selectedEntryEv && (
                    <div className="mt-2 text-[10px] leading-4 text-zinc-500 dark:text-zinc-400">
                      <span className="font-semibold uppercase">{selectedEntryEv.entryType}</span>:{" "}
                      EV/$ {selectedEntryEv.evPerDollar >= 0 ? "+" : ""}
                      {selectedEntryEv.evPerDollar.toFixed(3)} · joint full-win prob{" "}
                      {(selectedEntryEv.winProbabilityFull * 100).toFixed(2)}%
                      {(entryType === "flex" || entryType === "insurance") && (
                        <span> · payouts pay even on partial misses; see breakdown above.</span>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Underdog payout — table-default + manual override */}
              <div className="mt-4 grid grid-cols-1 gap-3 border-t border-emerald-200 pt-3 sm:grid-cols-2 dark:border-emerald-800">
                <div className="rounded-md border border-emerald-200 bg-white p-3 dark:border-emerald-800 dark:bg-zinc-950">
                  <div className="flex items-center justify-between text-[11px] uppercase tracking-wide text-emerald-700 dark:text-emerald-300">
                    <span>Underdog payout</span>
                    <span className="rounded-full border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 text-[9px] font-mono normal-case dark:border-emerald-800 dark:bg-emerald-950/40">
                      {parlayAnalytics.usingOverride ? "your slip" : "standard table"}
                    </span>
                  </div>
                  <div className="mt-1 flex items-baseline gap-2">
                    <div className="font-mono text-2xl font-semibold">
                      {parlayAnalytics.decimalOdds.toFixed(2)}x
                    </div>
                    <div className="text-[10px] font-mono text-zinc-500">
                      {parlayAnalytics.americanOdds}
                    </div>
                  </div>
                  <div className="mt-1 text-[10px] text-zinc-500">
                    Standard {parlayAnalytics.n}-pick: {parlayAnalytics.standardPayout.toFixed(2)}x
                    {" · "}
                    Per-leg product: {parlayAnalytics.legProductOdds.toFixed(2)}x
                  </div>
                </div>

                <div className="rounded-md border border-emerald-200 bg-white p-3 dark:border-emerald-800 dark:bg-zinc-950">
                  <label className="block text-[11px] uppercase tracking-wide text-emerald-700 dark:text-emerald-300">
                    Override with your Underdog slip multiplier
                  </label>
                  <div className="mt-1 flex items-center gap-2">
                    <input
                      type="text"
                      inputMode="decimal"
                      placeholder={`e.g. ${parlayAnalytics.standardPayout}`}
                      className="h-9 w-28 rounded-md border border-emerald-200 bg-white px-2 font-mono text-sm shadow-sm outline-none focus:ring-2 focus:ring-emerald-300 dark:border-emerald-800 dark:bg-zinc-950 dark:focus:ring-emerald-700"
                      value={parlayPayoutInput}
                      onChange={(e) => setParlayPayoutInput(e.target.value)}
                      onBlur={(e) => {
                        const v = Number(e.target.value.replace(/[^0-9.]/g, ""));
                        if (isFinite(v) && v > 1) {
                          setParlayPayoutOverride(v);
                          setParlayPayoutInput(String(v));
                        } else {
                          setParlayPayoutOverride(null);
                          setParlayPayoutInput("");
                        }
                      }}
                      onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                    />
                    <span className="text-xs text-zinc-500">x</span>
                    {parlayAnalytics.usingOverride && (
                      <button
                        className="text-[10px] text-rose-500 hover:underline"
                        onClick={() => { setParlayPayoutOverride(null); setParlayPayoutInput(""); }}
                      >
                        clear
                      </button>
                    )}
                  </div>
                  <div className="mt-1 text-[10px] leading-4 text-zinc-500">
                    Underdog uses different schedules for Standard, Insured, Power, and Champions entries. If your actual slip shows a different number, type it here and Kelly sizing will use it.
                  </div>
                </div>
              </div>

              {/* Top-line metrics */}
              <div className="mt-3 grid grid-cols-2 gap-3 text-xs sm:grid-cols-3">
                <div>
                  <div className="text-emerald-700 dark:text-emerald-300" title="Joint model probability assuming independence">
                    Model prob
                  </div>
                  <div className="font-mono font-semibold">{fmtPct(parlayAnalytics.jointModelProb)}</div>
                  <div className="text-[10px] text-zinc-500 font-mono" title="No-vig market estimate (de-juiced)">
                    no-vig {fmtPct(parlayAnalytics.jointNoVigProb)}
                  </div>
                </div>
                <div>
                  <div className="text-emerald-700 dark:text-emerald-300" title="Model prob − implied prob from combined odds">
                    Edge
                  </div>
                  <div className={`font-mono font-semibold ${edgeColor(parlayAnalytics.edge)}`}>
                    {fmtPct(parlayAnalytics.edge)}
                  </div>
                  <div className="text-[10px] text-zinc-500 font-mono" title="Expected profit per $1 staked">
                    EV/$ {parlayAnalytics.evPerDollar >= 0 ? "+" : ""}{parlayAnalytics.evPerDollar.toFixed(3)}
                  </div>
                </div>
                <div>
                  <div className="text-emerald-700 dark:text-emerald-300" title="Full Kelly fraction f* = (b·p − q)/b">
                    Full Kelly
                  </div>
                  <div className={`font-mono font-semibold ${parlayAnalytics.kellyFull > 0 ? "text-emerald-700 dark:text-emerald-300" : "text-rose-600 dark:text-rose-400"}`}>
                    {(parlayAnalytics.kellyFull * 100).toFixed(2)}%
                  </div>
                  <div className="text-[10px] text-zinc-500 font-mono">
                    1/{kellyDivisor} Kelly: {(parlayAnalytics.kellyUsed * 100).toFixed(2)}%
                  </div>
                </div>
              </div>

              {/* Recommended stake */}
              <div className="mt-3 rounded-lg border border-emerald-300 bg-white p-4 dark:border-emerald-700 dark:bg-zinc-950">
                {parlayAnalytics.hasNegativeEdge ? (
                  <div>
                    <div className="text-xs font-semibold text-rose-700 dark:text-rose-400">
                      Negative-EV parlay — Kelly says don&apos;t bet
                    </div>
                    <div className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
                      The joint model probability ({fmtPct(parlayAnalytics.jointModelProb)}) is below the combined implied
                      probability ({fmtPct(parlayAnalytics.impliedFromOdds)}). Recommended stake: <span className="font-mono font-semibold">$0.00</span>.
                      Drop a low-edge leg or wait for a better price.
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-wrap items-end justify-between gap-3">
                    <div>
                      <div className="text-[11px] uppercase tracking-wide text-emerald-700 dark:text-emerald-300">
                        Recommended stake (1/{kellyDivisor} Kelly)
                      </div>
                      <div className="mt-0.5 font-mono text-2xl font-semibold text-emerald-900 dark:text-emerald-100">
                        {fmtMoney(parlayAnalytics.recommendedStake)}
                      </div>
                      <div className="text-[10px] text-zinc-500 font-mono">
                        {(parlayAnalytics.kellyUsed * 100).toFixed(2)}% of {fmtMoney(bankroll)} bankroll
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
                      <div className="text-zinc-500">If wins:</div>
                      <div className="font-mono text-right text-emerald-700 dark:text-emerald-300">
                        +{fmtMoney(parlayAnalytics.potentialProfit)}
                      </div>
                      <div className="text-zinc-500">Total payout:</div>
                      <div className="font-mono text-right">{fmtMoney(parlayAnalytics.potentialPayout)}</div>
                      <div className="text-zinc-500">Expected profit:</div>
                      <div className={`font-mono text-right ${parlayAnalytics.expectedProfit >= 0 ? "text-emerald-700 dark:text-emerald-300" : "text-rose-600 dark:text-rose-400"}`}>
                        {parlayAnalytics.expectedProfit >= 0 ? "+" : ""}{fmtMoney(parlayAnalytics.expectedProfit)}
                      </div>
                    </div>
                  </div>
                )}
                <div className="mt-3 text-[10px] leading-4 text-zinc-500 dark:text-zinc-400">
                  Sizing follows Kelly (1956) <span className="font-mono">f* = (b·p − q)/b</span>, scaled by 1/{kellyDivisor}.
                  Quarter-Kelly captures ~44% of optimal log-growth at ~6% of full-Kelly variance — the standard for risk-aware bettors.
                  Joint probability assumes independence; correlation warnings below should reduce your stake further.
                  Payout uses {parlayAnalytics.usingOverride ? (
                    <>your <span className="font-semibold">slip override ({parlayAnalytics.decimalOdds.toFixed(2)}x)</span></>
                  ) : parlayAnalytics.hasSportsbookPricing ? (
                    <>true <span className="font-semibold">decimal-odds product ({parlayAnalytics.decimalOdds.toFixed(2)}x)</span> — slip contains a non-Pick&apos;em market so flat payout tables don&apos;t apply</>
                  ) : (
                    <>Underdog&apos;s <span className="font-semibold">standard {parlayAnalytics.n}-pick payout ({parlayAnalytics.standardPayout.toFixed(2)}x)</span></>
                  )} — override above if your slip differs.
                </div>
              </div>

              {parlayCorrelationWarnings.length > 0 && (
                <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-300">
                  <div className="font-semibold">Correlation warnings (joint prob will overstate true probability):</div>
                  <ul className="mt-1 list-disc pl-4 space-y-0.5">
                    {parlayCorrelationWarnings.map((w, i) => <li key={i}>{w}</li>)}
                  </ul>
                </div>
              )}

              {/* Portfolio Kelly readout — shows what each leg's recommended
                  single-bet stake would be at the current bankroll, plus the
                  total stake if the user split it across straight bets.
                  Useful comparison vs. parlaying. */}
              {(() => {
                const perLegKelly = parlayProps.map((p) => ({
                  id: p.underdog_option_id,
                  player: p.player_name,
                  pick: fmtPick(p),
                  kelly: kellyFullFraction(p.model_prob ?? 0.5, p.decimal_price ?? 1.81),
                }));
                const totalQuarterKelly = perLegKelly.reduce(
                  (s, l) => s + Math.max(0, l.kelly / Math.max(1, kellyDivisor)),
                  0,
                );
                const positiveLegs = perLegKelly.filter((l) => l.kelly > 0).length;
                if (perLegKelly.length === 0) return null;
                return (
                  <details className="mt-3 rounded-md border border-emerald-100 bg-white p-3 dark:border-emerald-900/40 dark:bg-zinc-950">
                    <summary className="cursor-pointer text-[11px] uppercase tracking-wide text-emerald-700 dark:text-emerald-300">
                      Portfolio Kelly view ({positiveLegs}/{perLegKelly.length} positive-edge legs)
                    </summary>
                    <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
                      <div className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs dark:border-emerald-900/40 dark:bg-emerald-950/30">
                        <div className="font-semibold">If played as straight bets (1/{kellyDivisor} Kelly each):</div>
                        <div className="mt-1 font-mono">
                          Total stake: {fmtMoney(bankroll * totalQuarterKelly)}{" "}
                          ({(totalQuarterKelly * 100).toFixed(2)}% of bankroll)
                        </div>
                      </div>
                      <div className="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-xs dark:border-zinc-800 dark:bg-zinc-900/40">
                        <div className="font-semibold">Parlay stake (current slip):</div>
                        <div className="mt-1 font-mono">
                          {fmtMoney(parlayAnalytics.recommendedStake)}{" "}
                          ({(parlayAnalytics.kellyUsed * 100).toFixed(2)}% of bankroll)
                        </div>
                      </div>
                    </div>
                    <ul className="mt-2 max-h-[160px] space-y-0.5 overflow-auto text-[11px] font-mono">
                      {perLegKelly.map((l) => (
                        <li key={l.id} className="flex items-center justify-between">
                          <span className="text-zinc-600 dark:text-zinc-400 truncate">
                            {l.player}: <span className="text-zinc-500">{l.pick}</span>
                          </span>
                          <span className={l.kelly > 0 ? "text-emerald-700 dark:text-emerald-300" : "text-rose-600 dark:text-rose-400"}>
                            f* {(l.kelly * 100).toFixed(2)}%
                          </span>
                        </li>
                      ))}
                    </ul>
                  </details>
                );
              })()}
            </div>
          )}

          {/* --- Learning Log Section ---
              Replaces the old multi-tab Learning Mode UI. Shows a unified
              time-ordered feed of every model event (calibration runs,
              tier-model upgrades, miss discoveries, weekly reports,
              resolved-pick batches) so you can see what changed and when. */}
          {showLearning && (
            <div className="mt-6 rounded-xl border border-violet-200 bg-white p-5 shadow-sm dark:border-violet-800/60 dark:bg-zinc-950">
              <div className="flex items-center justify-between">
                <h3 className="text-base font-semibold text-violet-900 dark:text-violet-200">Learning Log</h3>
                <div className="flex items-center gap-3">
                  <button
                    className="h-8 rounded-md bg-violet-600 px-4 text-xs font-medium text-white shadow-sm hover:bg-violet-700 disabled:opacity-50 dark:bg-violet-500 dark:hover:bg-violet-400"
                    onClick={() => void runLearningPipeline()}
                    disabled={learningLoading}
                  >
                    {learningLoading ? "Analyzing…" : "Run Analysis"}
                  </button>
                  <button
                    className="text-xs text-zinc-500 hover:underline"
                    onClick={() => setShowLearning(false)}
                  >
                    Close
                  </button>
                </div>
              </div>

              {learningLoading && (
                <div className="mt-3 flex items-center gap-2 text-sm text-violet-600 dark:text-violet-300">
                  <span className="relative flex h-2.5 w-2.5">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-violet-500 opacity-75" />
                    <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-violet-600" />
                  </span>
                  {learningStatus || "Starting..."}
                </div>
              )}

              {learningError && (
                <div className="mt-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-900 dark:border-red-800 dark:bg-red-950/30 dark:text-red-200">
                  {learningError}
                </div>
              )}

              {learningStatus && !learningLoading && (
                <div className="mt-3 rounded-md border border-violet-200 bg-violet-50 p-3 text-sm text-violet-800 dark:border-violet-800/50 dark:bg-violet-950/30 dark:text-violet-200">
                  {learningStatus}
                </div>
              )}

              {/* Compact stats strip — total resolved + hit rate + CLV.
                  Replaces the old "Overview" tab. Pure read-out, no charts.
                  Charts live below the log. */}
              {(() => {
                const resolved = learningEntries.filter((e) => e.resolved === 1);
                const hits = resolved.filter((e) => e.hit === 1).length;
                const total = resolved.length;
                const rate = total ? hits / total : 0;
                const withClv = resolved.filter((e) => typeof e.clv_cents === "number");
                const avgClv = withClv.length
                  ? withClv.reduce((s, e) => s + (e.clv_cents || 0), 0) / withClv.length
                  : null;
                return (
                  <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <div className="rounded-lg border border-zinc-100 bg-zinc-50/50 p-3 dark:border-zinc-800 dark:bg-zinc-900/30">
                      <div className="text-[10px] uppercase tracking-wide text-zinc-500">Resolved picks</div>
                      <div className="mt-1 font-mono text-xl font-bold">{total}</div>
                    </div>
                    <div className="rounded-lg border border-emerald-100 bg-emerald-50/50 p-3 dark:border-emerald-800/40 dark:bg-emerald-950/20">
                      <div className="text-[10px] uppercase tracking-wide text-emerald-600 dark:text-emerald-400">Hit rate</div>
                      <div className="mt-1 font-mono text-xl font-bold text-emerald-700 dark:text-emerald-300">
                        {(rate * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="rounded-lg border border-violet-100 bg-violet-50/50 p-3 dark:border-violet-800/40 dark:bg-violet-950/20">
                      <div className="text-[10px] uppercase tracking-wide text-violet-600 dark:text-violet-400">Log events</div>
                      <div className="mt-1 font-mono text-xl font-bold text-violet-700 dark:text-violet-300">
                        {learningLog.length}
                      </div>
                    </div>
                    <div className="rounded-lg border border-zinc-100 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-950">
                      <div className="text-[10px] uppercase tracking-wide text-zinc-500">Avg CLV</div>
                      <div className={`mt-1 font-mono text-xl font-bold ${
                        avgClv !== null && avgClv >= 0 ? "text-emerald-700 dark:text-emerald-300" : avgClv !== null ? "text-rose-600" : "text-zinc-400"
                      }`}>
                        {avgClv === null ? "—" : `${avgClv >= 0 ? "+" : ""}${avgClv.toFixed(2)}¢`}
                      </div>
                    </div>
                  </div>
                );
              })()}

              {/* Filter chips — narrow the log to one event source */}
              <div className="mt-4 flex flex-wrap items-center gap-2 text-xs">
                <span className="text-zinc-500">Filter:</span>
                {([
                  { id: "all", label: "All", count: learningLog.length },
                  { id: "calibration_run", label: "Calibration", count: learningLogTotals.calibration_run ?? 0 },
                  { id: "tier_train", label: "Tier model", count: learningLogTotals.tier_train ?? 0 },
                  { id: "miss_discovery", label: "Discoveries", count: learningLogTotals.miss_discovery ?? 0 },
                  { id: "resolution_batch", label: "Resolutions", count: learningLogTotals.resolution_batch ?? 0 },
                  { id: "weekly_report", label: "Reports", count: learningLogTotals.weekly_report ?? 0 },
                ] as const).map((opt) => {
                  const active = logFilter === opt.id;
                  return (
                    <button
                      key={opt.id}
                      onClick={() => setLogFilter(opt.id as typeof logFilter)}
                      className={
                        "rounded-full border px-2.5 py-1 transition-colors " +
                        (active
                          ? "border-violet-500 bg-violet-100 text-violet-800 dark:border-violet-400 dark:bg-violet-900/40 dark:text-violet-200"
                          : "border-zinc-200 bg-white text-zinc-600 hover:border-zinc-300 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-400 dark:hover:bg-zinc-900")
                      }
                    >
                      {opt.label} <span className="opacity-60">({opt.count})</span>
                    </button>
                  );
                })}
                <button
                  onClick={() => void refreshLearningLog()}
                  className="ml-auto rounded-md border border-zinc-200 bg-white px-2.5 py-1 text-zinc-600 hover:border-zinc-300 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-400 dark:hover:bg-zinc-900"
                  title="Re-fetch the model event log"
                >
                  Refresh
                </button>
              </div>

              {/* The unified time-ordered model-event feed. */}
              {(() => {
                const filtered = logFilter === "all"
                  ? learningLog
                  : learningLog.filter((e) => e.kind === logFilter);
                if (filtered.length === 0) {
                  return (
                    <div className="mt-4 rounded-md border border-zinc-200 bg-zinc-50 p-4 text-sm text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900/30 dark:text-zinc-400">
                      <div className="font-medium">No model events yet.</div>
                      <div className="mt-2 text-xs">
                        The continuous-learning loops populate this feed as they run.
                        Calibration cycles, tier-model retrains, miss discoveries, and
                        weekly reports show up here automatically.
                      </div>
                    </div>
                  );
                }
                const KIND_META: Record<LearningLogKind, { label: string; color: string }> = {
                  calibration_run: { label: "CALIBRATION", color: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200 border-emerald-200 dark:border-emerald-800/50" },
                  tier_train: { label: "TIER MODEL", color: "bg-violet-100 text-violet-800 dark:bg-violet-900/40 dark:text-violet-200 border-violet-200 dark:border-violet-800/50" },
                  miss_discovery: { label: "DISCOVERY", color: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200 border-amber-200 dark:border-amber-800/50" },
                  resolution_batch: { label: "RESOLVED", color: "bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-200 border-sky-200 dark:border-sky-800/50" },
                  weekly_report: { label: "REPORT", color: "bg-fuchsia-100 text-fuchsia-800 dark:bg-fuchsia-900/40 dark:text-fuchsia-200 border-fuchsia-200 dark:border-fuchsia-800/50" },
                };
                const STATUS_DOT: Record<string, string> = {
                  adopted: "bg-emerald-500",
                  rejected: "bg-rose-500",
                  skipped: "bg-zinc-400",
                  info: "bg-sky-500",
                };
                return (
                  <div className="mt-3 space-y-2">
                    {filtered.map((ev) => {
                      const meta = KIND_META[ev.kind];
                      const isOpen = expandedLogIds.has(ev.id);
                      return (
                        <div
                          key={ev.id}
                          className="rounded-lg border border-zinc-200 bg-white p-3 shadow-sm dark:border-zinc-800 dark:bg-zinc-950"
                        >
                          <button
                            type="button"
                            onClick={() => {
                              setExpandedLogIds((prev) => {
                                const next = new Set(prev);
                                if (next.has(ev.id)) next.delete(ev.id);
                                else next.add(ev.id);
                                return next;
                              });
                            }}
                            className="flex w-full items-start justify-between gap-3 text-left"
                          >
                            <div className="flex flex-1 items-start gap-2 min-w-0">
                              <span
                                className={`mt-1 inline-block h-2.5 w-2.5 flex-shrink-0 rounded-full ${STATUS_DOT[ev.status] || STATUS_DOT.info}`}
                                title={`status: ${ev.status}`}
                              />
                              <div className="min-w-0 flex-1">
                                <div className="flex flex-wrap items-center gap-2">
                                  <span
                                    className={`rounded-full border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${meta.color}`}
                                  >
                                    {meta.label}
                                  </span>
                                  <span className="text-sm font-medium text-zinc-800 dark:text-zinc-100 truncate">
                                    {ev.title}
                                  </span>
                                </div>
                                {ev.summary && (
                                  <div className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
                                    {ev.summary}
                                  </div>
                                )}
                              </div>
                            </div>
                            <div className="flex flex-col items-end gap-1 text-[10px] text-zinc-500 dark:text-zinc-400">
                              <span title={ev.timestamp || ""}>{fmtTimeAgo(ev.timestamp)}</span>
                              <span className={`transition-transform ${isOpen ? "rotate-90" : ""}`}>▶</span>
                            </div>
                          </button>

                          {isOpen && ev.details && (
                            <div className="mt-3 rounded-md border border-zinc-100 bg-zinc-50 p-3 text-xs dark:border-zinc-800 dark:bg-zinc-900/40">
                              {/* Tier train — show metric delta and the raw fields */}
                              {ev.kind === "tier_train" && (
                                <div className="space-y-1">
                                  {Object.entries(ev.details as Record<string, unknown>).map(([k, v]) => (
                                    <div key={k} className="flex justify-between gap-4">
                                      <span className="text-zinc-500">{k}</span>
                                      <span className="font-mono text-zinc-800 dark:text-zinc-200 text-right break-all">
                                        {typeof v === "object" ? JSON.stringify(v) : String(v)}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              )}

                              {/* Calibration run details */}
                              {ev.kind === "calibration_run" && (() => {
                                const d = ev.details as Record<string, any>;
                                return (
                                  <div className="space-y-2">
                                    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                                      {Object.entries(d.metrics || {}).map(([k, v]) => (
                                        <div key={k} className="rounded border border-zinc-200 bg-white px-2 py-1 dark:border-zinc-800 dark:bg-zinc-950">
                                          <div className="text-[10px] uppercase text-zinc-500">{k.replace(/_/g, " ")}</div>
                                          <div className="font-mono text-zinc-800 dark:text-zinc-200">
                                            {v === null || v === undefined ? "—" : typeof v === "number" ? v.toFixed(4) : String(v)}
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                    {d.params && (
                                      <details className="mt-2">
                                        <summary className="cursor-pointer text-zinc-500">Calibrated parameters</summary>
                                        <pre className="mt-1 overflow-auto rounded bg-zinc-100 p-2 text-[10px] text-zinc-800 dark:bg-zinc-900 dark:text-zinc-300">
                                          {JSON.stringify(d.params, null, 2)}
                                        </pre>
                                      </details>
                                    )}
                                  </div>
                                );
                              })()}

                              {/* Resolution batch — show sample */}
                              {ev.kind === "resolution_batch" && (() => {
                                const d = ev.details as Record<string, any>;
                                const samples: any[] = Array.isArray(d.sample_picks) ? d.sample_picks : [];
                                return (
                                  <div className="space-y-2">
                                    <div className="text-zinc-600 dark:text-zinc-400">
                                      {d.count} picks · {d.hits} hits · {d.misses} misses
                                    </div>
                                    {samples.length > 0 && (
                                      <ul className="space-y-1 max-h-[200px] overflow-auto">
                                        {samples.map((s, i) => (
                                          <li
                                            key={i}
                                            className={`flex items-center justify-between rounded px-2 py-1 ${
                                              s.hit === 1
                                                ? "bg-emerald-50 dark:bg-emerald-950/20"
                                                : "bg-rose-50 dark:bg-rose-950/20"
                                            }`}
                                          >
                                            <span className="truncate">
                                              <span className="font-semibold">{s.player_name}</span>{" "}
                                              <span className="text-zinc-500">
                                                {s.side?.toUpperCase()} {s.line} {s.stat}
                                              </span>
                                            </span>
                                            <span className="font-mono text-[10px] text-zinc-600 dark:text-zinc-400">
                                              {s.actual_value ?? "?"} ({s.sport})
                                            </span>
                                          </li>
                                        ))}
                                      </ul>
                                    )}
                                  </div>
                                );
                              })()}

                              {/* Miss discovery — categories + lessons */}
                              {ev.kind === "miss_discovery" && (() => {
                                const d = ev.details as Record<string, any>;
                                const cats: Record<string, number> = d.categories || {};
                                const lessons: string[] = Array.isArray(d.lessons) ? d.lessons : [];
                                const samples: any[] = Array.isArray(d.sample_misses) ? d.sample_misses : [];
                                return (
                                  <div className="space-y-2">
                                    <div className="flex flex-wrap gap-1">
                                      {Object.entries(cats)
                                        .sort(([, a], [, b]) => (b as number) - (a as number))
                                        .map(([cat, cnt]) => (
                                          <span
                                            key={cat}
                                            className="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] text-amber-800 dark:border-amber-800/50 dark:bg-amber-950/30 dark:text-amber-300"
                                          >
                                            {cat.replace(/_/g, " ")}: {cnt as number}
                                          </span>
                                        ))}
                                    </div>
                                    {lessons.length > 0 && (
                                      <div>
                                        <div className="font-semibold text-zinc-700 dark:text-zinc-300 mb-1">Lessons:</div>
                                        <ul className="list-disc pl-4 space-y-0.5 text-zinc-700 dark:text-zinc-300">
                                          {lessons.map((l, i) => (
                                            <li key={i}>{l}</li>
                                          ))}
                                        </ul>
                                      </div>
                                    )}
                                    {samples.length > 0 && (
                                      <details>
                                        <summary className="cursor-pointer text-zinc-500">Sample misses ({samples.length})</summary>
                                        <ul className="mt-1 space-y-0.5">
                                          {samples.map((s, i) => (
                                            <li key={i} className="text-[10px] font-mono text-zinc-600 dark:text-zinc-400">
                                              {s.player_name} · {s.side?.toUpperCase()} {s.line} {s.stat} · actual {s.actual_value ?? "?"} · {s.miss_category}
                                            </li>
                                          ))}
                                        </ul>
                                      </details>
                                    )}
                                  </div>
                                );
                              })()}

                              {/* Weekly report */}
                              {ev.kind === "weekly_report" && (() => {
                                const d = ev.details as Record<string, any>;
                                return (
                                  <div className="space-y-2">
                                    <div className="text-zinc-600 dark:text-zinc-400">
                                      {d.week_start?.slice(0, 10)} – {d.week_end?.slice(0, 10)} · {d.total_picks} picks · {d.hits}/{d.total_picks} hits
                                    </div>
                                    {d.biggest_blind_spot && (
                                      <div className="rounded border border-rose-200 bg-rose-50 p-2 dark:border-rose-800/50 dark:bg-rose-950/30">
                                        <div className="text-[10px] font-bold text-rose-700 dark:text-rose-400">BIGGEST BLIND SPOT</div>
                                        <div className="mt-1 text-zinc-800 dark:text-zinc-200">{d.biggest_blind_spot}</div>
                                      </div>
                                    )}
                                    {Array.isArray(d.stat_model_suggestions) && d.stat_model_suggestions.length > 0 && (
                                      <div>
                                        <div className="font-semibold text-amber-700 dark:text-amber-400">Stat model:</div>
                                        <ul className="list-disc pl-4 space-y-0.5">
                                          {d.stat_model_suggestions.map((s: string, i: number) => (
                                            <li key={i}>{s}</li>
                                          ))}
                                        </ul>
                                      </div>
                                    )}
                                    {Array.isArray(d.ai_prompt_suggestions) && d.ai_prompt_suggestions.length > 0 && (
                                      <div>
                                        <div className="font-semibold text-blue-700 dark:text-blue-400">AI prompt:</div>
                                        <ul className="list-disc pl-4 space-y-0.5">
                                          {d.ai_prompt_suggestions.map((s: string, i: number) => (
                                            <li key={i}>{s}</li>
                                          ))}
                                        </ul>
                                      </div>
                                    )}
                                    {Array.isArray(d.general_insights) && d.general_insights.length > 0 && (
                                      <div>
                                        <div className="font-semibold text-zinc-700 dark:text-zinc-300">Insights:</div>
                                        <ul className="list-disc pl-4 space-y-0.5">
                                          {d.general_insights.map((s: string, i: number) => (
                                            <li key={i}>{s}</li>
                                          ))}
                                        </ul>
                                      </div>
                                    )}
                                  </div>
                                );
                              })()}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                );
              })()}

              {/* Calibration + bankroll visualizations — kept as the only
                  numeric panels under the log because they're the
                  highest-signal "is this getting better" charts. */}
              {learningEntries.filter((e) => e.resolved === 1).length > 0 && (
                <div className="mt-6 grid gap-4 md:grid-cols-2">
                  <div className="rounded-lg border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-950">
                    <ReliabilityDiagram entries={learningEntries} />
                  </div>
                  <div className="rounded-lg border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-950">
                    <BankrollGrowth entries={learningEntries} />
                  </div>
                </div>
              )}

            </div>
          )}

          {/* --- History Section --- */}
          {showHistory && (
            <div className="mt-6 rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold">Prediction History</h3>
                <button
                  className="text-xs text-zinc-500 hover:underline"
                  onClick={() => setShowHistory(false)}
                >
                  Close
                </button>
              </div>
              {history.length === 0 ? (
                <div className="mt-3 text-sm text-zinc-500">No prediction history yet.</div>
              ) : (
                <div className="mt-3 space-y-3 max-h-[400px] overflow-auto">
                  {history.map((entry) => (
                    <div key={entry.id} className="rounded-lg border border-zinc-100 bg-zinc-50/50 p-3 dark:border-zinc-800 dark:bg-zinc-900/20">
                      <div className="flex items-center justify-between text-xs text-zinc-500">
                        <span className="font-mono">{new Date(entry.timestamp).toLocaleString()}</span>
                        <span className="font-mono">{entry.sport}</span>
                      </div>
                      <div className="mt-2 space-y-1">
                        {(entry.props ?? []).slice(0, 5).map((p: any, i: number) => (
                          <div key={i} className="flex items-center justify-between text-xs">
                            <span>
                              <span className="font-medium">{p.player_name}</span>{" "}
                              <span className="text-zinc-500">{p.side?.toUpperCase()} {p.line} {p.display_stat ?? p.stat}</span>
                            </span>
                            <span className={`font-mono ${edgeColor(p.edge)}`}>{fmtPct(p.edge)}</span>
                          </div>
                        ))}
                        {(entry.props ?? []).length > 5 && (
                          <div className="text-[11px] text-zinc-400">+{(entry.props ?? []).length - 5} more</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

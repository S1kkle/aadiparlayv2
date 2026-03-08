"use client";

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  fetchHistory,
  fetchPropsJobResult,
  fetchRankedProps,
  getBackendUrl,
  recommendParlay,
  saveHistory,
  startPropsJob,
} from "@/lib/api";
import type { HistoryEntry, ParlayRecommendation, Prop, RankedPropsResponse, SportId } from "@/lib/types";

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

  // Parlay builder
  const [parlayIds, setParlayIds] = useState<Set<string>>(new Set());

  // AI parlay recommendation
  const [parlayRec, setParlayRec] = useState<ParlayRecommendation | null>(null);
  const [parlayRecLoading, setParlayRecLoading] = useState(false);

  // Model-only props (shown before AI finishes)
  const [modelProps, setModelProps] = useState<Prop[]>([]);

  // History
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  const toggleParlay = useCallback((id: string) => {
    setParlayIds((prev) => {
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
    setParlayRec(null);
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
        requireAiCount: 10,
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
            try {
              await saveHistory({ sport, props: res.props.slice(0, 10) });
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

  async function loadParlayRec(legs: number) {
    const sourceProps = allProps.length ? allProps : modelProps;
    if (!sourceProps.length) return;
    setParlayRecLoading(true);
    setParlayRec(null);
    try {
      const rec = await recommendParlay({ sport, legs, props: sourceProps.slice(0, 30) });
      setParlayRec(rec);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setParlayRecLoading(false);
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
    try {
      const h = await fetchHistory();
      setHistory(h);
      setShowHistory(true);
    } catch {}
  }

  useEffect(() => {
    void load(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sport]);

  const allProps = data?.props ?? [];
  const displayProps = allProps.length ? allProps : modelProps;
  const aiFinished = !!data;
  const availableStats = useMemo(() => getUniqueStats(displayProps), [displayProps]);

  const hasAi = (p: Prop) => typeof p.ai_summary === "string" && p.ai_summary.trim().length > 0;

  const topProps = useMemo(() => {
    let list = displayProps;
    if (statFilter !== "all") list = list.filter((p) => p.stat === statFilter);
    const withAi = sortProps(list.filter(hasAi), sortKey, sortAsc);
    const seenPlayers = new Set<string>();
    const deduped: Prop[] = [];
    for (const p of withAi) {
      if (seenPlayers.has(p.player_name)) continue;
      seenPlayers.add(p.player_name);
      deduped.push(p);
      if (deduped.length >= 10) break;
    }
    return deduped;
  }, [displayProps, statFilter, sortKey, sortAsc]);

  const remainingProps = useMemo(() => {
    let list = displayProps;
    if (statFilter !== "all") list = list.filter((p) => p.stat === statFilter);
    const topIds = new Set(topProps.map((p) => p.underdog_option_id));
    return sortProps(list.filter((p) => !topIds.has(p.underdog_option_id)), sortKey, sortAsc);
  }, [displayProps, statFilter, sortKey, sortAsc, topProps]);

  const filteredProps = topProps;

  // Parlay computations
  const parlayProps = useMemo(
    () => displayProps.filter((p) => parlayIds.has(p.underdog_option_id)),
    [displayProps, parlayIds]
  );
  const parlayDecimalOdds = useMemo(() => {
    if (!parlayProps.length) return 0;
    return parlayProps.reduce((acc, p) => acc * (p.decimal_price ?? 1), 1);
  }, [parlayProps]);
  const parlayCombinedProb = useMemo(() => {
    if (!parlayProps.length) return 0;
    return parlayProps.reduce((acc, p) => acc * (p.model_prob ?? 0.5), 1);
  }, [parlayProps]);
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
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {data ? (
                  <div className="text-xs text-zinc-600 dark:text-zinc-400">
                    Updated{" "}
                    <span className="font-mono text-[11px]">{data.updated_at}</span>
                  </div>
                ) : null}

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
            <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
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
              <div className="flex flex-wrap items-center gap-2">
                <button
                  className="h-8 rounded-md border border-emerald-300 bg-emerald-50 px-3 text-xs font-medium text-emerald-800 shadow-sm hover:bg-emerald-100 disabled:opacity-50 dark:border-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300 dark:hover:bg-emerald-900/40"
                  onClick={() => void loadParlayRec(2)}
                  disabled={parlayRecLoading || (!allProps.length && !modelProps.length)}
                >
                  {parlayRecLoading ? "Building..." : "Best 2-Leg Parlay"}
                </button>
                <button
                  className="h-8 rounded-md border border-emerald-300 bg-emerald-50 px-3 text-xs font-medium text-emerald-800 shadow-sm hover:bg-emerald-100 disabled:opacity-50 dark:border-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300 dark:hover:bg-emerald-900/40"
                  onClick={() => void loadParlayRec(5)}
                  disabled={parlayRecLoading || (!allProps.length && !modelProps.length)}
                >
                  {parlayRecLoading ? "Building..." : "Best 5-Leg Parlay"}
                </button>
              </div>
            </div>
          ) : null}

          {/* --- AI Parlay Recommendation --- */}
          {parlayRec && (
            <div className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50/60 p-5 shadow-sm dark:border-emerald-800 dark:bg-emerald-950/30">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-emerald-800 dark:text-emerald-200">
                  AI-Recommended {parlayRec.legs}-Leg Parlay
                </h3>
                <button
                  className="text-xs text-emerald-700 hover:underline dark:text-emerald-400"
                  onClick={() => setParlayRec(null)}
                >
                  Dismiss
                </button>
              </div>

              <div className="mt-3 space-y-2">
                {parlayRec.props.map((p, i) => (
                  <div key={p.underdog_option_id ?? i} className="flex items-center justify-between rounded-lg border border-emerald-200 bg-white px-3 py-2 text-sm dark:border-emerald-800 dark:bg-zinc-950">
                    <div className="flex items-center gap-3">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald-100 text-[10px] font-bold text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300">
                        {i + 1}
                      </span>
                      <div>
                        <div className="font-medium">{p.player_name}</div>
                        <div className="text-xs text-zinc-500">
                          {p.side?.toUpperCase()} {p.line} {p.display_stat ?? p.stat}
                          {p.game_title ? ` — ${p.game_title}` : ""}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 text-xs font-mono">
                      <span className={edgeColor(p.edge)}>edge {fmtPct(p.edge)}</span>
                      <span>hit {p.hit_rate_str ?? "?"}</span>
                      <span>model {fmtPct(p.model_prob)}</span>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-4 rounded-lg border border-emerald-200 bg-white p-4 dark:border-emerald-800 dark:bg-zinc-950">
                <div className="text-xs font-semibold text-emerald-700 dark:text-emerald-300 mb-2">AI Parlay Analysis</div>
                <div className="text-sm leading-6 text-zinc-800 dark:text-zinc-200 whitespace-pre-wrap break-words">
                  {parlayRec.parlay_summary || "No summary generated."}
                </div>
                <div className="mt-3 flex flex-wrap gap-2 text-xs">
                  <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 font-mono text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300">
                    Combined prob: {fmtPct(parlayRec.combined_model_prob)}
                  </span>
                  <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 font-mono text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300">
                    AI confidence: {fmtNum(parlayRec.combined_confidence, 2)}
                  </span>
                </div>
                {parlayRec.risk_factors?.length > 0 && (
                  <div className="mt-3">
                    <div className="text-xs font-semibold text-rose-700 dark:text-rose-300">Risks</div>
                    <ul className="mt-1 list-disc pl-4 text-xs text-zinc-700 dark:text-zinc-300 space-y-0.5">
                      {parlayRec.risk_factors.map((r, i) => <li key={i}>{r}</li>)}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

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
                            <div>{p.side.toUpperCase()} {p.line} {p.display_stat ?? ""}</div>
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
                      </div>
                      <div className="mt-1 flex flex-wrap gap-2 text-xs text-zinc-500">
                        <span className="font-mono">{p.sport}</span>
                        <span>{p.side.toUpperCase()} {p.line} {p.display_stat ?? p.stat}</span>
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
                                {p.side.toUpperCase()} {p.line} <span className="text-zinc-500">{p.display_stat ?? p.stat}</span>
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

          {/* --- Parlay Slip --- */}
          {parlayProps.length > 0 && (
            <div className="mt-6 rounded-xl border border-emerald-200 bg-emerald-50/50 p-4 shadow-sm dark:border-emerald-800 dark:bg-emerald-950/30">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-emerald-800 dark:text-emerald-200">
                  Parlay Slip ({parlayProps.length} pick{parlayProps.length > 1 ? "s" : ""})
                </h3>
                <button
                  className="text-xs text-emerald-700 hover:underline dark:text-emerald-400"
                  onClick={() => setParlayIds(new Set())}
                >
                  Clear all
                </button>
              </div>
              <div className="mt-3 space-y-2">
                {parlayProps.map((p) => (
                  <div key={p.underdog_option_id} className="flex items-center justify-between text-xs">
                    <div>
                      <span className="font-medium">{p.player_name}</span>{" "}
                      <span className="text-zinc-500">{p.side.toUpperCase()} {p.line} {p.display_stat ?? p.stat}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-zinc-600 dark:text-zinc-400">
                        {fmtPct(p.model_prob)}
                      </span>
                      <button
                        className="text-rose-500 hover:text-rose-700"
                        onClick={() => toggleParlay(p.underdog_option_id)}
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-3 grid grid-cols-3 gap-3 border-t border-emerald-200 pt-3 text-xs dark:border-emerald-800">
                <div>
                  <div className="text-emerald-600 dark:text-emerald-400">Combined odds</div>
                  <div className="font-mono font-semibold">{parlayDecimalOdds.toFixed(2)}x</div>
                </div>
                <div>
                  <div className="text-emerald-600 dark:text-emerald-400">Combined prob</div>
                  <div className="font-mono font-semibold">{fmtPct(parlayCombinedProb)}</div>
                </div>
                <div>
                  <div className="text-emerald-600 dark:text-emerald-400">Payout ($10)</div>
                  <div className="font-mono font-semibold">${(10 * parlayDecimalOdds).toFixed(2)}</div>
                </div>
              </div>
              {parlayCorrelationWarnings.length > 0 && (
                <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-300">
                  <div className="font-semibold">Correlation warnings:</div>
                  <ul className="mt-1 list-disc pl-4 space-y-0.5">
                    {parlayCorrelationWarnings.map((w, i) => <li key={i}>{w}</li>)}
                  </ul>
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

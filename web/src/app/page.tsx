"use client";

import { Fragment, useEffect, useMemo, useState } from "react";

import { fetchRankedProps, getBackendUrl } from "@/lib/api";
import type { Prop, RankedPropsResponse, SportId } from "@/lib/types";

const SPORT_OPTIONS: { id: SportId; label: string }[] = [
  { id: "UNKNOWN", label: "All sports" },
  { id: "NBA", label: "NBA" },
  { id: "NFL", label: "NFL" },
  { id: "NHL", label: "NHL" },
  { id: "SOCCER", label: "Soccer (experimental)" },
  { id: "MMA", label: "MMA (experimental)" },
];

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
  // most props are integer-ish; keep one decimal only when needed
  return Number.isInteger(x) ? String(x) : x.toFixed(1);
}

function biasLabel(bias: number | null | undefined) {
  if (bias === 1) return "Lean OVER";
  if (bias === -1) return "Lean UNDER";
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

export default function Home() {
  const backendUrl = useMemo(() => getBackendUrl(), []);
  const [sport, setSport] = useState<SportId>("NBA");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<RankedPropsResponse | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  async function load(refresh = false) {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchRankedProps({
        sport,
        scope: "all",
        refresh,
        maxProps: 10,
        aiLimit: 10,
      });
      setData(res);
    } catch (e) {
      setData(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sport]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-50 to-zinc-100 text-zinc-950 dark:from-zinc-950 dark:to-black dark:text-zinc-50">
      <div className="mx-auto max-w-6xl px-6 py-10">
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
              <label className="flex items-center gap-2 text-sm">
                <span className="text-zinc-600 dark:text-zinc-400">Sport</span>
                <select
                  className="h-10 rounded-md border border-zinc-200 bg-white px-3 text-sm shadow-sm outline-none focus:ring-2 focus:ring-zinc-300 dark:border-zinc-800 dark:bg-zinc-950 dark:focus:ring-zinc-700"
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

              <div className="flex items-center gap-3">
                {data ? (
                  <div className="text-xs text-zinc-600 dark:text-zinc-400">
                    Updated{" "}
                    <span className="font-mono text-[11px]">
                      {data.updated_at}
                    </span>
                  </div>
                ) : null}

                <button
                  className="h-10 rounded-md bg-zinc-900 px-4 text-sm font-medium text-white shadow-sm hover:bg-zinc-800 disabled:opacity-50 dark:bg-zinc-50 dark:text-black dark:hover:bg-zinc-200"
                  onClick={() => void load(true)}
                  disabled={loading}
                >
                  {loading ? "Loading…" : "Refresh"}
                </button>
              </div>
            </div>
          </div>
        </header>

        <section className="mt-6">
          {error ? (
            <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-900 dark:border-red-900/40 dark:bg-red-950/40 dark:text-red-200">
              <div className="font-medium">Request failed</div>
              <div className="mt-1 font-mono text-xs whitespace-pre-wrap">
                {error}
              </div>
              <div className="mt-3 text-xs text-red-800/80 dark:text-red-200/80">
                Make sure the FastAPI backend is running on{" "}
                <span className="font-mono">{backendUrl}</span>.
              </div>
            </div>
          ) : null}

          {data ? (
            <div className="mt-4 flex items-center justify-between text-sm text-zinc-600 dark:text-zinc-400">
              <div>
                Scope: <span className="font-mono text-xs">{data.scope}</span>
              </div>
              <div>
                Props:{" "}
                <span className="font-mono text-xs">{data.props.length}</span>
              </div>
            </div>
          ) : null}

          <div className="mt-4 overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
            <div className="overflow-auto">
              <table className="min-w-[1150px] w-full text-sm">
                <thead className="sticky top-0 z-10 bg-white/95 backdrop-blur dark:bg-zinc-950/95">
                  <tr className="border-b border-zinc-200 text-left dark:border-zinc-800">
                    <th className="px-3 py-3"></th>
                    <th className="px-4 py-3">#</th>
                    <th className="px-4 py-3">Sport</th>
                    <th className="px-4 py-3">Player</th>
                    <th className="px-4 py-3">Pick</th>
                    <th className="px-4 py-3">Game</th>
                    <th className="px-4 py-3">AI Summary</th>
                    <th className="px-4 py-3">Implied</th>
                    <th className="px-4 py-3">Model</th>
                    <th className="px-4 py-3">Edge</th>
                    <th className="px-4 py-3">EV</th>
                    <th className="px-4 py-3">Vol</th>
                    <th className="px-4 py-3">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {(data?.props ?? []).slice(0, 10).map((p: Prop, idx) => {
                    const isOpen = !!expanded[p.underdog_option_id];
                    return (
                      <Fragment key={p.underdog_option_id}>
                        <tr
                          className={`border-b border-zinc-100 align-top hover:bg-zinc-50 dark:border-zinc-900 dark:hover:bg-zinc-900/30 ${
                            idx % 2 === 0
                              ? "bg-white dark:bg-zinc-950"
                              : "bg-zinc-50/40 dark:bg-zinc-950"
                          }`}
                        >
                          <td className="px-3 py-3">
                            <button
                              className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-zinc-200 bg-white text-zinc-700 shadow-sm hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200 dark:hover:bg-zinc-900"
                              onClick={() =>
                                setExpanded((prev) => ({
                                  ...prev,
                                  [p.underdog_option_id]:
                                    !prev[p.underdog_option_id],
                                }))
                              }
                              aria-label={isOpen ? "Collapse" : "Expand"}
                            >
                              <span
                                className={`transition-transform ${
                                  isOpen ? "rotate-90" : ""
                                }`}
                              >
                                ▶
                              </span>
                            </button>
                          </td>
                          <td className="px-4 py-3 text-zinc-500">{idx + 1}</td>
                      <td className="px-4 py-3 font-mono text-xs">{p.sport}</td>
                      <td className="px-4 py-3">
                        <div className="font-medium">{p.player_name}</div>
                        <div className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                          {p.stat}
                        </div>
                      </td>
                      <td className="px-4 py-3 font-mono text-xs">
                        {p.side.toUpperCase()} {p.line}{" "}
                        {p.display_stat ?? ""}
                      </td>
                      <td className="px-4 py-3 text-xs text-zinc-600 dark:text-zinc-400">
                        {p.game_title ?? "—"}
                        {p.team_abbr || p.opponent_abbr ? (
                          <div className="mt-1 font-mono text-[11px] text-zinc-500">
                            {p.team_abbr ? p.team_abbr : "?"} vs{" "}
                            {p.opponent_abbr ? p.opponent_abbr : "?"}
                          </div>
                        ) : null}
                        {p.scheduled_at ? (
                          <div className="mt-1 font-mono text-[11px]">
                            {p.scheduled_at}
                          </div>
                        ) : null}
                      </td>
                          <td className="px-4 py-3 text-xs text-zinc-700 dark:text-zinc-300">
                            <div className="leading-5">
                              {shortText(p.ai_summary, 140) ? (
                                shortText(p.ai_summary, 140)
                              ) : (
                                <span className="text-zinc-500">
                                  (waiting on AI / Ollama not running)
                                </span>
                              )}
                            </div>
                            <div className="mt-2 flex flex-wrap gap-2">
                              <span className="rounded-full border border-zinc-200 bg-white px-2 py-0.5 text-[11px] text-zinc-700 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200">
                                {biasLabel(p.ai_bias)}{" "}
                                {p.ai_confidence !== null &&
                                p.ai_confidence !== undefined
                                  ? `(${fmtNum(p.ai_confidence, 2)})`
                                  : ""}
                              </span>
                            </div>
                          </td>
                      <td className="px-4 py-3 font-mono text-xs">
                        {fmtPct(p.implied_prob)}
                      </td>
                      <td className="px-4 py-3 font-mono text-xs">
                        {fmtPct(p.model_prob)}
                      </td>
                      <td className="px-4 py-3 font-mono text-xs">
                        {fmtPct(p.edge)}
                      </td>
                      <td className="px-4 py-3 font-mono text-xs">
                        {fmtNum(p.ev, 4)}
                      </td>
                      <td className="px-4 py-3 font-mono text-xs">
                        {fmtNum(p.volatility, 2)}
                      </td>
                      <td className="px-4 py-3 font-mono text-xs">
                        {fmtNum(p.score, 3)}
                      </td>
                        </tr>

                        {isOpen ? (
                          <tr
                            key={`${p.underdog_option_id}__expanded`}
                            className="border-b border-zinc-100 dark:border-zinc-900"
                          >
                            <td colSpan={13} className="px-4 py-4">
                              <div className="grid gap-4 rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900/20">
                                <div className="grid gap-1">
                                  <div className="text-xs font-semibold text-zinc-600 dark:text-zinc-300">
                                    AI Summary
                                  </div>
                                  <div className="text-sm leading-6 text-zinc-800 dark:text-zinc-200">
                                    {p.ai_summary ? (
                                      p.ai_summary
                                    ) : (
                                      <span className="text-zinc-500">
                                        No AI summary yet (may be skipped by
                                        ai_limit, or Ollama isn’t running).
                                      </span>
                                    )}
                                  </div>
                                  <div className="mt-2 flex flex-wrap gap-2 text-xs">
                                    <span className="rounded-full border border-zinc-200 bg-white px-3 py-1 text-zinc-700 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200">
                                      {biasLabel(p.ai_bias)}{" "}
                                      {p.ai_confidence !== null &&
                                      p.ai_confidence !== undefined
                                        ? `(${fmtNum(p.ai_confidence, 2)})`
                                        : ""}
                                    </span>
                                    <span className="rounded-full border border-zinc-200 bg-white px-3 py-1 font-mono text-zinc-700 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200">
                                      model {fmtPct(p.model_prob)} | implied{" "}
                                      {fmtPct(p.implied_prob)} | edge{" "}
                                      {fmtPct(p.edge)}
                                    </span>
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
                                                  <td className="px-2 py-1 text-right font-mono text-[11px] text-zinc-950 dark:text-zinc-50">
                                                    {fmtVal(g.value)}
                                                  </td>
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
                                                <td colSpan={4} className="px-2 py-3 text-zinc-500">
                                                  No game log available for this stat.
                                                </td>
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
                                                  <td className="px-2 py-1 text-right font-mono text-[11px] text-zinc-950 dark:text-zinc-50">
                                                    {fmtVal(g.value)}
                                                  </td>
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
                                                <td colSpan={3} className="px-2 py-3 text-zinc-500">
                                                  No meetings logged yet.
                                                </td>
                                              </tr>
                                            )}
                                          </tbody>
                                        </table>
                                      </div>
                                    </div>
                                  </div>

                                  <div>
                                    <div className="text-xs font-semibold text-emerald-700 dark:text-emerald-300">
                                      Tailwinds
                                    </div>
                                    {p.ai_tailwinds?.length ? (
                                      <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-zinc-700 dark:text-zinc-300">
                                        {p.ai_tailwinds.slice(0, 8).map((t, i) => (
                                          <li key={i}>{t}</li>
                                        ))}
                                      </ul>
                                    ) : (
                                      <div className="mt-2 text-xs text-zinc-500">
                                        —
                                      </div>
                                    )}
                                  </div>
                                  <div>
                                    <div className="text-xs font-semibold text-rose-700 dark:text-rose-300">
                                      Risk factors
                                    </div>
                                    {p.ai_risk_factors?.length ? (
                                      <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-zinc-700 dark:text-zinc-300">
                                        {p.ai_risk_factors.slice(0, 8).map((t, i) => (
                                          <li key={i}>{t}</li>
                                        ))}
                                      </ul>
                                    ) : (
                                      <div className="mt-2 text-xs text-zinc-500">
                                        —
                                      </div>
                                    )}
                                  </div>
                                  <div>
                                    <div className="text-xs font-semibold text-zinc-600 dark:text-zinc-300">
                                      Notes
                                    </div>
                                    {p.notes?.length ? (
                                      <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-zinc-700 dark:text-zinc-300">
                                        {p.notes.slice(0, 8).map((t, i) => (
                                          <li key={i}>{t}</li>
                                        ))}
                                      </ul>
                                    ) : (
                                      <div className="mt-2 text-xs text-zinc-500">
                                        —
                                      </div>
                                    )}
                                  </div>
                                </div>
                              </div>
                            </td>
                          </tr>
                        ) : null}
                      </Fragment>
                    );
                  })}
                  {!data?.props?.length ? (
                    <tr>
                      <td
                        colSpan={13}
                        className="px-4 py-8 text-center text-sm text-zinc-500"
                      >
                        {loading
                          ? "Loading props…"
                          : "No props returned yet. Check backend config (.env)."}
                      </td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
            <div className="border-t border-zinc-200 px-4 py-3 text-xs text-zinc-600 dark:border-zinc-800 dark:text-zinc-400">
              Showing top 10 picks. AI is requested automatically for all 10 (if Ollama is running).
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

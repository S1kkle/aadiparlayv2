"use client";

/**
 * Bankroll growth chart — cumulative profit over time per resolved learning
 * entry. Plots 1) actual profit when stake_amount/profit fields are populated,
 * 2) hypothetical 1-unit-per-pick growth as a fallback so the chart is useful
 * even before the user starts logging stakes.
 */
import type { LearningEntry } from "@/lib/types";

interface Props {
  entries: LearningEntry[];
  unitProfitOnHit?: number; // hypothetical profit per win (default $0.81 = -110 odds)
  unitLossOnMiss?: number; // hypothetical loss per miss
  startingBankroll?: number;
  className?: string;
}

export function BankrollGrowth({
  entries,
  unitProfitOnHit = 0.81,
  unitLossOnMiss = 1.0,
  startingBankroll = 0,
  className,
}: Props) {
  const sorted = [...entries]
    .filter((e) => e.resolved === 1)
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp));

  if (sorted.length === 0) {
    return (
      <div className={className}>
        <div className="text-xs font-semibold text-zinc-700 dark:text-zinc-200">
          Bankroll growth
        </div>
        <div className="mt-2 rounded border border-dashed border-zinc-300 p-4 text-xs text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
          No resolved picks yet — chart will populate once outcomes settle.
        </div>
      </div>
    );
  }

  const series: { ts: string; cum: number; cumHypothetical: number }[] = [];
  let cum = startingBankroll;
  let cumHyp = startingBankroll;
  for (const e of sorted) {
    if (typeof e.profit === "number" && Number.isFinite(e.profit)) {
      cum += e.profit;
    } else {
      cum += e.hit === 1 ? unitProfitOnHit : -unitLossOnMiss;
    }
    cumHyp += e.hit === 1 ? unitProfitOnHit : -unitLossOnMiss;
    series.push({ ts: e.timestamp, cum, cumHypothetical: cumHyp });
  }

  const W = 320;
  const H = 200;
  const padL = 40;
  const padR = 10;
  const padT = 16;
  const padB = 24;

  const allValues = [
    startingBankroll,
    ...series.map((p) => p.cum),
    ...series.map((p) => p.cumHypothetical),
  ];
  const minY = Math.min(...allValues);
  const maxY = Math.max(...allValues);
  const yPad = (maxY - minY) * 0.1 || 1;
  const yLo = minY - yPad;
  const yHi = maxY + yPad;
  const yRange = yHi - yLo || 1;

  const x = (i: number) => padL + (i / Math.max(1, series.length - 1)) * (W - padL - padR);
  const y = (v: number) => padT + (1 - (v - yLo) / yRange) * (H - padT - padB);

  const path = (key: "cum" | "cumHypothetical") =>
    series.map((p, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(p[key])}`).join(" ");

  const final = series[series.length - 1];
  const finalCum = final.cum;
  const finalHyp = final.cumHypothetical;
  const showProfit = sorted.some((e) => typeof e.profit === "number");

  return (
    <div className={className}>
      <div className="flex items-baseline justify-between">
        <div className="text-xs font-semibold text-zinc-700 dark:text-zinc-200">
          Bankroll growth ({sorted.length} resolved)
        </div>
        <div className="text-[11px] font-mono text-zinc-500">
          {showProfit ? `actual: ${finalCum >= 0 ? "+" : ""}${finalCum.toFixed(2)}` : "hypothetical 1u/pick"}
          {" · "}
          theory: {finalHyp >= 0 ? "+" : ""}{finalHyp.toFixed(2)}
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Bankroll growth" className="w-full h-auto">
        <line x1={padL} y1={y(0)} x2={W - padR} y2={y(0)} stroke="currentColor" opacity={0.3} strokeDasharray="3 3" />
        {showProfit && (
          <path d={path("cum")} fill="none" stroke="rgb(16 185 129)" strokeWidth={2} />
        )}
        <path d={path("cumHypothetical")} fill="none" stroke="rgb(99 102 241)" strokeWidth={1.4} opacity={showProfit ? 0.6 : 1} strokeDasharray={showProfit ? "0" : "0"} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="currentColor" opacity={0.4} />
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="currentColor" opacity={0.4} />
        <text x={padL - 6} y={y(yLo) + 3} fontSize={9} fill="currentColor" opacity={0.6} textAnchor="end">{yLo.toFixed(1)}</text>
        <text x={padL - 6} y={y(yHi) + 3} fontSize={9} fill="currentColor" opacity={0.6} textAnchor="end">{yHi.toFixed(1)}</text>
      </svg>
    </div>
  );
}

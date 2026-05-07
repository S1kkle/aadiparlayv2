"use client";

/**
 * Reliability diagram — plots model confidence (X) vs observed hit rate (Y)
 * across resolved picks. Perfectly-calibrated models hug the y=x diagonal.
 *
 * Pure SVG so no chart-lib dependency is required.
 */
import type { LearningEntry } from "@/lib/types";

interface Props {
  entries: LearningEntry[];
  bins?: number;
  className?: string;
}

interface Bin {
  binCenter: number;
  count: number;
  hits: number;
  hitRate: number;
}

function computeBins(entries: LearningEntry[], n: number): Bin[] {
  const counts = Array.from({ length: n }, () => ({ count: 0, hits: 0 }));
  for (const e of entries) {
    if (e.resolved !== 1) continue;
    if (e.model_prob === null || e.model_prob === undefined) continue;
    const p = Math.max(0, Math.min(0.9999, e.model_prob));
    const idx = Math.min(n - 1, Math.floor(p * n));
    counts[idx].count += 1;
    counts[idx].hits += e.hit === 1 ? 1 : 0;
  }
  return counts.map((c, i) => ({
    binCenter: (i + 0.5) / n,
    count: c.count,
    hits: c.hits,
    hitRate: c.count ? c.hits / c.count : 0,
  }));
}

export function ReliabilityDiagram({ entries, bins = 10, className }: Props) {
  const data = computeBins(entries, bins);
  const totalResolved = data.reduce((s, b) => s + b.count, 0);
  const W = 320;
  const H = 240;
  const padL = 40;
  const padR = 12;
  const padT = 18;
  const padB = 32;

  const x = (v: number) => padL + v * (W - padL - padR);
  const y = (v: number) => H - padB - v * (H - padT - padB);

  const maxCount = Math.max(1, ...data.map((b) => b.count));
  const barW = (W - padL - padR) / bins;

  return (
    <div className={className}>
      <div className="text-xs font-semibold text-zinc-700 dark:text-zinc-200">
        Reliability ({totalResolved.toLocaleString()} resolved picks)
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Reliability diagram" className="w-full h-auto">
        <rect x={padL} y={padT} width={W - padL - padR} height={H - padT - padB} fill="transparent" stroke="currentColor" opacity={0.08} />
        {/* perfect-calibration diagonal */}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="currentColor" opacity={0.35} strokeDasharray="3 3" />
        {/* bin counts as faded bars (right axis) */}
        {data.map((b, i) => {
          const h = (b.count / maxCount) * (H - padT - padB);
          return (
            <rect
              key={`bar-${i}`}
              x={padL + i * barW + 1}
              y={H - padB - h}
              width={Math.max(1, barW - 2)}
              height={h}
              fill="rgb(160,160,180)"
              opacity={0.18}
            />
          );
        })}
        {/* observed hit-rate dots */}
        {data.map((b, i) =>
          b.count > 0 ? (
            <circle
              key={`dot-${i}`}
              cx={x(b.binCenter)}
              cy={y(b.hitRate)}
              r={Math.min(8, 2 + b.count * 0.5)}
              fill="rgb(16 185 129)"
              opacity={0.85}
              stroke="white"
              strokeWidth={1}
            />
          ) : null
        )}
        {/* axes */}
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="currentColor" opacity={0.4} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="currentColor" opacity={0.4} />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={`tick-${t}`}>
            <text x={x(t)} y={H - padB + 14} fontSize={9} textAnchor="middle" fill="currentColor" opacity={0.6}>{(t * 100).toFixed(0)}%</text>
            <text x={padL - 6} y={y(t) + 3} fontSize={9} textAnchor="end" fill="currentColor" opacity={0.6}>{(t * 100).toFixed(0)}%</text>
          </g>
        ))}
        <text x={(W - padL - padR) / 2 + padL} y={H - 4} fontSize={9} textAnchor="middle" fill="currentColor" opacity={0.6}>
          Predicted probability
        </text>
        <text x={10} y={padT - 6} fontSize={9} fill="currentColor" opacity={0.6}>
          Observed hit rate
        </text>
      </svg>
    </div>
  );
}

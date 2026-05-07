/**
 * Parlay math: EV across Underdog entry types, Kelly sizing, correlation
 * adjustments. Mirrors backend/app/services/underdog_payouts.py.
 */
import {
  EntryType,
  flexPayoutTable,
  insurancePayoutTable,
  standardPayout,
} from "./underdog";

function clampProb(p: number): number {
  if (!isFinite(p)) return 0.5;
  return Math.max(0.001, Math.min(0.999, p));
}

function comb(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  let result = 1;
  for (let i = 0; i < k; i++) {
    result *= n - i;
    result /= i + 1;
  }
  return result;
}

function binomEq(p: number, n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  return comb(n, k) * Math.pow(p, k) * Math.pow(1 - p, n - k);
}

export interface EntryEv {
  entryType: EntryType;
  legs: number;
  evPerDollar: number;
  expectedPayoutMultiplier: number;
  winProbabilityFull: number;
  breakdown: Record<number, number>;
}

/**
 * EV per $1 staked for a given Underdog entry type.
 * `legProbs` length must equal `legs`.
 */
export function entryEv(
  entryType: EntryType,
  legs: number,
  legProbs: number[],
): EntryEv | null {
  if (legProbs.length !== legs) return null;
  const probs = legProbs.map(clampProb);

  if (entryType === "standard") {
    const payout = standardPayout(legs);
    if (payout <= 0) return null;
    const joint = probs.reduce((acc, p) => acc * p, 1);
    return {
      entryType: "standard",
      legs,
      evPerDollar: joint * payout - 1,
      expectedPayoutMultiplier: joint * payout,
      winProbabilityFull: joint,
      breakdown: { [legs]: joint },
    };
  }

  const table =
    entryType === "insurance" ? insurancePayoutTable(legs) : flexPayoutTable(legs);
  if (!table) return null;

  // Use mean leg prob as binomial p — exact when probs are equal, close otherwise.
  const pMean = probs.reduce((a, b) => a + b, 0) / probs.length;
  const jointFull = probs.reduce((acc, p) => acc * p, 1);

  const breakdown: Record<number, number> = {};
  let expectedPayout = 0;
  for (const [winsRequiredStr, mult] of Object.entries(table)) {
    const winsRequired = Number(winsRequiredStr);
    const probAtExact = binomEq(pMean, legs, winsRequired);
    breakdown[winsRequired] = probAtExact;
    expectedPayout += probAtExact * mult;
  }

  return {
    entryType,
    legs,
    evPerDollar: expectedPayout - 1,
    expectedPayoutMultiplier: expectedPayout,
    winProbabilityFull: jointFull,
    breakdown,
  };
}

/** Returns the highest-EV entry-type variant for these legs (or null). */
export function bestEntryType(
  legs: number,
  legProbs: number[],
): EntryEv | null {
  const candidates: EntryEv[] = [];
  for (const et of ["standard", "insurance", "flex"] as EntryType[]) {
    const ev = entryEv(et, legs, legProbs);
    if (ev) candidates.push(ev);
  }
  if (!candidates.length) return null;
  return candidates.reduce(
    (best, cur) => (cur.evPerDollar > best.evPerDollar ? cur : best),
    candidates[0],
  );
}

/** Compute EV for every entry-type variant at this leg count. */
export function allEntryEvs(
  legs: number,
  legProbs: number[],
): EntryEv[] {
  const out: EntryEv[] = [];
  for (const et of ["standard", "insurance", "flex"] as EntryType[]) {
    const ev = entryEv(et, legs, legProbs);
    if (ev) out.push(ev);
  }
  return out;
}

/**
 * Kelly fraction for a single bet. Use a fractional value (e.g. 0.25 ×) in
 * production — full Kelly produces ~60% drawdowns.
 */
export function kellyFullFraction(p: number, decimalOdds: number): number {
  if (!isFinite(p) || !isFinite(decimalOdds)) return 0;
  if (decimalOdds <= 1) return 0;
  const b = decimalOdds - 1;
  const q = 1 - p;
  return (b * p - q) / b;
}

/**
 * Kelly fraction for a parlay treated as a single bet.
 * decimalOdds = entry payout multiplier; jointProb = product of leg probs
 * (or correlation-adjusted product).
 */
export function kellyParlayFraction(
  jointProb: number,
  payoutMultiplier: number,
): number {
  return kellyFullFraction(jointProb, payoutMultiplier);
}

/**
 * Light-touch correlation factor for a parlay. Same player or same game
 * legs are treated as positively correlated; cross-team game legs are
 * treated as ~independent (factor = 1).
 */
export interface CorrelationLeg {
  playerId?: string | null;
  gameId?: string | null;
  teamAbbr?: string | null;
}
export function parlayCorrelationFactor(legs: CorrelationLeg[]): number {
  if (legs.length < 2) return 1;
  let factor = 1;
  for (let i = 0; i < legs.length; i++) {
    for (let j = i + 1; j < legs.length; j++) {
      const a = legs[i];
      const b = legs[j];
      if (a.playerId && b.playerId && a.playerId === b.playerId) factor *= 1.15;
      else if (a.gameId && b.gameId && a.gameId === b.gameId) factor *= 1.05;
      else if (a.teamAbbr && b.teamAbbr && a.teamAbbr === b.teamAbbr) factor *= 1.03;
    }
  }
  // Cap at 1.5 — even highly correlated picks aren't truly redundant.
  return Math.min(1.5, factor);
}

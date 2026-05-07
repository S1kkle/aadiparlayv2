/**
 * Underdog Pick'em payout tables — single source of truth.
 *
 * MUST stay in sync with backend/app/services/underdog_payouts.py.
 *
 * Sources:
 *   - Underdog help center "Pick'em Power Plays" + "Insurance" + "Flex"
 *   - GamedayMath payout-math article
 *   - Occupy Fantasy Pick'em strategy article
 */

export type EntryType = "standard" | "flex" | "insurance";

/** Standard / Power Play payout multipliers (all-or-nothing). */
export const UD_STANDARD_PAYOUTS: Record<number, number> = {
  2: 3,
  3: 6,
  4: 10,
  5: 20,
  6: 37.5,
  7: 75,
  8: 150,
};

/** Insurance: pays a smaller multiplier when one leg loses. */
export const UD_INSURANCE_PAYOUTS: Record<number, Record<number, number>> = {
  3: { 3: 3, 2: 1 },
  4: { 4: 6, 3: 1.5 },
  5: { 5: 10, 4: 2.5 },
};

/** Flex: tolerates 1 (3-5 leg) or 2 (6-8 leg) misses with reduced payouts. */
export const UD_FLEX_PAYOUTS: Record<number, Record<number, number>> = {
  3: { 3: 2.25, 2: 1 },
  4: { 4: 5, 3: 1.5 },
  5: { 5: 10, 4: 2.5 },
  6: { 6: 25, 5: 2, 4: 0.4 },
  7: { 7: 50, 6: 5, 5: 0.5 },
  8: { 8: 100, 7: 10, 6: 1 },
};

export function standardPayout(legs: number): number {
  if (legs <= 1) return 0;
  if (legs in UD_STANDARD_PAYOUTS) return UD_STANDARD_PAYOUTS[legs];
  return Math.pow(2, legs - 1) * 1.5;
}

/** Backwards-compat alias used by older parts of the page. */
export function underdogStandardPayout(legs: number): number {
  return standardPayout(legs);
}

export function insurancePayoutTable(legs: number): Record<number, number> | null {
  return UD_INSURANCE_PAYOUTS[legs] ?? null;
}

export function flexPayoutTable(legs: number): Record<number, number> | null {
  return UD_FLEX_PAYOUTS[legs] ?? null;
}

/** Number of allowed misses for an entry type at a given leg count. */
export function allowedMisses(entryType: EntryType, legs: number): number {
  if (entryType === "standard") return 0;
  if (entryType === "insurance") {
    const t = insurancePayoutTable(legs);
    if (!t) return 0;
    const minWins = Math.min(...Object.keys(t).map(Number));
    return legs - minWins;
  }
  const t = flexPayoutTable(legs);
  if (!t) return 0;
  const minWins = Math.min(...Object.keys(t).map(Number));
  return legs - minWins;
}

export function availableEntryTypes(legs: number): EntryType[] {
  const out: EntryType[] = ["standard"];
  if (insurancePayoutTable(legs)) out.push("insurance");
  if (flexPayoutTable(legs)) out.push("flex");
  return out;
}

import { HistoryEntry, ParlayRecommendation, Prop, RankedPropsResponse, SportId } from "@/lib/types";

const DEFAULT_BACKEND_URL = "http://localhost:8000";

export function getBackendUrl() {
  return process.env.NEXT_PUBLIC_BACKEND_URL || DEFAULT_BACKEND_URL;
}

export async function fetchRankedProps(params: {
  sport: SportId;
  scope?: "all" | "featured";
  refresh?: boolean;
  maxProps?: number;
  aiLimit?: number;
  requireAi?: boolean;
  requireAiCount?: number;
}): Promise<RankedPropsResponse> {
  const backend = getBackendUrl();
  const url = new URL("/props", backend);
  url.searchParams.set("sport", params.sport);
  url.searchParams.set("scope", params.scope ?? "all");
  if (params.refresh) url.searchParams.set("refresh", "true");
  if (params.maxProps) url.searchParams.set("max_props", String(params.maxProps));
  if (params.aiLimit !== undefined) url.searchParams.set("ai_limit", String(params.aiLimit));
  if (params.requireAi) url.searchParams.set("require_ai", "true");
  if (params.requireAiCount) url.searchParams.set("require_ai_count", String(params.requireAiCount));

  const res = await fetch(url.toString(), {
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Backend error ${res.status}: ${text || res.statusText}`);
  }
  return (await res.json()) as RankedPropsResponse;
}

export async function startPropsJob(params: {
  sport: SportId;
  scope?: "all" | "featured";
  refresh?: boolean;
  maxProps?: number;
  aiLimit?: number;
  requireAiCount?: number;
}): Promise<{ job_id: string }> {
  const backend = getBackendUrl();
  const res = await fetch(new URL("/props/job", backend).toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sport: params.sport,
      scope: params.scope ?? "all",
      refresh: !!params.refresh,
      max_props: params.maxProps ?? 10,
      ai_limit: params.aiLimit ?? 10,
      require_ai_count: params.requireAiCount ?? 10,
    }),
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Backend error ${res.status}: ${text || res.statusText}`);
  }
  return (await res.json()) as { job_id: string };
}

export async function fetchPropsJobResult(jobId: string): Promise<RankedPropsResponse> {
  const backend = getBackendUrl();
  const res = await fetch(new URL(`/props/job/${jobId}/result`, backend).toString(), {
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Backend error ${res.status}: ${text || res.statusText}`);
  }
  return (await res.json()) as RankedPropsResponse;
}

export async function fetchHistory(): Promise<HistoryEntry[]> {
  const backend = getBackendUrl();
  const res = await fetch(new URL("/history", backend).toString(), { cache: "no-store" });
  if (!res.ok) return [];
  const data = await res.json();
  return (data?.entries ?? []) as HistoryEntry[];
}

export async function saveHistory(entry: { sport: string; props: any[] }): Promise<void> {
  const backend = getBackendUrl();
  await fetch(new URL("/history", backend).toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(entry),
    cache: "no-store",
  });
}

export async function recommendParlay(params: {
  sport: SportId;
  legs: number;
  props: Prop[];
}): Promise<ParlayRecommendation> {
  const backend = getBackendUrl();
  const res = await fetch(new URL("/parlay/recommend", backend).toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sport: params.sport,
      legs: params.legs,
      props: params.props,
    }),
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Backend error ${res.status}: ${text || res.statusText}`);
  }
  return (await res.json()) as ParlayRecommendation;
}


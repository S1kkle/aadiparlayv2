import { RankedPropsResponse, SportId } from "@/lib/types";

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
}): Promise<RankedPropsResponse> {
  const backend = getBackendUrl();
  const url = new URL("/props", backend);
  url.searchParams.set("sport", params.sport);
  url.searchParams.set("scope", params.scope ?? "all");
  if (params.refresh) url.searchParams.set("refresh", "true");
  if (params.maxProps) url.searchParams.set("max_props", String(params.maxProps));
  if (params.aiLimit !== undefined) url.searchParams.set("ai_limit", String(params.aiLimit));

  const res = await fetch(url.toString(), {
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Backend error ${res.status}: ${text || res.statusText}`);
  }
  return (await res.json()) as RankedPropsResponse;
}


export type SportId = "UNKNOWN" | "NBA" | "NFL" | "NHL" | "SOCCER" | "MMA";

export type PropSide = "over" | "under";

export type Prop = {
  sport: SportId;
  league?: string | null;
  player_name: string;
  player_position?: string | null;
  espn_athlete_id?: number | null;
  underdog_player_id?: string | null;
  underdog_option_id: string;
  game_title?: string | null;
  scheduled_at?: string | null;
  team_abbr?: string | null;
  opponent_abbr?: string | null;

  stat: string;
  display_stat?: string | null;
  line: number;
  side: PropSide;
  stat_field?: string | null;
  recent_games?: { game_date?: string | null; opponent_abbr?: string | null; value: number }[];
  vs_opponent_games?: { game_date?: string | null; opponent_abbr?: string | null; value: number }[];

  // context
  is_home?: boolean | null;
  is_b2b?: boolean | null;
  rest_days?: number | null;
  avg_minutes?: number | null;

  // trend
  trend_short_avg?: number | null;
  trend_direction?: string | null;
  hit_rate_last10?: number | null;
  hit_rate_str?: string | null;

  // odds
  american_price?: number | null;
  decimal_price?: number | null;
  model_prob?: number | null;
  implied_prob?: number | null;
  edge?: number | null;
  ev?: number | null;
  volatility?: number | null;

  // ai
  ai_bias?: number | null;
  ai_confidence?: number | null;
  ai_summary?: string | null;
  ai_tailwinds: string[];
  ai_risk_factors: string[];
  ai_prob_adjustment?: number | null;

  // derived
  confidence_tier?: string | null;
  model_ai_agree?: boolean | null;

  score?: number | null;
  notes: string[];
};

export type RankedPropsResponse = {
  scope: string;
  updated_at: string;
  props: Prop[];
};

export type HistoryEntry = {
  id: string;
  timestamp: string;
  sport: string;
  props: Prop[];
};

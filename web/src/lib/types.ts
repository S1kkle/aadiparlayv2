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
  projected_minutes?: number | null;
  vegas_total?: number | null;
  vegas_spread?: number | null;
  blowout_risk?: boolean | null;

  // trend / profile
  trend_short_avg?: number | null;
  trend_direction?: string | null;
  hit_rate_last10?: number | null;
  hit_rate_str?: string | null;
  stat_median?: number | null;
  stat_floor?: number | null;
  stat_ceiling?: number | null;
  stat_consistency?: number | null;
  current_streak?: number | null;
  line_percentile?: number | null;

  // odds
  american_price?: number | null;
  decimal_price?: number | null;
  payout_multiplier?: number | null;
  selection_subheader?: string | null;
  is_boosted?: boolean | null;
  breakeven_prob?: number | null;
  model_prob?: number | null;
  implied_prob?: number | null;
  no_vig_prob?: number | null;
  edge?: number | null;
  ev?: number | null;
  volatility?: number | null;
  kelly_fraction?: number | null;
  edge_confidence?: number | null;
  per_minute_rate?: number | null;

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

export type ParlayRecommendation = {
  legs: number;
  props: Prop[];
  parlay_summary: string;
  combined_confidence: number;
  risk_factors: string[];
  combined_model_prob: number;
  combined_model_prob_adjusted?: number;
  correlation_factor?: number;
  correlation_notes?: string[];
  entry_payout_multiplier?: number | null;
  ev_independent?: number | null;
  ev_corr_adjusted?: number | null;
};

export type LearningEntry = {
  id: string;
  history_id: string;
  timestamp: string;
  player_name: string;
  sport: string;
  stat: string;
  line: number;
  side: string;
  model_prob: number | null;
  implied_prob: number | null;
  edge: number | null;
  ai_bias: number | null;
  ai_confidence: number | null;
  actual_value: number | null;
  hit: number;
  miss_reason: string | null;
  miss_category: string | null;
  resolved: number;
};

export type LearningReport = {
  id: string;
  week_start: string;
  week_end: string;
  created_at: string;
  total_picks: number;
  hits: number;
  misses: number;
  hit_rate: number;
  miss_breakdown: Record<string, number>;
  suggestions: {
    stat_model: string[];
    ai_prompt: string[];
    general_insights: string[];
    biggest_blind_spot: string;
  };
};

export type SportId = "UNKNOWN" | "NBA" | "NFL" | "NHL" | "SOCCER" | "MMA";

export type PropSide = "over" | "under";

export type Prop = {
  sport: SportId;
  league?: string | null;
  player_name: string;
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

  american_price?: number | null;
  decimal_price?: number | null;
  model_prob?: number | null;
  implied_prob?: number | null;
  edge?: number | null;
  ev?: number | null;
  volatility?: number | null;

  ai_bias?: number | null;
  ai_confidence?: number | null;
  ai_summary?: string | null;
  ai_tailwinds: string[];
  ai_risk_factors: string[];

  score?: number | null;
  notes: string[];
};

export type RankedPropsResponse = {
  scope: string;
  updated_at: string;
  props: Prop[];
};


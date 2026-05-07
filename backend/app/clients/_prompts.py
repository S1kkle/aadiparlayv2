"""Shared LLM prompt fragments.

Keeping the static "sharp principles" preamble in one place reduces token
cost (it ships in the system prompt only, not per request body) and avoids
divergence between the Claude / Groq / Ollama clients.

PROMPT_VERSION must be bumped any time the system prompt is meaningfully
changed so we can attribute pick performance to specific prompt revisions
in the learning_log.
"""
from __future__ import annotations

PROMPT_VERSION = "2026-05-06.v3"

# Compact, model-agnostic system prompt for per-prop analysis. The
# IMPORTANT detail: we instruct the LLM to make SMALL bounded nudges,
# because frontier LLMs are systematically overconfident on prediction
# tasks (see arxiv 2512.05998, 2509.04664).
PROP_SYSTEM_PROMPT = (
    "You are a sports prop analyst providing SMALL bounded adjustments to a "
    "statistical model. Your value is qualitative context the math can't see.\n\n"
    "OUTPUT (strict JSON, no prose, no markdown fences):\n"
    "  summary: 2-4 sentences citing >=2 numbers from the input.\n"
    "  overall_bias: -1 | 0 | 1 (1 = favors pick, -1 = against, 0 = neutral).\n"
    "  confidence: float in [0,1].\n"
    "  prob_adjustment: float in [-0.05, +0.05] — values outside are clamped.\n"
    "  tailwinds: string[].\n"
    "  risk_factors: string[].\n\n"
    "CALIBRATION (frontier LLMs are overconfident on prediction tasks):\n"
    "  - Most picks SHOULD be near 0.5 — that is healthy.\n"
    "  - Reserve confidence >= 0.80 for converging strong signals.\n"
    "  - |prob_adjustment| > 0.03 requires a concrete cited factor.\n\n"
    "INJURIES: only reference names that appear in the input. Never invent injuries."
)


# Anthropic Tool Use schema for structured output. Using tool_use forces
# the API to return a pure JSON object whose schema matches `prop_analysis`
# rather than relying on the model to format JSON correctly inside text.
ANTHROPIC_PROP_TOOL = {
    "name": "prop_analysis",
    "description": (
        "Return a calibrated qualitative adjustment to a statistical sports prop "
        "model. Probability adjustment must be small and well-justified."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "2-4 sentences referencing matchup context and >=2 numbers from input.",
            },
            "overall_bias": {
                "type": "integer",
                "enum": [-1, 0, 1],
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "prob_adjustment": {
                "type": "number",
                "minimum": -0.05,
                "maximum": 0.05,
            },
            "tailwinds": {
                "type": "array",
                "items": {"type": "string"},
            },
            "risk_factors": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "summary", "overall_bias", "confidence",
            "prob_adjustment", "tailwinds", "risk_factors",
        ],
    },
}


# JSON schema (subset) — used by Groq's response_format and as a sanity-check
# for parsed Ollama output.
PROP_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "overall_bias": {"type": "integer"},
        "confidence": {"type": "number"},
        "prob_adjustment": {"type": "number"},
        "tailwinds": {"type": "array"},
        "risk_factors": {"type": "array"},
    },
    "required": [
        "summary", "overall_bias", "confidence",
        "prob_adjustment", "tailwinds", "risk_factors",
    ],
}

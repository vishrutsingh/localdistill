"""
localdistill — Quality scoring engine.

Detects natural conversational signals to score conversation quality.
Signals are regex-based pattern matching on user messages.

Categories:
  NEGATIVE (penalize):
    - hallucination: user calls out fabricated info
    - correction: user corrects the model
    - code_error: code doesn't work
    - misunderstanding: model missed the point
    - vagueness: user wants more specifics
    - frustration: user is annoyed
    - regeneration: user re-prompts the same question
  
  POSITIVE (boost):
    - acceptance: user explicitly accepts the answer
    - continuation: user builds on the answer
    - execution: code/command was used as-is (detected externally)
"""

import re
from typing import List, Dict, Tuple

# ── Signal Definitions ───────────────────────────────────────

SIGNALS = {
    # ── NEGATIVE ──
    "hallucination": {
        "weight": -0.40,
        "description": "User indicates model fabricated information",
        "patterns": [
            r"(?i)\b(that['\u2019]s\s+not\s+(a\s+)?(real|actual|valid|true))\b",
            r"(?i)\byou\s+made\s+that\s+up\b",
            r"(?i)\bthat\s+doesn['\u2019]t\s+exist\b",
            r"(?i)\bwhere\s+did\s+you\s+get\s+that\b",
            r"(?i)\b(that['\u2019]s\s+not\s+(in\s+)?the\s+(docs?|documentation|API|codebase))\b",
            r"(?i)\bthere['\u2019]s\s+no\s+such\s+(function|method|class|endpoint|API|file)\b",
            r"(?i)\bthat\s+function\s+doesn['\u2019]t\s+(exist|work)\b",
            r"(?i)\byou['\u2019]re\s+(making|just\s+making)\s+(things|stuff)\s+up\b",
            r"(?i)\bhallucinat",
        ]
    },
    "correction": {
        "weight": -0.20,
        "description": "User corrects the model's output",
        "patterns": [
            r"(?i)\b(actually[,:]?\s+(no[,:]?\s+)?that['\u2019]s\s+(wrong|incorrect|not\s+right))\b",
            r"(?i)\b(no[,:]?\s+that['\u2019]s\s+(wrong|incorrect|not\s+right|not\s+how))\b",
            r"(?i)\bthat['\u2019]s\s+not\s+(correct|right|accurate|how\s+it\s+works)\b",
            r"(?i)\bthe\s+correct\s+(way|answer|approach|syntax)\s+is\b",
            r"(?i)\byou\s+got\s+that\s+wrong\b",
            r"(?i)\bthat['\u2019]s\s+a\s+(mistake|bug|error|typo)\b",
            r"(?i)\bincorrect[,:]\b",
        ]
    },
    "code_error": {
        "weight": -0.20,
        "description": "Code provided by model doesn't work",
        "patterns": [
            r"(?i)\b(this|that|it)\s+(gives|throws|produces|returns)\s+(an?\s+)?(error|exception)\b",
            r"(?i)\bdoesn['\u2019]t\s+(compile|run|work|execute)\b",
            r"(?i)\b(fails|failed|failing)\s+(with|at|on)\b",
            r"(?i)\b(traceback|stack\s+trace)\b",
            r"(?i)\b(syntax\s+error|type\s+error|runtime\s+error|import\s+error)\b",
            r"(?i)\bgot\s+an?\s+(error|exception)\b",
            r"(?i)\bnot\s+working\b",
        ]
    },
    "misunderstanding": {
        "weight": -0.15,
        "description": "Model didn't understand the user's request",
        "patterns": [
            r"(?i)\bthat['\u2019]s\s+not\s+what\s+I\s+(asked|meant|wanted|said)\b",
            r"(?i)\byou['\u2019]re\s+(misunderstanding|missing\s+the\s+point)\b",
            r"(?i)\b(read|re-read)\s+my\s+(question|message|prompt)\b",
            r"(?i)\byou\s+didn['\u2019]t\s+(understand|answer|address)\b",
            r"(?i)\bI\s+(asked|said|meant|wanted)\s+.*\bnot\b",
            r"(?i)\bthat['\u2019]s\s+not\s+(relevant|what\s+I\s+need|helpful)\b",
        ]
    },
    "vagueness": {
        "weight": -0.10,
        "description": "User wants more specific/detailed response",
        "patterns": [
            r"(?i)\b(be\s+more\s+specific|can\s+you\s+elaborate)\b",
            r"(?i)\bthat['\u2019]s\s+too\s+(vague|general|high.level|broad)\b",
            r"(?i)\b(give\s+me|I\s+need|show\s+me)\s+(actual|real|concrete)\s+(code|examples)\b",
            r"(?i)\b(stop|don['\u2019]t\s+just)\s+explain.*(just\s+)?(do|show|write)\s+it\b",
            r"(?i)\bI\s+need\s+more\s+(detail|specifics|information)\b",
        ]
    },
    "frustration": {
        "weight": -0.30,
        "description": "User is visibly frustrated",
        "patterns": [
            r"(?i)\byou['\u2019]re\s+not\s+(listening|helping|getting\s+it)\b",
            r"(?i)\bI\s+already\s+(said|told\s+you|mentioned)\b",
            r"(?i)\b(sigh|facepalm|ugh|argh|ffs|wtf)\b",
            r"(?i)\b(never\s+mind[,:]?\s+I['\u2019]ll\s+do\s+it\s+myself)\b",
            r"(?i)\bjust\s+stop\b",
            r"(?i)\byou['\u2019]re\s+(useless|hopeless|terrible)\b",
            r"(?i)\bthis\s+is\s+(going|taking)\s+(nowhere|too\s+long|forever)\b",
        ]
    },
    "regeneration": {
        "weight": -0.05,
        "description": "User re-prompts or regenerates response",
        "patterns": [
            r"(?i)\b(try\s+again|regenerate|redo|retry)\b",
            r"(?i)\b(let['\u2019]s\s+try\s+(that\s+)?again)\b",
            r"(?i)\b(start\s+over|from\s+scratch|from\s+the\s+beginning)\b",
        ]
    },

    # ── POSITIVE ──
    "acceptance": {
        "weight": +0.20,
        "description": "User explicitly accepts the response",
        "patterns": [
            r"(?i)\b(thanks?[,:]?\s+(that|this)\s+(works?|is\s+(perfect|great|exactly)))\b",
            r"(?i)\b(perfect[,:]?\s+(this|that|thanks?|exactly))\b",
            r"(?i)\b(exactly\s+what\s+I\s+(needed|wanted|was\s+looking\s+for))\b",
            r"(?i)\b(that|this)\s+(works?|worked|is\s+working)\s*(perfectly|great|well)?\b",
            r"(?i)\b(awesome|brilliant|fantastic|excellent)[,!:.]?\s*(thanks?|that)?\b",
            r"(?i)\b(great|nice|good)\s+(job|work|one)\b",
            r"(?i)\bthank\s+you[,:]?\s*(that|this)\s+(helped|works?|is)\b",
        ]
    },
    "continuation": {
        "weight": +0.10,
        "description": "User builds on the response naturally",
        "patterns": [
            r"(?i)\b(now[,:]?\s+(can\s+you|let['\u2019]s|add|extend|modify))\b",
            r"(?i)\b(can\s+you\s+(also|additionally|furthermore|next))\b",
            r"(?i)\b(building\s+on\s+that|following\s+up|as\s+a\s+follow.up)\b",
            r"(?i)\b(can\s+we\s+(also|now|next)\s+)\b",
        ]
    },
}


def analyze_message(text: str) -> List[Dict]:
    """
    Scan a user message for all quality signals.
    Returns list of detected signals with their matched text.
    """
    if not text:
        return []
    
    found = []
    for signal_type, config in SIGNALS.items():
        for pattern in config["patterns"]:
            match = re.search(pattern, text)
            if match:
                found.append({
                    "type": signal_type,
                    "weight": config["weight"],
                    "description": config["description"],
                    "matched_text": match.group(0),
                    "confidence": min(1.0, len(match.group(0)) / 15.0 + 0.5),
                })
                break  # One match per signal type per message
    
    return found


def score_conversation(signals: List[Dict], turn_count: int, 
                       has_tool_calls: bool = False,
                       was_abandoned: bool = False) -> Tuple[float, Dict]:
    """
    Compute overall quality score from accumulated signals.
    
    Base score: 0.60
    Adjusted by signal weights and structural factors.
    Returns (score, breakdown_dict).
    """
    score = 0.60  # Neutral baseline
    
    breakdown = {
        "base": 0.60,
        "signals_found": [],
        "structural": {},
    }
    
    # Apply signal weights
    for signal in signals:
        score += signal["weight"]
        breakdown["signals_found"].append({
            "type": signal["type"],
            "weight": signal["weight"],
            "matched": signal["matched_text"]
        })
    
    # Structural bonuses/penalties
    if turn_count >= 3:
        score += 0.05
        breakdown["structural"]["multi_turn"] = "+0.05"
    
    if turn_count >= 6:
        score += 0.05
        breakdown["structural"]["deep_conversation"] = "+0.05"
    
    if has_tool_calls:
        score += 0.10
        breakdown["structural"]["tool_use"] = "+0.10"
    
    if was_abandoned:
        score -= 0.10
        breakdown["structural"]["abandoned"] = "-0.10"
    
    if turn_count == 1:
        score -= 0.05
        breakdown["structural"]["single_turn"] = "-0.05"
    
    # Clamp to [0.0, 1.0]
    score = max(0.0, min(1.0, score))
    breakdown["final_score"] = round(score, 2)
    
    return round(score, 2), breakdown


# Load signal type → weight mapping for DB-based scoring
_SIGNAL_WEIGHTS = {name: cfg["weight"] for name, cfg in SIGNALS.items()}


def score_conversation_from_db(signals: List[Dict], turn_count: int,
                                has_tool_calls: bool = False,
                                was_abandoned: bool = False) -> Tuple[float, Dict]:
    """
    Compute overall quality score from DB signal records.
    DB signals have: signal_type, matched_text, confidence
    """
    score = 0.60
    breakdown = {"base": 0.60, "signals_found": [], "structural": {}}
    
    for signal in signals:
        signal_type = signal.get("signal_type", "")
        weight = _SIGNAL_WEIGHTS.get(signal_type, 0.0)
        score += weight
        breakdown["signals_found"].append({
            "type": signal_type,
            "weight": weight,
            "matched": signal.get("matched_text", "")
        })
    
    # Structural bonuses/penalties
    if turn_count >= 3:
        score += 0.05
        breakdown["structural"]["multi_turn"] = "+0.05"
    if turn_count >= 6:
        score += 0.05
        breakdown["structural"]["deep_conversation"] = "+0.05"
    if has_tool_calls:
        score += 0.10
        breakdown["structural"]["tool_use"] = "+0.10"
    if was_abandoned:
        score -= 0.10
        breakdown["structural"]["abandoned"] = "-0.10"
    if turn_count == 1:
        score -= 0.05
        breakdown["structural"]["single_turn"] = "-0.05"
    
    score = max(0.0, min(1.0, score))
    breakdown["final_score"] = round(score, 2)
    return round(score, 2), breakdown


# ── Scoring presets ──

AUTO_PROMOTE_THRESHOLD = 0.70
AUTO_EXCLUDE_THRESHOLD = 0.25


def should_auto_promote(score: float) -> bool:
    return score >= AUTO_PROMOTE_THRESHOLD


def should_auto_exclude(score: float) -> bool:
    return score <= AUTO_EXCLUDE_THRESHOLD
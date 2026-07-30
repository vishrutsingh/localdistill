"""
LocalDistill Evaluation

Evaluates trained model against holdout set.
Compares student responses to chosen responses.

Judging rules (see judge_pair):
  - Every pair is judged in BOTH orders. A win counts only if the judge picks
    the same response after the order is flipped; disagreement is recorded as
    position bias, not as a win. This removes position bias and makes the
    comparison deterministic (no RNG).
  - A judge reply that cannot be parsed is 'unparseable' and an API failure is
    'error'. Neither is ever silently folded into 'tie' — both are counted and
    reported, and too many aborts the stage.
"""

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field

# Judge outcome statuses
OK = "ok"
UNPARSEABLE = "unparseable"
ERROR = "error"

# Abort the stage past this rate of judge failures (needs a few observations
# first so one early blip doesn't kill an otherwise healthy run).
JUDGE_FAILURE_ABORT_RATE = 0.02
JUDGE_FAILURE_ABORT_MIN = 3


class JudgeUnavailable(RuntimeError):
    """Judge failed on too many items — results would be meaningless."""


@dataclass
class Verdict:
    """One order-bias-corrected comparison of two responses, x and y."""
    winner: str                       # "x" | "y" | "tie"
    status: str                       # OK | UNPARSEABLE | ERROR
    consistent: bool                  # judge agreed with itself when flipped
    raw: List[str] = field(default_factory=list)   # replies, [x-first, y-first]
    error: str = ""


@dataclass
class EvalResult:
    """Result of a single evaluation."""
    prompt: str
    student_response: str
    chosen_response: str
    winner: str  # "student", "chosen", or "tie"
    score_chosen: float = None
    judge_status: str = OK
    judge_consistent: bool = True
    judge_raw: List[str] = field(default_factory=list)
    judge_error: str = ""


def load_holdout(holdout_path: str) -> List[Dict]:
    """Load holdout pairs from JSONL file."""
    pairs = []
    with open(holdout_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def generate_student_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    """Generate response from student model."""
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy for reproducibility
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    return response.strip()


def simple_judge(
    prompt: str,
    response_a: str,
    response_b: str,
) -> Tuple[Optional[str], str, str]:
    """Heuristic judge - compares response quality without LLM.

    Returns (winner, raw, error) like llm_judge, where winner is "a"/"b"/"tie".
    Never fails, so error is always "".

    ponytail: this scores formatting (length, structure, keyword echo), not
    quality — it rewards exactly what an overfit model learns. Smoke tests only;
    the caller must refuse to gate on it. Upgrade path: llm_judge.

    Heuristics:
    1. Penalize empty/too short responses
    2. Penalize repetition
    3. Check if response addresses the prompt
    4. Prefer structured responses (lists, paragraphs)
    5. Penalize excessive length (rambling)
    """
    
    def score_response(response: str, prompt: str) -> float:
        score = 0.0
        
        # 1. Length checks
        length = len(response)
        if length < 20:
            return -10  # Too short, useless
        if length < 50:
            score -= 3  # Very short
        elif length < 100:
            score -= 1  # Short
        elif length > 2000:
            score -= 2  # Too long, probably rambling
        elif length > 500:
            score += 1  # Good length
        
        # 2. Repetition detection
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score -= 5  # Heavy repetition
            elif unique_ratio < 0.5:
                score -= 2  # Some repetition
            elif unique_ratio > 0.7:
                score += 1  # Good variety
        
        # 3. Prompt relevance - check if key words from prompt appear
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        # Remove common words
        common = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                  "being", "have", "has", "had", "do", "does", "did", "will",
                  "would", "could", "should", "may", "might", "must", "shall",
                  "can", "to", "of", "in", "for", "on", "with", "at", "by",
                  "from", "as", "into", "through", "during", "before", "after",
                  "above", "below", "between", "under", "again", "further",
                  "then", "once", "here", "there", "when", "where", "why",
                  "how", "all", "each", "few", "more", "most", "other", "some",
                  "such", "no", "nor", "not", "only", "own", "same", "so",
                  "than", "too", "very", "just", "and", "but", "if", "or",
                  "because", "until", "while", "what", "which", "who", "whom",
                  "this", "that", "these", "those", "i", "me", "my", "myself",
                  "you", "your", "yourself", "he", "him", "his", "she", "her",
                  "it", "its", "we", "us", "our", "they", "them", "their"}
        prompt_keywords = prompt_words - common
        overlap = prompt_keywords & response_words
        if prompt_keywords:
            relevance = len(overlap) / len(prompt_keywords)
            if relevance > 0.5:
                score += 2  # Addresses prompt well
            elif relevance > 0.2:
                score += 1  # Somewhat relevant
            elif relevance < 0.1:
                score -= 2  # Doesn't address prompt
        
        # 4. Structure indicators
        if '\n' in response:
            score += 1  # Has paragraphs/structure
        if any(marker in response for marker in ['1.', '2.', '- ', '* ', '•']):
            score += 1  # Has lists
        if '```' in response:
            score += 0.5  # Has code blocks (good for technical)
        
        # 5. Coherence - starts with capital, ends with punctuation
        if response and response[0].isupper():
            score += 0.5
        if response and response.rstrip()[-1] in '.!?':
            score += 0.5
        
        return score
    
    score_a = score_response(response_a, prompt)
    score_b = score_response(response_b, prompt)

    diff = score_a - score_b
    winner = "a" if diff > 1.5 else ("b" if diff < -1.5 else "tie")
    return winner, f"score_a={score_a:.1f} score_b={score_b:.1f}", ""


_VERDICT_RE = re.compile(r"^\W*(?:RESPONSE\s+)?(A|B|TIE)\b", re.IGNORECASE)


def parse_verdict(text: str) -> Optional[str]:
    """Parse a judge reply into "a" | "b" | "tie", or None if unparseable.

    Anchored at the start of the reply. Substring matching cannot be used here:
    "A is better" contains a "B" (in "BETTER"), so an `"A" in text and "B" not
    in text` test silently scores the most natural reply shape as a tie.
    """
    if not text:
        return None
    m = _VERDICT_RE.match(text.strip())
    return m.group(1).lower() if m else None


def llm_judge(
    prompt: str,
    response_a: str,
    response_b: str,
    judge_model: str = "openrouter/openai/gpt-4o-mini",
    retries: int = 3,
) -> Tuple[Optional[str], str, str]:
    """Use an LLM to compare two responses.

    Returns (winner, raw_reply, error) where winner is "a"/"b"/"tie", or None
    if the reply could not be parsed or every attempt failed. Errors are
    returned, never swallowed into a verdict — a down judge must not silently
    award half a point to every item.
    """
    import litellm

    judge_prompt = f"""You are an impartial judge. Compare these two responses to the user's question.

User question: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider: helpfulness, accuracy, clarity, and completeness.

Reply with ONLY one of: "A", "B", or "TIE". Do not explain.
"""

    last_error = ""
    for attempt in range(retries):
        try:
            resp = litellm.completion(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=10,
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            return parse_verdict(raw), raw, ""
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None, "", last_error


def judge_pair(
    prompt: str,
    response_x: str,
    response_y: str,
    judge_fn: Callable[[str, str, str], Tuple[Optional[str], str, str]],
) -> Verdict:
    """Judge x against y in both orders and resolve to an unbiased verdict.

    A win requires the judge to pick the same response after the order flips.
    Self-disagreement is position bias, and is recorded as a tie with
    consistent=False rather than being credited to whichever side won once.
    """
    first, raw_first, err_first = judge_fn(prompt, response_x, response_y)
    if err_first:
        return Verdict("tie", ERROR, False, [raw_first, ""], err_first)

    second, raw_second, err_second = judge_fn(prompt, response_y, response_x)
    if err_second:
        return Verdict("tie", ERROR, False, [raw_first, raw_second], err_second)

    raw = [raw_first, raw_second]
    if first is None or second is None:
        return Verdict("tie", UNPARSEABLE, False, raw,
                       f"unparseable judge reply: {raw!r}")

    # Map each order back into x/y space: in the flipped call, "a" means y.
    pick_first = {"a": "x", "b": "y", "tie": "tie"}[first]
    pick_second = {"a": "y", "b": "x", "tie": "tie"}[second]

    if pick_first == pick_second:
        return Verdict(pick_first, OK, True, raw)
    return Verdict("tie", OK, False, raw)


def evaluate_model(
    model,
    tokenizer,
    holdout_path: str,
    use_llm_judge: bool = False,
    judge_model: str = "openrouter/openai/gpt-4o-mini",
    max_examples: int = None,
    logger=None,
) -> Dict:
    """Evaluate model on holdout set.
    
    Args:
        model: Loaded student model
        tokenizer: Tokenizer
        holdout_path: Path to holdout.jsonl
        use_llm_judge: Whether to use LLM judge (costs money)
        judge_model: Model to use for judging
        max_examples: Limit number of examples to evaluate
        logger: Optional logger
    
    Returns:
        Dict with evaluation results and statistics
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(f"[eval] {msg}")
    
    pairs = load_holdout(holdout_path)
    if max_examples:
        pairs = pairs[:max_examples]

    log(f"Evaluating on {len(pairs)} holdout examples")

    if use_llm_judge:
        judge_fn = lambda p, a, b: llm_judge(p, a, b, judge_model)
    else:
        judge_fn = simple_judge

    results = []
    wins = {"student": 0, "chosen": 0, "tie": 0}
    failures = {UNPARSEABLE: 0, ERROR: 0}
    inconsistent = 0

    for i, pair in enumerate(pairs):
        prompt = pair["prompt"]
        chosen_msgs = pair["chosen"]

        # Extract chosen response (last assistant message)
        chosen_response = ""
        for msg in reversed(chosen_msgs):
            if msg["role"] == "assistant":
                chosen_response = msg["content"]
                break

        # Generate student response
        log(f"[{i+1}/{len(pairs)}] Generating student response...")
        student_response = generate_student_response(model, tokenizer, prompt)

        # Judge both orders; x=student, y=chosen
        verdict = judge_pair(prompt, student_response, chosen_response, judge_fn)
        winner = {"x": "student", "y": "chosen", "tie": "tie"}[verdict.winner]
        wins[winner] += 1
        if verdict.status != OK:
            failures[verdict.status] += 1
        elif not verdict.consistent:
            inconsistent += 1

        results.append(EvalResult(
            prompt=prompt,
            student_response=student_response,
            chosen_response=chosen_response,
            winner=winner,
            score_chosen=pair.get("score_chosen"),
            judge_status=verdict.status,
            judge_consistent=verdict.consistent,
            judge_raw=verdict.raw,
            judge_error=verdict.error,
        ))

        # Fail fast: a broken judge makes every later GPU-hour worthless
        n_failed = failures[UNPARSEABLE] + failures[ERROR]
        if (n_failed >= JUDGE_FAILURE_ABORT_MIN
                and n_failed / (i + 1) > JUDGE_FAILURE_ABORT_RATE):
            raise JudgeUnavailable(
                f"Judge failed on {n_failed}/{i + 1} items "
                f"({failures[ERROR]} errors, {failures[UNPARSEABLE]} unparseable). "
                f"Last error: {verdict.error or 'n/a'}"
            )

        if verdict.status != OK:
            note = f" [{verdict.status}]"
        elif not verdict.consistent:
            note = " [position-biased]"
        else:
            note = ""
        log(f"  Winner: {winner}{note}")

    # Calculate statistics
    total = len(results)
    if total == 0:
        raise ValueError(f"Holdout {holdout_path} produced no examples to evaluate")

    n_failed = failures[UNPARSEABLE] + failures[ERROR]
    judged_ok = total - n_failed
    decisive = wins["student"] + wins["chosen"]
    student_win_rate = (wins["student"] + wins["tie"] * 0.5) / total
    # Ties carry no information about direction; reporting both makes a
    # tie-heavy (i.e. uninformative) judge visible instead of averaging it away.
    win_rate_excl_ties = wins["student"] / decisive if decisive else None
    tie_rate = wins["tie"] / total
    # Only items the judge actually ruled on twice can show order disagreement.
    position_bias_rate = inconsistent / judged_ok if judged_ok else 0.0

    log(f"\nResults:")
    log(f"  Student wins: {wins['student']} ({wins['student']/total*100:.1f}%)")
    log(f"  Chosen wins: {wins['chosen']} ({wins['chosen']/total*100:.1f}%)")
    log(f"  Ties: {wins['tie']} ({tie_rate*100:.1f}%)")
    log(f"  Student win rate (ties=0.5): {student_win_rate*100:.1f}%")
    if win_rate_excl_ties is not None:
        log(f"  Student win rate (excl. ties): {win_rate_excl_ties*100:.1f}% of {decisive}")
    log(f"  Position bias (judge flipped): {inconsistent}/{judged_ok} ({position_bias_rate*100:.1f}%)")
    log(f"  Judge failures: {n_failed} ({failures[ERROR]} errors, {failures[UNPARSEABLE]} unparseable)")

    if tie_rate > 0.40:
        log(f"  WARNING: tie rate {tie_rate*100:.0f}% — judge is barely discriminating, "
            f"win rate is pulled toward 50% regardless of model quality")
    if position_bias_rate > 0.30:
        log(f"  WARNING: position bias {position_bias_rate*100:.0f}% — judge disagrees "
            f"with itself on order; treat this comparison as low confidence")

    return {
        "total": total,
        "wins": wins,
        "student_win_rate": student_win_rate,
        "win_rate_excl_ties": win_rate_excl_ties,
        "decisive": decisive,
        "tie_rate": tie_rate,
        "position_bias_rate": position_bias_rate,
        "judge_failures": n_failed,
        "judge_errors": failures[ERROR],
        "judge_unparseable": failures[UNPARSEABLE],
        "judge_mode": "llm" if use_llm_judge else "heuristic",
        "judge_model": judge_model if use_llm_judge else None,
        "valid_for_gating": use_llm_judge,
        "results": results,
    }


if __name__ == "__main__":
    # Self-check: verdict parsing and both-order resolution. Run: python -m lib.evaluate
    assert parse_verdict("A") == "a"
    assert parse_verdict("B") == "b"
    assert parse_verdict("TIE") == "tie"
    assert parse_verdict("tie") == "tie"
    assert parse_verdict('"A"') == "a"
    assert parse_verdict("Response B") == "b"
    assert parse_verdict("A.") == "a"
    # The regression this parser exists for: substring matching scored these
    # as ties because "BETTER"/"BECAUSE" contain a B.
    assert parse_verdict("A is better") == "a"
    assert parse_verdict("B is better") == "b"
    assert parse_verdict("A, because it is clearer") == "a"
    # Genuinely undecidable replies stay unparseable — never silently a tie.
    assert parse_verdict("Both are good") is None
    assert parse_verdict("") is None
    assert parse_verdict("I cannot decide") is None

    def _fixed(*replies):
        """Judge stub returning the given replies in call order."""
        seq = iter(replies)

        def judge(prompt, a, b):
            reply = next(seq)
            return parse_verdict(reply), reply, ""

        return judge

    # Consistent: x wins first order ("A"), x wins flipped order ("B")
    v = judge_pair("q", "x", "y", _fixed("A", "B"))
    assert (v.winner, v.status, v.consistent) == ("x", OK, True), v
    # Consistent: y wins both ways
    v = judge_pair("q", "x", "y", _fixed("B", "A"))
    assert (v.winner, v.status, v.consistent) == ("y", OK, True), v
    # Position bias: judge picks whichever came first -> tie, flagged
    v = judge_pair("q", "x", "y", _fixed("A", "A"))
    assert (v.winner, v.status, v.consistent) == ("tie", OK, False), v
    # Agreed tie is consistent
    v = judge_pair("q", "x", "y", _fixed("TIE", "TIE"))
    assert (v.winner, v.status, v.consistent) == ("tie", OK, True), v
    # Unparseable and error never become a scored win
    v = judge_pair("q", "x", "y", _fixed("Both are good", "A"))
    assert (v.winner, v.status) == ("tie", UNPARSEABLE), v
    v = judge_pair("q", "x", "y", lambda p, a, b: (None, "", "429 rate limited"))
    assert (v.winner, v.status) == ("tie", ERROR), v

    # simple_judge keeps the (winner, raw, error) contract and is order-symmetric
    long_good = "This is a clear, structured answer.\n- point one\n- point two\n" + "detail " * 40
    w1, _, e1 = simple_judge("q", long_good, "no")
    w2, _, e2 = simple_judge("q", "no", long_good)
    assert (w1, e1) == ("a", "") and (w2, e2) == ("b", ""), (w1, w2)

    print("lib/evaluate.py self-check passed")

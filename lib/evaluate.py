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
    generation: "Generation" = None


def load_holdout(holdout_path: str) -> List[Dict]:
    """Load holdout pairs from JSONL file."""
    pairs = []
    with open(holdout_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


@dataclass
class Generation:
    """One model response plus the facts needed to trust it."""
    text: str
    n_tokens: int = 0
    truncated: bool = False       # hit the cap without terminating
    eos: bool = False             # terminated on its own
    repetition_ratio: float = 0.0
    input_truncated: bool = False
    seconds: float = 0.0


def prompt_messages(pair: Dict) -> List[Dict[str, str]]:
    """Message prefix to condition on, from a holdout pair.

    Everything up to the final assistant turn — not just the flat `prompt`
    string. On multi-turn holdouts (i.e. any captured real conversation) the
    flat form asks the model to answer the last question with no context while
    grading it against a reference that had all of it.
    """
    chosen = pair.get("chosen") or []
    last_assistant = max(
        (i for i, m in enumerate(chosen) if m.get("role") == "assistant"),
        default=None,
    )
    if last_assistant is not None and last_assistant > 0:
        return [{"role": m["role"], "content": m["content"]} for m in chosen[:last_assistant]]
    return [{"role": "user", "content": pair.get("prompt", "")}]


def reference_response(pair: Dict) -> str:
    """The reference (last assistant turn) from a holdout pair."""
    for msg in reversed(pair.get("chosen") or []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def repetition_ratio(text: str, n: int = 4) -> float:
    """Fraction of n-grams that are repeats. High values mean a decode loop.

    Greedy decoding on a memorized model degenerates into repetition, and that
    otherwise shows up only as a slightly lower win rate, indistinguishable
    from any other cause.
    """
    words = text.split()
    if len(words) < 2 * n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between two texts (SQuAD-style), 0..1.

    Used for the regurgitation probe: high similarity to *training* targets
    next to low similarity on held-out targets is memorisation, stated as a
    number instead of inferred from a loss curve.
    """
    from collections import Counter
    pred = prediction.lower().split()
    ref = reference.lower().split()
    if not pred or not ref:
        return float(pred == ref)
    overlap = Counter(pred) & Counter(ref)
    same = sum(overlap.values())
    if same == 0:
        return 0.0
    precision = same / len(pred)
    recall = same / len(ref)
    return 2 * precision * recall / (precision + recall)


def _terminator_ids(tokenizer) -> set:
    """Token ids that mean 'the model stopped on its own'."""
    ids = {tokenizer.eos_token_id}
    for tok in ("<|eot_id|>", "<|im_end|>", "<|end|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            continue
        if tid is not None and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return {i for i in ids if i is not None}


def generate_batch(
    model,
    tokenizer,
    message_lists: List[List[Dict[str, str]]],
    max_new_tokens: int = 1024,
    max_seq_length: int = 2048,
    batch_size: int = 8,
    logger=None,
) -> List[Generation]:
    """Greedy-decode a batch of conversations, recording generation health.

    Greedy (do_sample=False) so the comparison is reproducible. Batched with
    left padding, because at one-at-a-time the sample sizes that make a win
    rate meaningful take hours, and people then turn the sample size back down.
    """
    import torch

    terminators = _terminator_ids(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # decoder-only models must pad on the left
    max_input_len = max(16, max_seq_length - max_new_tokens)

    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in message_lists
    ]

    out: List[Generation] = []
    try:
        i = 0
        while i < len(texts):
            chunk = texts[i:i + batch_size]
            # add_special_tokens=False: the chat template already emits BOS
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_input_len, add_special_tokens=False)
            over_length = [
                len(tokenizer(t, add_special_tokens=False)["input_ids"]) > max_input_len
                for t in chunk
            ]
            enc = {k: v.to(model.device) for k, v in enc.items()}

            started = time.time()
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                if logger:
                    logger.warning(f"OOM during generation — retrying at batch_size={batch_size}")
                torch.cuda.empty_cache()
                continue
            elapsed = time.time() - started

            prompt_len = enc["input_ids"].shape[1]
            for row, was_over in zip(outputs, over_length):
                new_ids = row[prompt_len:].tolist()
                hit_eos = any(t in terminators for t in new_ids)
                # trim padding/terminators for an accurate token count
                body = []
                for t in new_ids:
                    if t in terminators:
                        break
                    body.append(t)
                text = tokenizer.decode(body, skip_special_tokens=True).strip()
                out.append(Generation(
                    text=text,
                    n_tokens=len(body),
                    truncated=(not hit_eos and len(new_ids) >= max_new_tokens),
                    eos=hit_eos,
                    repetition_ratio=repetition_ratio(text),
                    input_truncated=was_over,
                    seconds=elapsed / max(1, len(chunk)),
                ))
            i += len(chunk)
    finally:
        tokenizer.padding_side = prev_side

    return out


def generation_health(gens: List[Generation]) -> Dict:
    """Aggregate generation diagnostics — the overfitting/decode tells."""
    n = len(gens) or 1
    lengths = [g.n_tokens for g in gens]
    return {
        "truncation_rate": sum(g.truncated for g in gens) / n,
        "eos_rate": sum(g.eos for g in gens) / n,
        "input_truncation_rate": sum(g.input_truncated for g in gens) / n,
        "degenerate_rate": sum(g.repetition_ratio > 0.3 for g in gens) / n,
        "empty_rate": sum(not g.text for g in gens) / n,
        "short_rate": sum(len(g.text) < 20 for g in gens) / n,
        "mean_tokens": sum(lengths) / n,
        "max_repetition_ratio": max((g.repetition_ratio for g in gens), default=0.0),
        "gen_seconds": sum(g.seconds for g in gens),
    }


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


# ── Statistics ────────────────────────────────────────────────────────────────

def wilson_interval(successes: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion.

    Used instead of the normal approximation because it stays sane at the
    sample sizes an overnight run can afford, and near 0/1.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = p + z2 / (2 * n)
    margin = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def _binom_tail_p(k: int, n: int) -> float:
    """Two-sided exact binomial p-value against p=0.5."""
    from math import comb
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def sign_test(wins_x: int, wins_y: int) -> float:
    """Two-sided exact sign test on a head-to-head. Ties are uninformative
    about direction and are excluded, which is what makes this exact."""
    return _binom_tail_p(min(wins_x, wins_y), wins_x + wins_y)


def mcnemar_test(only_x: int, only_y: int) -> float:
    """Two-sided exact McNemar on discordant pairs.

    For paired outcomes: items where x succeeded and y failed (only_x) versus
    the reverse (only_y). Items where both agree carry no information about
    which is better, so excluding them is what buys the extra power over
    treating the two conditions as independent samples.
    """
    return _binom_tail_p(min(only_x, only_y), only_x + only_y)


# ── Judging ───────────────────────────────────────────────────────────────────

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


@dataclass
class Comparison:
    """One head-to-head between two named variants over the same items."""
    x: str
    y: str
    wins_x: int = 0
    wins_y: int = 0
    ties: int = 0
    unparseable: int = 0
    errors: int = 0
    inconsistent: int = 0
    verdicts: List[Verdict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.wins_x + self.wins_y + self.ties

    @property
    def decisive(self) -> int:
        return self.wins_x + self.wins_y

    @property
    def judged_ok(self) -> int:
        return self.total - self.unparseable - self.errors

    @property
    def win_rate(self) -> float:
        """Win rate for x with ties at half credit."""
        return (self.wins_x + self.ties * 0.5) / self.total if self.total else 0.0

    def summary(self) -> Dict:
        lo, hi = wilson_interval(self.wins_x + self.ties * 0.5, self.total)
        return {
            "x": self.x,
            "y": self.y,
            "n": self.total,
            "wins_x": self.wins_x,
            "wins_y": self.wins_y,
            "ties": self.ties,
            "win_rate": self.win_rate,
            "ci95": [lo, hi],
            "win_rate_excl_ties": (self.wins_x / self.decisive) if self.decisive else None,
            "decisive": self.decisive,
            "sign_test_p": sign_test(self.wins_x, self.wins_y),
            "tie_rate": self.ties / self.total if self.total else 0.0,
            "position_bias_rate": (self.inconsistent / self.judged_ok) if self.judged_ok else 0.0,
            "unparseable": self.unparseable,
            "errors": self.errors,
        }


def judge_comparison(
    prompts: List[str],
    responses_x: List[str],
    responses_y: List[str],
    name_x: str,
    name_y: str,
    judge_fn: Callable[[str, str, str], Tuple[Optional[str], str, str]],
    concurrency: int = 8,
    logger=None,
) -> Comparison:
    """Judge x against y over aligned items, in parallel, both orders each.

    Aborts once judge failures pass JUDGE_FAILURE_ABORT_RATE: a judge that is
    down or unparseable produces a result centred on 50%, which reads exactly
    like "no effect" and would otherwise be reported as a finding.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n = len(prompts)
    comp = Comparison(x=name_x, y=name_y, verdicts=[None] * n)
    if n == 0:
        return comp

    def work(i):
        return i, judge_pair(prompts[i], responses_x[i], responses_y[i], judge_fn)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(work, i) for i in range(n)]
        try:
            for fut in as_completed(futures):
                i, verdict = fut.result()
                comp.verdicts[i] = verdict
                done += 1

                if verdict.status == UNPARSEABLE:
                    comp.unparseable += 1
                elif verdict.status == ERROR:
                    comp.errors += 1
                elif not verdict.consistent:
                    comp.inconsistent += 1

                failed = comp.unparseable + comp.errors
                if (failed >= JUDGE_FAILURE_ABORT_MIN
                        and failed / done > JUDGE_FAILURE_ABORT_RATE):
                    raise JudgeUnavailable(
                        f"Judge failed on {failed}/{done} items judging "
                        f"{name_x} vs {name_y} ({comp.errors} errors, "
                        f"{comp.unparseable} unparseable). "
                        f"Last error: {verdict.error or 'n/a'}"
                    )
        except BaseException:
            for f in futures:
                f.cancel()
            raise

    for verdict in comp.verdicts:
        if verdict.winner == "x":
            comp.wins_x += 1
        elif verdict.winner == "y":
            comp.wins_y += 1
        else:
            comp.ties += 1

    if logger:
        s = comp.summary()
        logger.info(
            f"{name_x} vs {name_y}: {s['win_rate']*100:.1f}% "
            f"[{s['ci95'][0]*100:.1f}, {s['ci95'][1]*100:.1f}] "
            f"(W{comp.wins_x}/L{comp.wins_y}/T{comp.ties}, p={s['sign_test_p']:.3f})"
        )
    return comp


def paired_outcomes(comp_x_ref: Comparison, comp_y_ref: Comparison) -> Dict:
    """McNemar between two variants scored against a shared reference.

    Both comparisons must cover the same items in the same order. Items where
    the two variants agree carry no information about which is better; only the
    discordant ones do, and using that is worth roughly 2-3x the sample size.
    """
    only_x = only_y = both = neither = 0
    for vx, vy in zip(comp_x_ref.verdicts, comp_y_ref.verdicts):
        # "beat the reference" = this variant won outright
        x_won = vx is not None and vx.winner == "x"
        y_won = vy is not None and vy.winner == "x"
        if x_won and not y_won:
            only_x += 1
        elif y_won and not x_won:
            only_y += 1
        elif x_won and y_won:
            both += 1
        else:
            neither += 1
    return {
        "only_x": only_x,
        "only_y": only_y,
        "both": both,
        "neither": neither,
        "discordant": only_x + only_y,
        "mcnemar_p": mcnemar_test(only_x, only_y),
    }


def adapter_fingerprint(model) -> Dict:
    """Evidence that a LoRA adapter is loaded, attached and actually trained.

    LoRA B matrices initialise to zero, so an all-zero B means the adapter is
    present but untrained -- and an adapter that never attached means the whole
    evaluation silently compares the base model against itself and reports no
    effect. Both are indistinguishable from "the method does not work" unless
    checked.
    """
    n_modules = n_trained = 0
    n_params = 0
    total_abs = 0.0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        n_params += param.numel()
        if "lora_B" in name:
            n_modules += 1
            magnitude = float(param.detach().abs().sum().item())
            total_abs += magnitude
            if magnitude > 0:
                n_trained += 1
    return {
        "lora_modules": n_modules,
        "lora_modules_trained": n_trained,
        "lora_params": n_params,
        "lora_b_abs_sum": total_abs,
        "attached": n_modules > 0,
        "trained": n_trained > 0,
    }


def identical_rate(a: List[str], b: List[str]) -> float:
    """Fraction of items where two variants produced byte-identical output."""
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x.strip() == y.strip()) / len(a)


def _smoke(model_name: str):
    """Generation smoke test on real weights — needs a GPU, so it is not part
    of the assert self-check. Run: python -m lib.evaluate --smoke <model>
    """
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=2048, dtype=None, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    convs = [
        [{"role": "user", "content": "Name three primary colours."}],
        [{"role": "user", "content": "Write a haiku about winter."}],
        [{"role": "user", "content": "What is 17 * 23? Show your working."}],
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
         {"role": "user", "content": "What did I just say?"}],   # multi-turn context
    ]
    gens = generate_batch(model, tokenizer, convs, max_new_tokens=128, batch_size=4)
    for c, g in zip(convs, gens):
        print(f"\n--- {c[-1]['content'][:50]!r}")
        print(f"    tokens={g.n_tokens} eos={g.eos} truncated={g.truncated} "
              f"rep={g.repetition_ratio:.2f} {g.seconds:.1f}s")
        print(f"    {g.text[:200]!r}")
    print("\nhealth:", json.dumps(generation_health(gens), indent=2))
    assert all(g.text for g in gens), "empty generation — batching or padding is wrong"
    assert gens[-1].eos, "multi-turn generation did not terminate"
    print("\nsmoke test passed")


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        _smoke(sys.argv[sys.argv.index("--smoke") + 1])
        raise SystemExit(0)

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

    # Repetition: the decode-loop signature must score high, prose must not.
    assert repetition_ratio("the cat sat on the mat and then went home to sleep") == 0.0
    assert repetition_ratio("a b c d " * 20) > 0.8
    assert repetition_ratio("too short") == 0.0

    # Multi-turn holdouts must be conditioned on the whole prefix, not the
    # flat prompt — otherwise the model answers with no context and is graded
    # against a reference that had all of it.
    multi = {"prompt": "q1", "chosen": [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]}
    assert prompt_messages(multi) == multi["chosen"][:3], prompt_messages(multi)
    assert reference_response(multi) == "a2"
    single = {"prompt": "q", "chosen": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    assert prompt_messages(single) == [{"role": "user", "content": "q"}]
    assert reference_response(single) == "a"
    # Degenerate holdout rows must not crash the prefix builder
    assert prompt_messages({"prompt": "q", "chosen": []}) == [{"role": "user", "content": "q"}]

    g = [Generation("ok", 10, False, True, 0.0), Generation("", 1024, True, False, 0.9)]
    h = generation_health(g)
    assert h["truncation_rate"] == 0.5 and h["eos_rate"] == 0.5
    assert h["degenerate_rate"] == 0.5 and h["empty_rate"] == 0.5

    # Statistics. Wilson 60/100 is a textbook value; the binomial cases are
    # hand-checkable (5 wins 0 losses -> 2 * 1/32).
    lo, hi = wilson_interval(60, 100)
    assert (round(lo, 4), round(hi, 4)) == (0.5020, 0.6906), (lo, hi)
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi, (lo, hi)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(10, 10)[1] == 1.0

    assert round(sign_test(60, 40), 4) == 0.0569, sign_test(60, 40)
    assert round(sign_test(5, 0), 4) == 0.0625, sign_test(5, 0)
    assert sign_test(50, 50) == 1.0
    assert sign_test(0, 0) == 1.0
    assert sign_test(30, 70) == sign_test(70, 30)          # symmetric
    assert sign_test(90, 10) < sign_test(60, 40)           # stronger -> smaller p

    assert round(mcnemar_test(5, 0), 4) == 0.0625
    assert mcnemar_test(10, 10) == 1.0
    assert mcnemar_test(0, 0) == 1.0
    # Pairing is the point: 20 vs 10 discordant is significant-ish, while the
    # same counts buried in 400 concordant items would not be if unpaired.
    assert mcnemar_test(20, 10) < 0.15, mcnemar_test(20, 10)

    # Comparison bookkeeping and the reported summary
    comp = Comparison("tuned", "base", wins_x=60, wins_y=30, ties=10, inconsistent=5)
    s = comp.summary()
    assert s["n"] == 100 and s["decisive"] == 90
    assert s["win_rate"] == 0.65 and round(s["win_rate_excl_ties"], 4) == 0.6667
    assert s["ci95"][0] > 0.5, s["ci95"]        # a real effect clears the interval
    assert round(s["position_bias_rate"], 4) == 0.05
    flat = Comparison("tuned", "base", wins_x=50, wins_y=50, ties=0).summary()
    assert flat["ci95"][0] < 0.5 < flat["ci95"][1], flat["ci95"]   # null does not

    # paired_outcomes counts only discordant items
    def _v(w):
        return Verdict(w, OK, True, [])
    a = Comparison("tuned", "reference", verdicts=[_v("x"), _v("x"), _v("y"), _v("tie")])
    b = Comparison("base", "reference", verdicts=[_v("x"), _v("y"), _v("y"), _v("x")])
    p = paired_outcomes(a, b)
    assert p == {"only_x": 1, "only_y": 1, "both": 1, "neither": 1,
                 "discordant": 2, "mcnemar_p": 1.0}, p

    # gap_closed: base loses badly, tuned draws level -> 100% of the gap
    from distill import DistillPipeline as _P
    assert _P._gap_closed(0.10, 0.50) == 1.0
    assert _P._gap_closed(0.10, 0.10) == 0.0
    assert _P._gap_closed(0.50, 0.50) is None      # no gap to close

    assert identical_rate(["a", "b"], ["a", "c"]) == 0.5
    assert identical_rate([], []) == 0.0

    # Regurgitation probe metric
    assert token_f1("the cat sat", "the cat sat") == 1.0
    assert token_f1("totally different words here", "nothing alike at all") == 0.0
    assert 0.4 < token_f1("the cat sat on the mat", "the cat sat") < 0.75
    assert token_f1("", "") == 1.0 and token_f1("x", "") == 0.0

    print("lib/evaluate.py self-check passed")

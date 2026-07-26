"""
LocalDistill Evaluation

Evaluates trained model against holdout set.
Compares student responses to chosen responses.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Result of a single evaluation."""
    prompt: str
    student_response: str
    chosen_response: str
    winner: str  # "student", "chosen", or "tie"
    score_chosen: float = None


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
) -> str:
    """Heuristic judge - compares response quality without LLM.
    
    Returns: "a", "b", or "tie"
    
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
    
    if diff > 1.5:
        return "a"
    elif diff < -1.5:
        return "b"
    else:
        return "tie"


def llm_judge(
    prompt: str,
    response_a: str,
    response_b: str,
    judge_model: str = "openrouter/openai/gpt-4o-mini",
) -> str:
    """Use LLM as judge to compare responses.
    
    Returns: "a", "b", or "tie"
    """
    import litellm
    
    judge_prompt = f"""You are an impartial judge. Compare these two responses to the user's question.

User question: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider: helpfulness, accuracy, clarity, and completeness.

Reply with ONLY one of: "A", "B", or "TIE"
"""
    
    try:
        resp = litellm.completion(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=10,
        )
        answer = resp.choices[0].message.content.strip().upper()
        
        if "A" in answer and "B" not in answer:
            return "a"
        elif "B" in answer and "A" not in answer:
            return "b"
        else:
            return "tie"
    except Exception as e:
        print(f"Judge error: {e}")
        return "tie"


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
    
    results = []
    wins = {"student": 0, "chosen": 0, "tie": 0}
    
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
        
        # Judge
        if use_llm_judge:
            # Randomize order to avoid position bias
            import random
            if random.random() < 0.5:
                winner_raw = llm_judge(prompt, student_response, chosen_response, judge_model)
                winner = "student" if winner_raw == "a" else ("chosen" if winner_raw == "b" else "tie")
            else:
                winner_raw = llm_judge(prompt, chosen_response, student_response, judge_model)
                winner = "chosen" if winner_raw == "a" else ("student" if winner_raw == "b" else "tie")
        else:
            # Simple heuristic judge
            import random
            if random.random() < 0.5:
                winner_raw = simple_judge(prompt, student_response, chosen_response)
                winner = "student" if winner_raw == "a" else ("chosen" if winner_raw == "b" else "tie")
            else:
                winner_raw = simple_judge(prompt, chosen_response, student_response)
                winner = "chosen" if winner_raw == "a" else ("student" if winner_raw == "b" else "tie")
        
        wins[winner] += 1
        
        results.append(EvalResult(
            prompt=prompt,
            student_response=student_response,
            chosen_response=chosen_response,
            winner=winner,
            score_chosen=pair.get("score_chosen"),
        ))
        
        log(f"  Winner: {winner}")
    
    # Calculate statistics
    total = len(results)
    student_win_rate = (wins["student"] + wins["tie"] * 0.5) / total if total > 0 else 0
    
    log(f"\nResults:")
    log(f"  Student wins: {wins['student']} ({wins['student']/total*100:.1f}%)")
    log(f"  Chosen wins: {wins['chosen']} ({wins['chosen']/total*100:.1f}%)")
    log(f"  Ties: {wins['tie']} ({wins['tie']/total*100:.1f}%)")
    log(f"  Student win rate (ties=0.5): {student_win_rate*100:.1f}%")
    
    return {
        "total": total,
        "wins": wins,
        "student_win_rate": student_win_rate,
        "results": results,
    }


if __name__ == "__main__":
    # Test with a simple example
    print("Evaluation module loaded successfully")
    print("Use: evaluate_model(model, tokenizer, holdout_path)")

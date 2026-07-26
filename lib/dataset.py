"""
LocalDistill Dataset Loader

Loads training data from:
- Local JSONL files (ChatML, ShareGPT, Alpaca formats)
- HuggingFace datasets
- Preference datasets (chosen/rejected pairs)

Applies curation filters and exports to training format.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass, field

from lib.config import Config, CurationConfig, DatasetConfig


@dataclass
class Conversation:
    """A single conversation with messages."""
    messages: List[Dict[str, str]]  # [{"role": "user", "content": "..."}, ...]
    metadata: Dict[str, Any] = None
    
    @property
    def turn_count(self) -> int:
        return len(self.messages)
    
    @property
    def total_chars(self) -> int:
        return sum(len(m.get("content", "")) for m in self.messages)
    
    def to_chatml(self) -> Dict[str, Any]:
        """Convert to ChatML format."""
        return {"messages": self.messages}
    
    def to_sharegpt(self) -> Dict[str, Any]:
        """Convert to ShareGPT format."""
        conversations = []
        for msg in self.messages:
            role = "human" if msg["role"] == "user" else "gpt"
            conversations.append({"from": role, "value": msg["content"]})
        return {"conversations": conversations}
    
    def to_alpaca(self) -> Dict[str, Any]:
        """Convert to Alpaca format (first turn only)."""
        if len(self.messages) < 2:
            return None
        return {
            "instruction": self.messages[0]["content"],
            "input": "",
            "output": self.messages[1]["content"],
        }


@dataclass
class PreferencePair:
    """A preference pair with chosen and rejected responses."""
    prompt: str
    chosen: List[Dict[str, str]]   # Full conversation with good response
    rejected: List[Dict[str, str]] # Full conversation with bad response
    score_chosen: float = None
    score_rejected: float = None
    
    def to_chosen_conversation(self) -> Conversation:
        """Extract just the chosen conversation for SFT."""
        return Conversation(messages=self.chosen)
    
    def to_rejected_conversation(self) -> Conversation:
        """Extract the rejected conversation (for analysis)."""
        return Conversation(messages=self.rejected)


class DatasetLoader:
    """
    Unified dataset loader supporting multiple sources and formats.
    """
    
    def __init__(self, config: Config, logger=None):
        self.config = config
        self.dataset_config = config.dataset
        self.curation_config = config.curation
        self.logger = logger
    
    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(f"[dataset] {msg}")
    
    def load(self) -> List[Conversation]:
        """Load dataset from configured source."""
        if self.dataset_config.source == "file":
            return self._load_from_file()
        elif self.dataset_config.source == "huggingface":
            return self._load_from_huggingface()
        elif self.dataset_config.source == "preference":
            # Load preference dataset, return only chosen conversations
            pairs = self._load_preference_dataset()
            return [p.to_chosen_conversation() for p in pairs]
        else:
            raise ValueError(f"Unknown dataset source: {self.dataset_config.source}")
    
    def load_preference_pairs(self) -> List[PreferencePair]:
        """Load preference dataset as pairs (for evaluation)."""
        if self.dataset_config.source != "preference":
            raise ValueError("load_preference_pairs requires source='preference'")
        return self._load_preference_dataset()
    
    def _load_preference_dataset(self) -> List[PreferencePair]:
        """Load HuggingFaceH4/ultrafeedback_binarized or similar preference dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
        
        hf_config = self.dataset_config.huggingface
        dataset_id = hf_config.dataset_id
        split = hf_config.split
        
        self._log(f"Loading preference dataset: {dataset_id} ({split})")
        
        dataset = load_dataset(dataset_id, split=split)
        
        pairs = []
        max_examples = self.curation_config.max_examples
        
        for item in dataset:
            # Handle ultrafeedback_binarized format
            chosen = item.get("chosen", [])
            rejected = item.get("rejected", [])
            prompt = item.get("prompt", "")
            
            if not chosen or not rejected:
                continue
            
            # Normalize message format
            chosen_msgs = self._normalize_messages(chosen)
            rejected_msgs = self._normalize_messages(rejected)
            
            if not chosen_msgs or not rejected_msgs:
                continue
            
            pair = PreferencePair(
                prompt=prompt,
                chosen=chosen_msgs,
                rejected=rejected_msgs,
                score_chosen=item.get("score_chosen"),
                score_rejected=item.get("score_rejected"),
            )
            pairs.append(pair)
            
            if max_examples and len(pairs) >= max_examples:
                break
        
        self._log(f"Loaded {len(pairs)} preference pairs")
        return pairs
    
    def _normalize_messages(self, messages: List) -> List[Dict[str, str]]:
        """Normalize message format to [{role, content}, ...]."""
        if not messages:
            return []
        
        normalized = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                normalized.append({
                    "role": msg["role"],
                    "content": str(msg["content"]),
                })
        return normalized
    
    def _load_from_file(self) -> List[Conversation]:
        """Load from local JSONL file."""
        path = Path(self.dataset_config.path).expanduser()
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        self._log(f"Loading from file: {path}")
        
        conversations = []
        format_type = self.dataset_config.format
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    conv = self._parse_conversation(data, format_type)
                    if conv:
                        conversations.append(conv)
                except json.JSONDecodeError as e:
                    self._log(f"Skipping line {line_num}: JSON parse error - {e}")
                except Exception as e:
                    self._log(f"Skipping line {line_num}: {e}")
        
        self._log(f"Loaded {len(conversations)} conversations from file")
        return conversations
    
    def _load_from_huggingface(self) -> List[Conversation]:
        """Load from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
        
        hf_config = self.dataset_config.huggingface
        self._log(f"Loading from HuggingFace: {hf_config.dataset_id}")
        
        # Load dataset
        kwargs = {"split": hf_config.split}
        if hf_config.config:
            kwargs["name"] = hf_config.config
        
        dataset = load_dataset(hf_config.dataset_id, **kwargs)
        
        conversations = []
        mapping = hf_config.field_mapping
        
        for item in dataset:
            try:
                conv = self._parse_huggingface_item(item, mapping)
                if conv:
                    conversations.append(conv)
            except Exception as e:
                # Skip malformed items
                continue
        
        self._log(f"Loaded {len(conversations)} conversations from HuggingFace")
        return conversations
    
    def _parse_conversation(self, data: Dict, format_type: str) -> Optional[Conversation]:
        """Parse a single item based on format."""
        if format_type == "chatml":
            return self._parse_chatml(data)
        elif format_type == "sharegpt":
            return self._parse_sharegpt(data)
        elif format_type == "alpaca":
            return self._parse_alpaca(data)
        else:
            raise ValueError(f"Unknown format: {format_type}")
    
    def _parse_chatml(self, data: Dict) -> Optional[Conversation]:
        """Parse ChatML format: {"messages": [{"role": "...", "content": "..."}]}"""
        messages = data.get("messages", [])
        if not messages:
            return None
        
        # Validate structure
        valid_messages = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                valid_messages.append({
                    "role": msg["role"],
                    "content": str(msg["content"]),
                })
        
        if not valid_messages:
            return None
        
        return Conversation(messages=valid_messages, metadata=data.get("metadata"))
    
    def _parse_sharegpt(self, data: Dict) -> Optional[Conversation]:
        """Parse ShareGPT format: {"conversations": [{"from": "human", "value": "..."}]}"""
        convs = data.get("conversations", [])
        if not convs:
            return None
        
        messages = []
        for turn in convs:
            from_field = turn.get("from", "")
            value = turn.get("value", "")
            
            if from_field in ("human", "user"):
                role = "user"
            elif from_field in ("gpt", "assistant", "bot"):
                role = "assistant"
            elif from_field == "system":
                role = "system"
            else:
                continue
            
            messages.append({"role": role, "content": str(value)})
        
        if not messages:
            return None
        
        return Conversation(messages=messages)
    
    def _parse_alpaca(self, data: Dict) -> Optional[Conversation]:
        """Parse Alpaca format: {"instruction": "...", "input": "...", "output": "..."}"""
        instruction = data.get("instruction", "")
        input_text = data.get("input", "")
        output = data.get("output", "")
        
        if not instruction or not output:
            return None
        
        # Combine instruction and input
        user_content = instruction
        if input_text:
            user_content = f"{instruction}\n\nInput: {input_text}"
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        
        return Conversation(messages=messages)
    
    def _parse_huggingface_item(self, item: Dict, mapping: Dict) -> Optional[Conversation]:
        """Parse HuggingFace dataset item with field mapping."""
        conv_field = mapping.get("conversations", "conversations")
        role_field = mapping.get("role_field", "from")
        content_field = mapping.get("content_field", "value")
        human_role = mapping.get("human_role", "human")
        assistant_role = mapping.get("assistant_role", "gpt")
        
        turns = item.get(conv_field, [])
        if not turns:
            return None
        
        messages = []
        for turn in turns:
            role_value = turn.get(role_field, "")
            content = turn.get(content_field, "")
            
            if role_value == human_role:
                role = "user"
            elif role_value == assistant_role:
                role = "assistant"
            elif role_value == "system":
                role = "system"
            else:
                continue
            
            messages.append({"role": role, "content": str(content)})
        
        if not messages:
            return None
        
        return Conversation(messages=messages)
    
    def curate(self, conversations: List[Conversation]) -> List[Conversation]:
        """Apply curation filters to conversations."""
        filters = self.curation_config.filters
        min_score = self.curation_config.min_quality_score
        max_examples = self.curation_config.max_examples
        
        curated = []
        stats = {
            "total": len(conversations),
            "filtered_turns": 0,
            "filtered_chars": 0,
            "filtered_code": 0,
            "passed": 0,
        }
        
        for conv in conversations:
            # Turn count filter
            min_turns = filters.get("min_turns", 2)
            max_turns = filters.get("max_turns", 50)
            if conv.turn_count < min_turns or conv.turn_count > max_turns:
                stats["filtered_turns"] += 1
                continue
            
            # Character count filter
            min_chars = filters.get("min_chars", 100)
            max_chars = filters.get("max_chars", 50000)
            if conv.total_chars < min_chars or conv.total_chars > max_chars:
                stats["filtered_chars"] += 1
                continue
            
            # Code-heavy filter
            if filters.get("exclude_code_heavy", False):
                if self._is_code_heavy(conv):
                    stats["filtered_code"] += 1
                    continue
            
            curated.append(conv)
            stats["passed"] += 1
            
            # Max examples limit
            if max_examples and len(curated) >= max_examples:
                break
        
        self._log(f"Curation stats: {stats}")
        self._log(f"Curated {len(curated)} conversations")
        
        return curated
    
    def _is_code_heavy(self, conv: Conversation, threshold: float = 0.8) -> bool:
        """Check if conversation is >threshold code blocks."""
        total_lines = 0
        code_lines = 0
        
        for msg in conv.messages:
            if msg["role"] != "assistant":
                continue
            
            content = msg["content"]
            lines = content.split("\n")
            inside_code = False
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("```"):
                    inside_code = not inside_code
                    continue
                
                if stripped:
                    total_lines += 1
                    if inside_code:
                        code_lines += 1
        
        if total_lines == 0:
            return False
        
        return (code_lines / total_lines) > threshold
    
    def export(
        self,
        conversations: List[Conversation],
        output_path: str,
        format_type: str = "chatml",
    ) -> int:
        """Export conversations to JSONL file."""
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                if format_type == "chatml":
                    data = conv.to_chatml()
                elif format_type == "sharegpt":
                    data = conv.to_sharegpt()
                elif format_type == "alpaca":
                    data = conv.to_alpaca()
                    if data is None:
                        continue
                else:
                    data = conv.to_chatml()
                
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1
        
        self._log(f"Exported {count} conversations to {output_path}")
        return count
    
    def load_and_curate(self) -> Tuple[List[Conversation], str]:
        """
        Full pipeline: load, curate, export.
        
        Returns (conversations, output_path).
        """
        # Load from source
        conversations = self.load()
        
        # Apply curation
        curated = self.curate(conversations)
        
        # Export to training file
        output_path = Path(self.config.logging.dir).parent / "train.jsonl"
        self.export(curated, str(output_path), "chatml")
        
        return curated, str(output_path)


def load_dataset_from_config(config: Config, logger=None) -> Tuple[List[Conversation], str]:
    """Convenience function to load and curate dataset from config."""
    loader = DatasetLoader(config, logger)
    return loader.load_and_curate()


def split_dataset(
    items: List,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[List, List]:
    """Split dataset into train and holdout sets.
    
    Args:
        items: List of items to split (Conversations or PreferencePairs)
        train_ratio: Fraction for training (default 0.9 = 90% train, 10% holdout)
        seed: Random seed for reproducibility
    
    Returns:
        (train_set, holdout_set)
    """
    items = list(items)  # Copy to avoid mutating original
    random.seed(seed)
    random.shuffle(items)
    
    split_idx = int(len(items) * train_ratio)
    return items[:split_idx], items[split_idx:]


if __name__ == "__main__":
    # Test the loader
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "preference":
        # Test preference dataset loading
        print("Testing preference dataset loading...")
        print("Dataset: HuggingFaceH4/ultrafeedback_binarized")
        
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        
        # Convert to PreferencePairs
        pairs = []
        for i, item in enumerate(ds):
            if i >= 100:  # Just load 100 for testing
                break
            chosen = item.get("chosen", [])
            rejected = item.get("rejected", [])
            if chosen and rejected:
                pairs.append(PreferencePair(
                    prompt=item.get("prompt", ""),
                    chosen=chosen,
                    rejected=rejected,
                    score_chosen=item.get("score_chosen"),
                    score_rejected=item.get("score_rejected"),
                ))
        
        print(f"Loaded {len(pairs)} preference pairs")
        
        # Split
        train, holdout = split_dataset(pairs, train_ratio=0.9)
        print(f"Split: {len(train)} train, {len(holdout)} holdout")
        
        # Show sample
        if pairs:
            p = pairs[0]
            print(f"\nSample pair:")
            print(f"  Prompt: {p.prompt[:100]}...")
            print(f"  Chosen score: {p.score_chosen}")
            print(f"  Rejected score: {p.score_rejected}")
            print(f"  Chosen response: {p.chosen[-1]['content'][:100]}...")
    else:
        # Original test
        from lib.config import load_config
        
        config = load_config()
        loader = DatasetLoader(config)
        
        conversations = loader.load()
        print(f"Loaded {len(conversations)} conversations")
        
        if conversations:
            print(f"\nFirst conversation ({conversations[0].turn_count} turns):")
            for msg in conversations[0].messages[:2]:
                print(f"  [{msg['role']}]: {msg['content'][:100]}...")
        
        curated = loader.curate(conversations)
        print(f"\nAfter curation: {len(curated)} conversations")

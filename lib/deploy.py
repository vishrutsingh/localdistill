"""
LocalDistill Deployment Utilities

Handles:
- GGUF export for llama.cpp/Ollama
- Ollama model registration
- Adapter management
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime


def check_ollama_installed() -> bool:
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def export_gguf(
    model,
    tokenizer,
    adapter_dir: str,
    quantization: str = "q4_k_m",
    logger=None,
) -> Optional[str]:
    """
    Export LoRA adapter as GGUF for Ollama/llama.cpp.
    
    Args:
        model: The trained model (with LoRA)
        tokenizer: The tokenizer
        adapter_dir: Directory where adapter is saved
        quantization: GGUF quantization method (q4_k_m, q5_k_m, q8_0, f16)
        logger: Optional logger instance
    
    Returns:
        Path to GGUF file, or None if export failed.
    """
    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(f"[deploy] {msg}")
    
    try:
        gguf_dir = Path(adapter_dir) / "gguf"
        gguf_dir.mkdir(exist_ok=True)
        
        _log(f"Exporting GGUF with {quantization} quantization...")
        
        # Use Unsloth's save_pretrained_gguf
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=quantization,
        )
        
        # Find the generated GGUF file
        gguf_files = list(gguf_dir.glob("*.gguf"))
        if not gguf_files:
            _log("Warning: No GGUF file generated")
            return None
        
        gguf_path = gguf_files[0]
        _log(f"GGUF exported: {gguf_path}")
        
        # Create Ollama Modelfile
        modelfile_path = create_modelfile(gguf_dir, gguf_path.name)
        _log(f"Modelfile created: {modelfile_path}")
        
        return str(gguf_path)
        
    except Exception as e:
        _log(f"GGUF export failed: {e}")
        return None


def create_modelfile(
    output_dir: Path,
    gguf_filename: str,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Create an Ollama Modelfile for the GGUF.
    
    Returns path to Modelfile.
    """
    modelfile_path = output_dir / "Modelfile"
    
    # Default chat template (Llama-style)
    template = '''{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
'''
    
    content = f'FROM {gguf_filename}\n'
    content += f'TEMPLATE """{template}"""\n'
    
    if system_prompt:
        content += f'SYSTEM """{system_prompt}"""\n'
    
    # Add some reasonable defaults
    content += '''
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
'''
    
    modelfile_path.write_text(content)
    return str(modelfile_path)


def register_ollama_model(
    modelfile_path: str,
    model_name: str = "localdistill",
    logger=None,
) -> bool:
    """
    Register the model with Ollama using `ollama create`.
    
    Args:
        modelfile_path: Path to the Modelfile
        model_name: Name for the Ollama model
        logger: Optional logger instance
    
    Returns:
        True if registration succeeded.
    """
    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(f"[deploy] {msg}")
    
    if not check_ollama_installed():
        _log("Ollama not installed or not running")
        return False
    
    modelfile_path = Path(modelfile_path)
    if not modelfile_path.exists():
        _log(f"Modelfile not found: {modelfile_path}")
        return False
    
    try:
        _log(f"Registering model '{model_name}' with Ollama...")
        
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            cwd=modelfile_path.parent,
            timeout=300,  # 5 min timeout for large models
        )
        
        if result.returncode == 0:
            _log(f"Model registered: ollama run {model_name}")
            return True
        else:
            _log(f"Ollama registration failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        _log("Ollama registration timed out")
        return False
    except Exception as e:
        _log(f"Ollama registration error: {e}")
        return False


def list_adapters(adapters_dir: str = "~/localdistill/adapters") -> List[Dict[str, Any]]:
    """
    List all trained adapters.
    
    Returns list of adapter info dicts.
    """
    adapters_dir = Path(adapters_dir).expanduser()
    if not adapters_dir.exists():
        return []
    
    adapters = []
    for adapter_path in sorted(adapters_dir.iterdir(), reverse=True):
        if not adapter_path.is_dir():
            continue
        
        # Check for adapter files
        config_file = adapter_path / "adapter_config.json"
        if not config_file.exists():
            continue
        
        # Get metadata
        info = {
            "id": adapter_path.name,
            "path": str(adapter_path),
            "created_at": datetime.fromtimestamp(adapter_path.stat().st_mtime).isoformat(),
            "has_gguf": (adapter_path / "gguf").exists(),
        }
        
        # Check for GGUF files
        gguf_dir = adapter_path / "gguf"
        if gguf_dir.exists():
            gguf_files = list(gguf_dir.glob("*.gguf"))
            info["gguf_files"] = [f.name for f in gguf_files]
            info["modelfile"] = str(gguf_dir / "Modelfile") if (gguf_dir / "Modelfile").exists() else None
        
        # Try to read status.json if exists
        status_file = adapter_path / "status.json"
        if status_file.exists():
            import json
            with open(status_file) as f:
                info["status"] = json.load(f)
        
        adapters.append(info)
    
    return adapters


def get_latest_adapter(adapters_dir: str = "~/localdistill/adapters") -> Optional[str]:
    """Get path to the most recent adapter."""
    adapters = list_adapters(adapters_dir)
    if adapters:
        return adapters[0]["path"]
    return None


def cleanup_old_adapters(
    adapters_dir: str = "~/localdistill/adapters",
    keep: int = 5,
    logger=None,
) -> int:
    """
    Remove old adapters, keeping the most recent N.
    
    Returns number of adapters removed.
    """
    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(f"[deploy] {msg}")
    
    adapters = list_adapters(adapters_dir)
    
    if len(adapters) <= keep:
        return 0
    
    removed = 0
    for adapter in adapters[keep:]:
        try:
            shutil.rmtree(adapter["path"])
            _log(f"Removed old adapter: {adapter['id']}")
            removed += 1
        except Exception as e:
            _log(f"Failed to remove {adapter['id']}: {e}")
    
    return removed


def deploy_adapter(
    adapter_path: str,
    model_name: str = "localdistill",
    quantization: str = "q4_k_m",
    register_ollama: bool = True,
    logger=None,
) -> Dict[str, Any]:
    """
    Full deployment pipeline for an adapter.
    
    If GGUF doesn't exist, this won't create it (needs model/tokenizer).
    Use export_gguf() during training for that.
    
    Returns deployment status dict.
    """
    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(f"[deploy] {msg}")
    
    adapter_path = Path(adapter_path)
    result = {
        "adapter_path": str(adapter_path),
        "gguf_path": None,
        "modelfile_path": None,
        "ollama_registered": False,
        "ollama_model_name": None,
    }
    
    # Check for GGUF
    gguf_dir = adapter_path / "gguf"
    if gguf_dir.exists():
        gguf_files = list(gguf_dir.glob("*.gguf"))
        if gguf_files:
            result["gguf_path"] = str(gguf_files[0])
            
            modelfile = gguf_dir / "Modelfile"
            if modelfile.exists():
                result["modelfile_path"] = str(modelfile)
    
    if not result["gguf_path"]:
        _log("No GGUF found. Run training with deploy.gguf.enabled=true")
        return result
    
    # Register with Ollama
    if register_ollama and result["modelfile_path"]:
        if register_ollama_model(result["modelfile_path"], model_name, logger):
            result["ollama_registered"] = True
            result["ollama_model_name"] = model_name
    
    return result


if __name__ == "__main__":
    # Test adapter listing
    print("Checking Ollama:", check_ollama_installed())
    
    adapters = list_adapters()
    print(f"\nFound {len(adapters)} adapters:")
    for a in adapters[:3]:
        print(f"  - {a['id']} ({a['created_at']}) GGUF: {a['has_gguf']}")

# -*- coding: utf-8 -*-
import os
from typing import Dict, Any, Optional

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1].strip()
    return s

def read_simple_yaml(path: str) -> Dict[str, Any]:
    """
    Minimal YAML reader for simple 'key: value' lines.
    Supports:
      - comments with '#'
      - blank lines
      - values as strings (no nested dict/list)
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"config yaml not found: {path}")

    out: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            # remove inline comment
            if "#" in raw:
                raw = raw.split("#", 1)[0].strip()
                if not raw:
                    continue
            if ":" not in raw:
                continue
            k, v = raw.split(":", 1)
            k = k.strip()
            v = _strip_quotes(v.strip())
            if k:
                out[k] = v
    return out

def resolve_path(repo_root: str, p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    p = str(p).strip()
    if not p:
        return None
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(repo_root, p))

def load_config(config_path: str, repo_root: Optional[str] = None) -> Dict[str, Any]:
    repo_root = repo_root or os.getcwd()
    cfg = read_simple_yaml(config_path)
    cfg["_config_path"] = os.path.abspath(config_path)
    cfg["_repo_root"] = os.path.abspath(repo_root)

    # resolve common path keys
    for k in ["train", "valid", "test", "dev", "train_all", "categories"]:
        if k in cfg:
            cfg[k] = resolve_path(cfg["_repo_root"], cfg[k])
    return cfg

def apply_cfg_defaults(args, cfg: Dict[str, Any], mapping: Dict[str, str]):
    """
    mapping: { cfg_key -> args_attr }
    Only fills args_attr if args_attr is None or empty.
    """
    for ck, ak in mapping.items():
        if ck not in cfg:
            continue
        cur = getattr(args, ak, None)
        if cur is None or (isinstance(cur, str) and cur.strip() == ""):
            setattr(args, ak, cfg[ck])
    return args

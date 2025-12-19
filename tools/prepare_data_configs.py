# -*- coding: utf-8 -*-
"""
Prepare local splits (60/20/20), categories.txt, and data.yaml for each data/<lang>-<domain> folder.

Assumptions:
- data/<lang>-<domain>/official/*train_alltasks*.jsonl exists (labeled)
- data/<lang>-<domain>/official/*dev_task3*.jsonl exists (unlabeled)
Outputs:
- data/<lang>-<domain>/local/{train,valid,test}.jsonl
- configs/<lang>-<domain>/{data.yaml,<domain>_categories.txt}
"""

import os, re, json, argparse, random
from pathlib import Path
from typing import List, Dict, Any, Set

INVALID = "INVALID"

def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(p: Path, rows: List[Dict[str, Any]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def find_one(glob_paths: List[Path]) -> Path:
    for p in glob_paths:
        if p.exists():
            return p
    return None

def extract_categories_from_train(rows: List[Dict[str, Any]]) -> Set[str]:
    cats = set()
    for r in rows:
        qs = r.get("Quadruplet") or r.get("Quadruplets") or r.get("quadruplets") or []
        for q in qs:
            c = q.get("Category", None)
            if c is None:
                continue
            t = str(c).strip()
            if not t:
                continue
            cats.add(t.upper())
    # enforce INVALID last & unique
    cats.discard(INVALID)
    return set(cats)

def write_categories_file(p: Path, cats_upper: Set[str]):
    p.parent.mkdir(parents=True, exist_ok=True)
    cats_sorted = sorted(list(cats_upper))
    cats_sorted = [c for c in cats_sorted if c != INVALID] + [INVALID]
    with p.open("w", encoding="utf-8") as f:
        for c in cats_sorted:
            f.write(c + "\n")

def split_rows(rows: List[Dict[str, Any]], seed: int, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2):
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6
    n = len(rows)
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)

    n_train = int(round(n * train_ratio))
    n_valid = int(round(n * valid_ratio))
    # ensure total matches
    n_train = min(n_train, n)
    n_valid = min(n_valid, n - n_train)
    n_test = n - n_train - n_valid

    tr = [rows[i] for i in idx[:n_train]]
    va = [rows[i] for i in idx[n_train:n_train+n_valid]]
    te = [rows[i] for i in idx[n_train+n_valid:]]
    assert len(tr) + len(va) + len(te) == n
    return tr, va, te

def infer_domain_name(folder_name: str) -> str:
    # folder_name like "eng-Laptop" -> "Laptop"
    parts = folder_name.split("-", 1)
    return parts[1] if len(parts) == 2 else folder_name

def domain_to_cat_filename(domain: str) -> str:
    d = domain.lower()
    if d == "laptop":
        return "laptop_categories.txt"
    if d == "restaurant":
        return "restaurant_categories.txt"
    if d == "hotel":
        return "hotel_categories.txt"
    # fallback
    return f"{d}_categories.txt"

def write_data_yaml(p: Path, train_p: Path, valid_p: Path, test_p: Path, dev_p: Path, train_all_p: Path, cats_p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    txt = "\n".join([
        f"train: {train_p.as_posix()}",
        f"valid: {valid_p.as_posix()}",
        f"test:  {test_p.as_posix()}",
        f"dev:   {dev_p.as_posix()}",
        f"train_all: {train_all_p.as_posix()}",
        f"categories: {cats_p.as_posix()}",
        ""
    ])
    p.write_text(txt, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".", help="repo root, run from One_ASQP")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--configs_dir", type=str, default="configs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite existing local splits/cfg")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    data_root = (repo / args.data_dir).resolve()
    cfg_root = (repo / args.configs_dir).resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"data_dir not found: {data_root}")
    if not cfg_root.exists():
        cfg_root.mkdir(parents=True, exist_ok=True)

    folders = [p for p in data_root.iterdir() if p.is_dir()]
    folders.sort(key=lambda x: x.name)

    print(f"[Scan] data_root={data_root} folders={len(folders)} seed={args.seed}")
    for fd in folders:
        official = fd / "official"
        if not official.exists():
            continue

        # locate official train/dev
        train_all = find_one([
            official / f"{fd.name.replace('-', '_').lower()}_train_alltasks.jsonl",
        ])
        if train_all is None:
            # fallback: glob
            cands = list(official.glob("*train_alltasks*.jsonl"))
            train_all = cands[0] if cands else None

        dev = None
        cands = list(official.glob("*dev_task3*.jsonl"))
        dev = cands[0] if cands else None

        if train_all is None or dev is None:
            print(f"[Skip] {fd.name}: missing train_all/dev_task3 in {official}")
            continue

        # output local splits
        local = fd / "local"
        out_train = local / "train.jsonl"
        out_valid = local / "valid.jsonl"
        out_test  = local / "test.jsonl"

        if (not args.force) and out_train.exists() and out_valid.exists() and out_test.exists():
            print(f"[Keep] {fd.name}: local splits exist (use --force to overwrite)")
        else:
            rows = read_jsonl(train_all)
            tr, va, te = split_rows(rows, seed=args.seed, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2)
            write_jsonl(out_train, tr)
            write_jsonl(out_valid, va)
            write_jsonl(out_test, te)
            print(f"[Split] {fd.name}: train={len(tr)} valid={len(va)} test={len(te)} -> {local}")

        # categories + data.yaml
        domain = infer_domain_name(fd.name)
        cat_fn = domain_to_cat_filename(domain)
        cfg_dir = cfg_root / fd.name
        cats_p = cfg_dir / cat_fn
        yaml_p = cfg_dir / "data.yaml"

        if args.force or (not cats_p.exists()):
            rows = read_jsonl(train_all)
            cats = extract_categories_from_train(rows)
            if not cats:
                print(f"[Warn] {fd.name}: extracted 0 categories from {train_all} (check file format)")
            write_categories_file(cats_p, cats)
            print(f"[Cats] {fd.name}: {cats_p} (INVALID last)")

        if args.force or (not yaml_p.exists()):
            write_data_yaml(
                yaml_p,
                train_p=Path(args.data_dir) / fd.name / "local/train.jsonl",
                valid_p=Path(args.data_dir) / fd.name / "local/valid.jsonl",
                test_p=Path(args.data_dir) / fd.name / "local/test.jsonl",
                dev_p=Path(args.data_dir) / fd.name / f"official/{dev.name}",
                train_all_p=Path(args.data_dir) / fd.name / f"official/{train_all.name}",
                cats_p=Path(args.configs_dir) / fd.name / cat_fn
            )
            print(f"[YAML] {fd.name}: {yaml_p}")

    print("[Done]")

if __name__ == "__main__":
    main()

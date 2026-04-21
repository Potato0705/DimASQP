"""Format / safety tests for the LLM pseudo-labeling pipeline.

These tests run without an OPENROUTER_API_KEY (they never hit the network)
and exercise:
  * the output-file format produced by a simulated pseudo-label run
  * the dev/test leak guard in data/llm_pseudo_labeler.py
  * the merger in data/merge_pseudo_with_gold.py
  * prompts.build_pseudo_label_prompt returns the expected structure
  * openrouter_client.parse_json_content recovers JSON from fenced output
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from llm.openrouter_client import parse_json_content  # noqa: E402
from llm.prompts import build_pseudo_label_prompt  # noqa: E402


# ------------------------------------------------------------------ prompts
def test_build_pseudo_label_prompt_shape():
    shots = [
        {"Text": "the pasta was great", "Quadruplet": [
            {"Aspect": "pasta", "Opinion": "great",
             "Category": "FOOD#QUALITY", "VA": "8.00#6.00"}
        ]},
    ]
    msgs = build_pseudo_label_prompt(
        sentence="service was slow",
        category_list=["FOOD#QUALITY", "SERVICE#GENERAL"],
        shots=shots,
    )
    # system + shot(user+assistant) + final user = 1 + 2 + 1 = 4
    assert len(msgs) == 4
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert "service was slow" in msgs[-1]["content"]
    # The assistant shot must be valid JSON with the schema we enforce.
    shot_json = json.loads(msgs[2]["content"])
    assert "quadruplets" in shot_json
    assert shot_json["quadruplets"][0]["category"] == "FOOD#QUALITY"


# ---------------------------------------------------------- JSON extraction
def test_parse_json_content_plain():
    out = parse_json_content('{"quadruplets": []}')
    assert out == {"quadruplets": []}


def test_parse_json_content_fenced():
    raw = textwrap.dedent("""
        ```json
        {"quadruplets": [{"aspect": "x", "opinion": "y",
                          "category": "FOOD#QUALITY",
                          "valence": 6.0, "arousal": 5.0}]}
        ```
    """).strip()
    out = parse_json_content(raw)
    assert out is not None
    assert out["quadruplets"][0]["category"] == "FOOD#QUALITY"


def test_parse_json_content_trailing_prose():
    raw = 'sure, here it is: {"quadruplets": []} — hope that helps!'
    out = parse_json_content(raw)
    assert out == {"quadruplets": []}


def test_parse_json_content_returns_none_on_garbage():
    assert parse_json_content("") is None
    assert parse_json_content("nope") is None


# ------------------------------------------------ pseudo labeler leak guard
def _run_labeler(*args):
    """Invoke data/llm_pseudo_labeler.py as a subprocess, return CompletedProcess."""
    cmd = [sys.executable, str(REPO_ROOT / "data" / "llm_pseudo_labeler.py"), *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)


def test_leak_guard_rejects_dev_path(tmp_path):
    # Needs a plausible gold jsonl so category collection doesn't crash first.
    gold_jsonl = tmp_path / "fake_train.jsonl"
    gold_jsonl.write_text(json.dumps({
        "ID": "x", "Text": "good food",
        "Quadruplet": [{"Aspect": "food", "Opinion": "good",
                         "Category": "FOOD#QUALITY", "VA": "7.0#6.0"}]
    }) + "\n", encoding="utf-8")
    dev_path = tmp_path / "something_dev.jsonl"
    dev_path.write_text("", encoding="utf-8")
    out_prefix = tmp_path / "out"
    res = _run_labeler(
        "--task_domain", "eng_restaurant",
        "--gold_jsonl", str(gold_jsonl),
        "--source_file", str(dev_path),
        "--out_prefix", str(out_prefix),
        "--dry_run",
    )
    assert res.returncode != 0, res.stdout + res.stderr
    assert "leak" in (res.stderr + res.stdout).lower()


# --------------------------------------------- pseudo labeler dry-run format
def test_dry_run_emits_parseable_outputs(tmp_path, monkeypatch):
    # Build a minimal gold jsonl (one sentence, one quad).
    gold_jsonl = tmp_path / "fake_train.jsonl"
    gold_jsonl.write_text(json.dumps({
        "ID": "g1", "Text": "good food",
        "Quadruplet": [{"Aspect": "food", "Opinion": "good",
                         "Category": "FOOD#QUALITY", "VA": "7.0#6.0"}]
    }) + "\n", encoding="utf-8")

    out_prefix = tmp_path / "pseudo_out"

    # --dry_run produces empty outputs (no cache hits), but exits cleanly.
    res = _run_labeler(
        "--task_domain", "eng_restaurant",
        "--gold_jsonl", str(gold_jsonl),
        "--source_file", str(gold_jsonl),
        "--out_prefix", str(out_prefix),
        "--dry_run",
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert (out_prefix.with_suffix(".txt")).exists()
    assert (out_prefix.with_suffix(".jsonl")).exists()
    stats = json.loads((tmp_path / "pseudo_out_stats.json").read_text(encoding="utf-8"))
    assert stats["task_domain"] == "eng_restaurant"
    assert stats["n_sentences"] == 1
    # dry_run with an empty cache should produce zero rows.
    assert stats["n_written"] == 0
    assert stats["n_llm_calls"] == 0


# ---------------------------------------------------------- merger behavior
def test_merger_preserves_format_and_counts(tmp_path):
    gold_txt = tmp_path / "gold.txt"
    gold_txt.write_text(
        'sentence one####[["FOOD#QUALITY", "0,8", "9,12", "7.00#6.00"]]\n'
        'sentence two####[["SERVICE#GENERAL", "0,8", "9,12", "3.00#7.00"]]\n',
        encoding="utf-8",
    )
    pseudo_txt = tmp_path / "pseudo.txt"
    pseudo_txt.write_text(
        'pseudo one####[["FOOD#QUALITY", "0,6", "7,10", "6.00#5.00"]]\n'
        'pseudo two####[["FOOD#QUALITY", "0,6", "7,10", "6.00#5.00"]]\n'
        'pseudo three####[["FOOD#QUALITY", "0,8", "9,12", "6.00#5.00"]]\n'
        'pseudo four####[["FOOD#QUALITY", "0,7", "8,11", "6.00#5.00"]]\n',
        encoding="utf-8",
    )
    out_txt = tmp_path / "merged.txt"
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "data" / "merge_pseudo_with_gold.py"),
         "--gold", str(gold_txt),
         "--pseudo", str(pseudo_txt),
         "--ratio", "0.5", "--seed", "0",
         "--out", str(out_txt)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    lines = [ln for ln in out_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # 2 gold + floor(0.5 * 4)=2 pseudo = 4 rows total
    assert len(lines) == 4
    # Every row must still match the "<text>####<json>" format.
    for ln in lines:
        assert "####" in ln
        txt, quads = ln.split("####", 1)
        parsed = json.loads(quads)
        assert isinstance(parsed, list) and len(parsed) >= 1
        assert len(parsed[0]) == 4  # [category, aspect_span, opinion_span, va]


def test_merger_leak_guard(tmp_path):
    bad = tmp_path / "eng_restaurant_dev.txt"
    bad.write_text("x####[]\n", encoding="utf-8")
    gold = tmp_path / "gold.txt"
    gold.write_text("x####[]\n", encoding="utf-8")
    out = tmp_path / "merged.txt"
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "data" / "merge_pseudo_with_gold.py"),
         "--gold", str(gold), "--pseudo", str(bad), "--out", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert res.returncode != 0
    assert "dev" in (res.stderr + res.stdout).lower()


# -------------------------------------- real gold files load + align
@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "v2" / "eng" / "eng_restaurant_train.txt").exists(),
    reason="v2 data not yet generated",
)
def test_real_gold_txt_matches_sidecar():
    txt_path = REPO_ROOT / "data" / "v2" / "eng" / "eng_restaurant_train.txt"
    side_path = REPO_ROOT / "data" / "v2" / "eng" / "eng_restaurant_train_sidecar.json"
    lines = [ln for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sidecar = json.loads(side_path.read_text(encoding="utf-8"))
    assert len(lines) == len(sidecar)
    # Row 0 sanity: sidecar line_index consistent.
    assert sidecar[0]["line_index"] == 0

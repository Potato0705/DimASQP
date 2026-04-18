"""Tests for the CCA (Compositional Category Augmentation) pipeline.

Tests run without an OPENROUTER_API_KEY (no network calls) and exercise:
  * category_analysis.py gap report generation
  * CCA prompt builders (entity/attribute grounding, compositional gen, cross-verify)
  * cca_generator.py dry-run and output format
  * eval_category_coverage.py CCR/ZCR computation
  * confusion_analysis.py confusion matrix
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from llm.prompts import (  # noqa: E402
    build_entity_grounding_prompt,
    build_attribute_grounding_prompt,
    build_cca_generation_prompt,
    build_cross_verify_prompt,
)
from llm.openrouter_client import parse_json_content  # noqa: E402


# ---------------------------------------------------------- prompt builders
def _make_gold_examples():
    return [
        {
            "Text": "the pasta was delicious and fresh",
            "Quadruplet": [
                {"Aspect": "pasta", "Opinion": "delicious",
                 "Category": "FOOD#QUALITY", "VA": "8.00#6.00"},
            ],
        },
        {
            "Text": "reasonable prices for the area",
            "Quadruplet": [
                {"Aspect": "prices", "Opinion": "reasonable",
                 "Category": "RESTAURANT#PRICES", "VA": "7.00#4.00"},
            ],
        },
        {
            "Text": "the wine list was impressive with many options",
            "Quadruplet": [
                {"Aspect": "wine list", "Opinion": "impressive",
                 "Category": "DRINKS#STYLE_OPTIONS", "VA": "7.50#6.50"},
            ],
        },
    ]


def test_entity_grounding_prompt_shape():
    examples = _make_gold_examples()
    msgs = build_entity_grounding_prompt("FOOD", examples[:2], max_examples=5)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "FOOD" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "FOOD" in msgs[1]["content"]


def test_attribute_grounding_prompt_shape():
    examples = _make_gold_examples()
    msgs = build_attribute_grounding_prompt("QUALITY", examples[:2], max_examples=5)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "QUALITY" in msgs[0]["content"]


def test_cca_generation_prompt_shape():
    examples = _make_gold_examples()
    msgs = build_cca_generation_prompt(
        entity="AMBIENCE",
        attribute="STYLE_OPTIONS",
        entity_summary='{"entity": "AMBIENCE", "typical_aspects": ["decor"]}',
        attribute_summary='{"attribute": "STYLE_OPTIONS", "measures": "variety"}',
        anchor_examples=examples,
        n_generate=5,
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "AMBIENCE#STYLE_OPTIONS" in msgs[0]["content"]
    assert "5" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "AMBIENCE#STYLE_OPTIONS" in msgs[1]["content"]


def test_cross_verify_prompt_shape():
    all_cats = ["FOOD#QUALITY", "SERVICE#GENERAL", "AMBIENCE#STYLE_OPTIONS"]
    msgs = build_cross_verify_prompt(
        sentence="the decor was lovely",
        aspect="decor",
        opinion="lovely",
        candidate_category="AMBIENCE#STYLE_OPTIONS",
        all_categories=all_cats,
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "FOOD#QUALITY" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "decor" in msgs[1]["content"]
    assert "lovely" in msgs[1]["content"]


# ------------------------------------------------- category analysis script
def _make_gold_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def test_category_analysis_produces_gap_report(tmp_path):
    train_entries = [
        {"ID": "t1", "Text": "good food", "Quadruplet": [
            {"Aspect": "food", "Opinion": "good", "Category": "FOOD#QUALITY", "VA": "7.0#6.0"}
        ]},
        {"ID": "t2", "Text": "nice service", "Quadruplet": [
            {"Aspect": "service", "Opinion": "nice", "Category": "SERVICE#GENERAL", "VA": "7.0#5.0"}
        ]},
    ]
    dev_entries = [
        {"ID": "d1", "Text": "stylish ambience", "Quadruplet": [
            {"Aspect": "ambience", "Opinion": "stylish", "Category": "AMBIENCE#STYLE_OPTIONS", "VA": "7.0#5.0"}
        ]},
    ]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_gold_jsonl(data_dir / "test_domain_train.jsonl", train_entries)
    _make_gold_jsonl(data_dir / "test_domain_dev.jsonl", dev_entries)
    _make_gold_jsonl(data_dir / "test_domain_test.jsonl", [])

    out_dir = tmp_path / "output"
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "category_analysis.py"),
         "--task_domain", "test_domain",
         "--data_dir", str(data_dir),
         "--out_dir", str(out_dir)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert res.returncode == 0, res.stderr

    report_path = out_dir / "category_gap_report_test_domain.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["task_domain"] == "test_domain"
    assert len(report["gaps"]) > 0

    zero_shot = [g for g in report["gaps"] if g["status"] == "zero_shot"]
    assert any(g["category"] == "AMBIENCE#STYLE_OPTIONS" for g in zero_shot)

    csv_path = out_dir / "category_heatmap_test_domain.csv"
    assert csv_path.exists()


# ------------------------------------------------- CCA generator dry-run
def test_cca_generator_dry_run(tmp_path):
    train_entries = [
        {"ID": "t1", "Text": "good food", "Quadruplet": [
            {"Aspect": "food", "Opinion": "good", "Category": "FOOD#QUALITY", "VA": "7.0#6.0"}
        ]},
        {"ID": "t2", "Text": "nice ambience", "Quadruplet": [
            {"Aspect": "ambience", "Opinion": "nice", "Category": "AMBIENCE#GENERAL", "VA": "7.0#5.0"}
        ]},
    ]
    gold_jsonl = tmp_path / "train.jsonl"
    _make_gold_jsonl(gold_jsonl, train_entries)

    gap_report = {
        "task_domain": "test_domain",
        "entities": ["FOOD", "AMBIENCE"],
        "attributes": ["QUALITY", "GENERAL", "STYLE_OPTIONS"],
        "n_train_sentences": 2,
        "gaps": [
            {"entity": "AMBIENCE", "attribute": "STYLE_OPTIONS",
             "category": "AMBIENCE#STYLE_OPTIONS", "train_count": 0,
             "dev_count": 1, "test_count": 5, "status": "zero_shot", "priority": 106},
        ],
    }
    gap_path = tmp_path / "gap_report.json"
    gap_path.write_text(json.dumps(gap_report), encoding="utf-8")

    out_prefix = tmp_path / "cca_out"
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "data" / "cca_generator.py"),
         "--task_domain", "test_domain",
         "--gap_report", str(gap_path),
         "--gold_jsonl", str(gold_jsonl),
         "--out_prefix", str(out_prefix),
         "--nrows", "1",
         "--dry_run", "--verbose"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert res.returncode == 0, res.stderr + res.stdout

    assert (out_prefix.with_suffix(".txt")).exists()
    assert (out_prefix.with_suffix(".jsonl")).exists()
    stats = json.loads((tmp_path / "cca_out_stats.json").read_text(encoding="utf-8"))
    assert stats["pipeline"] == "CCA"
    assert stats["n_target_categories"] == 1
    assert stats["n_llm_calls"] == 0


# ------------------------------------------- eval_category_coverage basics
def test_eval_coverage_computes_ccr(tmp_path):
    gold = [
        {"ID": "1", "Text": "good food", "Quadruplet": [
            {"Aspect": "food", "Opinion": "good", "Category": "FOOD#QUALITY", "VA": "7.0#6.0"}
        ]},
        {"ID": "2", "Text": "nice service", "Quadruplet": [
            {"Aspect": "service", "Opinion": "nice", "Category": "SERVICE#GENERAL", "VA": "7.0#5.0"}
        ]},
        {"ID": "3", "Text": "lovely decor", "Quadruplet": [
            {"Aspect": "decor", "Opinion": "lovely", "Category": "AMBIENCE#STYLE_OPTIONS", "VA": "7.0#5.0"}
        ]},
    ]
    pred = [
        {"ID": "1", "Text": "good food", "Quadruplet": [
            {"Aspect": "food", "Opinion": "good", "Category": "FOOD#QUALITY", "VA": "7.2#5.8"}
        ]},
        {"ID": "2", "Text": "nice service", "Quadruplet": [
            {"Aspect": "service", "Opinion": "nice", "Category": "SERVICE#GENERAL", "VA": "6.8#5.2"}
        ]},
        {"ID": "3", "Text": "lovely decor", "Quadruplet": []},
    ]
    train = [
        {"ID": "t1", "Text": "x", "Quadruplet": [
            {"Aspect": "x", "Opinion": "y", "Category": "FOOD#QUALITY", "VA": "5#5"}
        ]},
        {"ID": "t2", "Text": "x", "Quadruplet": [
            {"Aspect": "x", "Opinion": "y", "Category": "SERVICE#GENERAL", "VA": "5#5"}
        ]},
    ]
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    train_path = tmp_path / "train.jsonl"
    out_path = tmp_path / "results.json"
    _make_gold_jsonl(gold_path, gold)
    _make_gold_jsonl(pred_path, pred)
    _make_gold_jsonl(train_path, train)

    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "eval_category_coverage.py"),
         "--pred", str(pred_path),
         "--gold", str(gold_path),
         "--train", str(train_path),
         "--out", str(out_path)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert res.returncode == 0, res.stderr

    results = json.loads(out_path.read_text(encoding="utf-8"))
    assert "ccr" in results
    assert "zcr" in results
    assert "seen" in results
    assert "unseen" in results
    assert results["seen"]["cF1"] > 0
    assert results["unseen"]["cF1"] == 0
    assert "AMBIENCE#STYLE_OPTIONS" in results["unseen_categories"]


# ------------------------------------------- confusion_analysis basics
def test_confusion_analysis_runs(tmp_path):
    gold = [
        {"ID": "1", "Text": "good food", "Quadruplet": [
            {"Aspect": "food", "Opinion": "good", "Category": "FOOD#QUALITY", "VA": "7.0#6.0"}
        ]},
    ]
    pred = [
        {"ID": "1", "Text": "good food", "Quadruplet": [
            {"Aspect": "food", "Opinion": "good", "Category": "FOOD#PRICES", "VA": "7.0#6.0"}
        ]},
    ]
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _make_gold_jsonl(gold_path, gold)
    _make_gold_jsonl(pred_path, pred)

    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "confusion_analysis.py"),
         "--pred", str(pred_path),
         "--gold", str(gold_path),
         "--task_domain", "test",
         "--out_dir", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert res.returncode == 0, res.stderr

    out_path = tmp_path / "confusion_matrix_test.json"
    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["summary"]["misclassified"] == 1
    assert data["confusion_matrix"]["FOOD#QUALITY"]["FOOD#PRICES"] == 1

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_SEEDS = [7, 42, 66, 123, 2045]


GROUP_DESCRIPTIONS = {
    "table1_position": "Table1 baseline position",
    "table1_span_pair": "Table1 span-pair baseline",
    "table1_og_full": "Table1 opinion-guided full",
    "ablation_plain_sp": "Disentangle plain span-pair",
    "ablation_sp_prior": "Disentangle span-pair + prior loss",
    "ablation_og_no_prior": "Disentangle opinion-guided w/o prior loss",
    "ablation_og_full": "Disentangle opinion-guided full",
}


GROUP_ALIASES = {
    "position": "table1_position",
    "span_pair": "table1_span_pair",
    "og_full": "table1_og_full",
    "plain_sp": "ablation_plain_sp",
    "sp_prior": "ablation_sp_prior",
    "sp_prior_loss": "ablation_sp_prior",
    "og_no_prior": "ablation_og_no_prior",
}


@dataclass
class RunInfo:
    run_dir: Path
    task_domain: str
    seed: int
    va_mode: str
    weight_va_prior: float
    use_va_prior_aux: bool
    namespace: str
    output_dir: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["run_dir"] = str(self.run_dir)
        return data


def normalize_group_name(group: str) -> str:
    if group in GROUP_DESCRIPTIONS:
        return group
    if group in GROUP_ALIASES:
        return GROUP_ALIASES[group]
    raise KeyError(f"Unknown experiment group: {group}")


def _to_bool(value) -> bool:
    return bool(value)


def _to_float(value) -> float:
    if value is None:
        return 0.0
    return float(value)


def _approx(value: float, target: float, tol: float = 1e-6) -> bool:
    return abs(float(value) - float(target)) <= tol


def _namespace_for(run_dir: Path, runs_root: Path) -> str:
    rel = run_dir.relative_to(runs_root)
    return "root" if len(rel.parts) == 1 else rel.parts[0]


def _run_mtime(run_dir: Path) -> float:
    for name in ("best_score.json", "best_model.pt", "train_history.json", "args.json"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate.stat().st_mtime
    return run_dir.stat().st_mtime


def iter_runs(runs_root: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for args_path in runs_root.rglob("args.json"):
        run_dir = args_path.parent
        best_score = run_dir / "best_score.json"
        best_model = run_dir / "best_model.pt"
        if not best_score.exists() or not best_model.exists():
            continue
        with args_path.open("r", encoding="utf-8-sig") as f:
            args = json.load(f)
        runs.append(
            RunInfo(
                run_dir=run_dir,
                task_domain=args["task_domain"],
                seed=int(args["seed"]),
                va_mode=args.get("va_mode", "position"),
                weight_va_prior=_to_float(args.get("weight_va_prior")),
                use_va_prior_aux=_to_bool(args.get("use_va_prior_aux", False)),
                namespace=_namespace_for(run_dir, runs_root),
                output_dir=args.get("output_dir", ""),
            )
        )
    return runs


def matches_group(run: RunInfo, group: str) -> bool:
    group = normalize_group_name(group)

    if group == "table1_position":
        return run.namespace == "root" and run.va_mode == "position"
    if group == "table1_span_pair":
        return run.namespace == "root" and run.va_mode == "span_pair"
    if group == "table1_og_full":
        return run.namespace == "root" and run.va_mode == "opinion_guided"
    if group == "ablation_plain_sp":
        return (
            run.va_mode == "span_pair"
            and not run.use_va_prior_aux
            and _approx(run.weight_va_prior, 0.0)
        )
    if group == "ablation_sp_prior":
        return (
            run.va_mode == "span_pair"
            and run.use_va_prior_aux
            and _approx(run.weight_va_prior, 0.3)
        )
    if group == "ablation_og_no_prior":
        return run.va_mode == "opinion_guided" and _approx(run.weight_va_prior, 0.0)
    if group == "ablation_og_full":
        return (
            run.namespace != "root"
            and run.va_mode == "opinion_guided"
            and _approx(run.weight_va_prior, 0.3)
        )
    return False


def select_latest_runs(
    runs_root: Path,
    group: str,
    task_domain: str,
    seeds: list[int] | None = None,
) -> tuple[list[RunInfo], list[int]]:
    group = normalize_group_name(group)
    wanted_seeds = list(seeds or DEFAULT_SEEDS)
    selected: list[RunInfo] = []
    missing: list[int] = []
    runs = iter_runs(runs_root)

    for seed in wanted_seeds:
        candidates = [
            run
            for run in runs
            if run.task_domain == task_domain
            and run.seed == seed
            and matches_group(run, group)
        ]
        if not candidates:
            missing.append(seed)
            continue
        selected.append(max(candidates, key=lambda run: _run_mtime(run.run_dir)))

    selected.sort(key=lambda run: run.seed)
    return selected, missing


def load_best_score(run_dir: Path) -> dict:
    best_score_path = run_dir / "best_score.json"
    if not best_score_path.exists():
        return {}
    with best_score_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Batch-evaluate RL-Games checkpoints from local and CARC training logs.

Scans ``logs/rl_games/Forge/`` and ``carc_logs/`` for ``<run>/nn/Forge.pth``
(best checkpoint), auto-detects the task from ``<run>/params/env.yaml``, and
launches ``play.py --max_completed_episodes`` for each new checkpoint.

A JSON registry remembers what was already evaluated so re-runs skip them.
Results go into a concise CSV with checkpoint identity and experiment context.

Usage (from repo root)::

    # Dry-run — list what would be evaluated
    python batch_eval.py --dry-run

    # Evaluate all best checkpoints (default)
    python batch_eval.py --headless --num-envs 128

    # Also evaluate intermediate epoch checkpoints
    python batch_eval.py --all --headless --num-envs 128

    # Re-evaluate everything
    python batch_eval.py --force --headless --num-envs 128
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_LINE_RE = re.compile(
    r"episodes=(\d+)\s+successes=(\d+)\s+success_rate=([\d.]+%)",
)

TASK_GYM_IDS: dict[str, dict[str, str]] = {
    "rj45_insert": {
        "kuka": "Isaac-Forge-RJ45Insert-Kuka-Direct-v0",
        "franka": "Isaac-Forge-RJ45Insert-Direct-v0",
    },
    "box_lid_insert": {
        "kuka": "Isaac-Forge-BoxLidInsert-Kuka-Direct-v0",
        "franka": "Isaac-Forge-BoxLidInsert-Direct-v0",
    },
    "bnc_insert": {
        "kuka": "Isaac-Forge-BNCSmallInsert-Kuka-Direct-v0",
        "franka": "Isaac-Forge-BNCSmallInsert-Direct-v0",
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _read_yaml_field(path: Path, field: str) -> str | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                m = re.match(rf"^{re.escape(field)}:\s*(.+)", line.strip())
                if m:
                    return m.group(1).strip()
    except OSError:
        pass
    return None


def _detect_robot(env_yaml: Path) -> str:
    """Return 'kuka' or 'franka' by checking for 'kuka_arm:' in the actuators section."""
    if not env_yaml.is_file():
        return "franka"
    try:
        with env_yaml.open("r", encoding="utf-8") as f:
            for line in f:
                if "kuka_arm:" in line:
                    return "kuka"
    except OSError:
        pass
    return "franka"


def _read_yaml_nested(path: Path, *keys: str) -> str | None:
    if not path.is_file():
        return None
    try:
        target_depth = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                key_match = re.match(r"^(\s*)([\w_]+):\s*(.*)", line)
                if not key_match:
                    continue
                cur_indent = len(key_match.group(1))
                cur_key = key_match.group(2)
                cur_val = key_match.group(3).strip()
                if target_depth < len(keys) and cur_key == keys[target_depth]:
                    if target_depth == len(keys) - 1:
                        return cur_val if cur_val else None
                    target_depth += 1
                elif target_depth > 0 and cur_indent <= (target_depth - 1) * 2:
                    target_depth = 0
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Experiment info (lightweight extraction from YAML, no PyYAML needed)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentInfo:
    task_name: str
    robot: str  # "kuka" or "franka"
    gym_id: str
    run_dir: str
    run_timestamp: str
    source: str
    success_threshold: str
    engage_threshold: str
    init_mode: str
    num_envs_train: str


def _extract_experiment_info(run_dir: Path) -> ExperimentInfo | None:
    env_yaml = run_dir / "params" / "env.yaml"
    task_name = _read_yaml_field(env_yaml, "task_name")
    if not task_name:
        return None
    robot = _detect_robot(env_yaml)
    gym_ids = TASK_GYM_IDS.get(task_name)
    if not gym_ids:
        return None
    gym_id = gym_ids.get(robot, "")
    if not gym_id:
        return None
    return ExperimentInfo(
        task_name=task_name,
        robot=robot,
        gym_id=gym_id,
        run_dir=str(run_dir),
        run_timestamp=run_dir.name,
        source="carc" if "carc_logs" in str(run_dir) else "local",
        success_threshold=_read_yaml_nested(env_yaml, "task", "success_threshold") or "",
        engage_threshold=_read_yaml_nested(env_yaml, "task", "engage_threshold") or "",
        init_mode=_read_yaml_nested(env_yaml, "task", "init_mode") or "",
        num_envs_train=_read_yaml_nested(env_yaml, "scene", "num_envs") or "",
    )


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    pth_path: Path
    run_dir: Path
    info: ExperimentInfo
    is_best: bool = False

    @property
    def name(self) -> str:
        return self.pth_path.name

    @property
    def epoch_reward(self) -> str:
        m = re.match(r"last_Forge_ep_(\d+)_rew_(\d+\.?\d*)", self.pth_path.stem)
        if m:
            return f"ep{m.group(1)}_rew{m.group(2)}"
        if self.pth_path.name == "Forge.pth":
            return "best"
        return ""


def _discover_checkpoints(log_roots: list[Path], best_only: bool) -> list[Checkpoint]:
    out: list[Checkpoint] = []
    seen: set[str] = set()
    for log_root in log_roots:
        if not log_root.is_dir():
            continue
        for run_dir in sorted(log_root.iterdir()):
            if not run_dir.is_dir():
                continue
            nn = run_dir / "nn"
            if not nn.is_dir():
                continue
            info = _extract_experiment_info(run_dir)
            if info is None:
                continue
            pths = [nn / "Forge.pth"] if best_only else sorted(nn.glob("*.pth"))
            for pth in pths:
                if not pth.is_file():
                    continue
                real = os.path.realpath(pth)
                if real in seen:
                    continue
                seen.add(real)
                out.append(Checkpoint(
                    pth_path=pth.resolve(), run_dir=run_dir.resolve(),
                    info=info, is_best=(pth.name == "Forge.pth"),
                ))
    return out


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse_eval(text: str) -> dict[str, Any] | None:
    last = None
    for line in text.splitlines():
        if "[EVAL]" not in line or "success_rate=" not in line:
            continue
        m = EVAL_LINE_RE.search(line)
        if not m:
            continue
        last = {
            "total_episodes": int(m.group(1)),
            "total_successes": int(m.group(2)),
            "success_rate": float(m.group(3).rstrip("%")) / 100.0,
            "success_rate_str": m.group(3),
        }
    return last


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _load_registry(path: Path) -> dict:
    if not path.is_file():
        return {"version": 1, "entries": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _key(gym_id: str, pth: Path) -> str:
    return f"{gym_id}::{os.path.realpath(pth)}"


# ---------------------------------------------------------------------------
# Play launcher  (python play.py ..., not isaaclab.sh -p)
# ---------------------------------------------------------------------------

@dataclass
class PlayResult:
    exit_code: int
    stdout: str
    stderr: str


def _run_play(
    *, play_py: Path, repo: Path, gym_id: str, checkpoint: Path,
    num_envs: int | None, headless: bool, max_ep: int,
    seed: int | None, extra: list[str], timeout: int | None,
) -> PlayResult:
    cmd = [
        sys.executable, str(play_py),
        "--task", gym_id,
        "--checkpoint", str(checkpoint),
        "--max_completed_episodes", str(max_ep),
    ]
    if num_envs is not None:
        cmd += ["--num_envs", str(num_envs)]
    if headless:
        cmd.append("--headless")
    if seed is not None:
        cmd += ["--seed", str(seed)]
    cmd += extra

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print(f"  [CMD] {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd, cwd=str(repo), env=env,
            capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        return PlayResult(proc.returncode, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = (e.stderr or "") if isinstance(e.stderr, str) else str(e)
        return PlayResult(124, out, err + "\n[batch] timeout\n")


# ---------------------------------------------------------------------------
# CSV — concise columns
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "run_timestamp",
    "source",
    "task_name",
    "robot",
    "checkpoint",
    "success_rate",
    "total_episodes",
    "total_successes",
    "success_threshold",
    "init_mode",
    "num_envs_train",
    "checkpoint_path",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-root", type=str, action="append", default=None,
        help="Log root(s) (default: logs/rl_games/Forge + carc_logs). Repeatable.",
    )
    parser.add_argument("--task", type=str, default=None, help="Override gym task id.")
    parser.add_argument("--num-envs", type=int, default=None, help="Forwarded to play.py.")
    parser.add_argument("--headless", action="store_true", help="Forwarded to play.py.")
    parser.add_argument(
        "--max-completed-episodes", type=int, default=2048,
        help="Episodes to evaluate per checkpoint (default: 2048).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Forwarded to play.py.")
    parser.add_argument("--timeout-sec", type=int, default=None, help="Wall-clock timeout per checkpoint.")
    parser.add_argument("--force", action="store_true", help="Re-evaluate all.")
    parser.add_argument(
        "--force-checkpoint", type=str, action="append", default=[],
        help="Re-evaluate this specific checkpoint (repeatable).",
    )
    parser.add_argument(
        "--checkpoint", type=str, action="append", default=[],
        help="Include this .pth even if outside --log-root (repeatable).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate ALL checkpoints per run (default: only Forge.pth best).",
    )
    parser.add_argument("--dry-run", action="store_true", help="List what would run.")
    parser.add_argument("--registry", type=str, default=None)
    parser.add_argument("--csv-out", type=str, default=None)
    parser.add_argument(
        "extra_play_args", nargs="*",
        help="Extra args forwarded to play.py.",
    )
    args = parser.parse_args()

    repo = _repo_root()
    results_dir = repo / "eval_results"

    if args.log_root is None:
        log_roots = [repo / "logs" / "rl_games" / "Forge", repo / "carc_logs"]
    else:
        log_roots = [Path(p) if os.path.isabs(p) else repo / p for p in args.log_root]
    log_roots = [p.resolve() for p in log_roots]

    registry_path = Path(args.registry).resolve() if args.registry else results_dir / "play_eval_registry.json"
    csv_path = Path(args.csv_out).resolve() if args.csv_out else results_dir / "play_eval_results.csv"
    play_py = repo / "scripts" / "reinforcement_learning" / "rl_games" / "play.py"

    if not play_py.is_file():
        print(f"[ERROR] play.py not found: {play_py}", file=sys.stderr)
        return 1

    best_only = not args.all
    checkpoints = _discover_checkpoints(log_roots, best_only)

    # merge explicit paths
    seen = {os.path.realpath(c.pth_path) for c in checkpoints}
    for p_str in args.checkpoint + args.force_checkpoint:
        p = Path(p_str).expanduser().resolve()
        if os.path.realpath(p) in seen:
            continue
        seen.add(os.path.realpath(p))
        info = _extract_experiment_info(p.parent.parent)
        if info is None:
            print(f"[WARN] Cannot detect task for {p}, skipping")
            continue
        checkpoints.append(Checkpoint(p, p.parent.parent, info, is_best=(p.name == "Forge.pth")))

    if args.task:
        for c in checkpoints:
            c.info.gym_id = args.task

    if not checkpoints:
        print("[WARN] No checkpoints found")
        return 0

    registry = _load_registry(registry_path)
    entries = registry.setdefault("entries", {})
    force_set = {os.path.realpath(Path(p).resolve()) for p in args.force_checkpoint}

    to_run = []
    for c in checkpoints:
        k = _key(c.info.gym_id, c.pth_path)
        if args.force or os.path.realpath(c.pth_path) in force_set or k not in entries:
            to_run.append(c)

    print(f"[INFO] log_roots    = {[str(p) for p in log_roots]}")
    print(f"[INFO] registry     = {registry_path}")
    print(f"[INFO] csv          = {csv_path}")
    print(f"[INFO] best_only    = {best_only}")
    print(f"[INFO] discovered   = {len(checkpoints)}")
    print(f"[INFO] new to run   = {len(to_run)}")
    print(f"[INFO] already done = {len(checkpoints) - len(to_run)}")

    if args.dry_run:
        fmt = "  {src:5s} | {ts:19s} | {task:16s} | {robot:6s} | {ckpt}"
        print(fmt.format(src="SRC", ts="RUN", task="TASK", robot="ROBOT", ckpt="CHECKPOINT"))
        print("  " + "-" * 90)
        for c in to_run:
            print(fmt.format(
                src=c.info.source, ts=c.info.run_timestamp,
                task=c.info.task_name, robot=c.info.robot, ckpt=c.name,
            ))
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_new = not csv_path.is_file()

    for i, c in enumerate(to_run):
        k = _key(c.info.gym_id, c.pth_path)
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(to_run)}] {c.info.source.upper()} | {c.info.run_timestamp} | {c.info.task_name} | {c.info.robot} | {c.name}")

        result = _run_play(
            play_py=play_py, repo=repo,
            gym_id=c.info.gym_id, checkpoint=c.pth_path,
            num_envs=args.num_envs, headless=args.headless,
            max_ep=args.max_completed_episodes, seed=args.seed,
            extra=list(args.extra_play_args), timeout=args.timeout_sec,
        )

        stats = _parse_eval(result.stdout + "\n" + result.stderr)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        if stats:
            print(f"  >> success_rate={stats['success_rate_str']}  "
                  f"({stats['total_successes']}/{stats['total_episodes']})")
        else:
            print("  >> FAILED — no [EVAL] line found in output")
            if result.stderr:
                for line in result.stderr.strip().splitlines()[-5:]:
                    print(f"     {line}")

        # registry update
        rec: dict[str, Any] = {
            "gym_id": c.info.gym_id, "task_name": c.info.task_name,
            "checkpoint": str(c.pth_path), "source": c.info.source,
            "evaluated_at": now, "exit_code": result.exit_code,
        }
        if stats:
            rec.update(stats)
        entries[k] = rec
        _save_registry(registry_path, registry)

        # csv append
        hdr = csv_new
        csv_new = False
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if hdr:
                w.writeheader()
            w.writerow({
                "run_timestamp": c.info.run_timestamp,
                "source": c.info.source,
                "task_name": c.info.task_name,
                "robot": c.info.robot,
                "checkpoint": c.name,
                "success_rate": stats["success_rate"] if stats else "",
                "total_episodes": stats["total_episodes"] if stats else "",
                "total_successes": stats["total_successes"] if stats else "",
                "success_threshold": c.info.success_threshold,
                "init_mode": c.info.init_mode,
                "num_envs_train": c.info.num_envs_train,
                "checkpoint_path": str(c.pth_path),
            })

        if result.exit_code != 0:
            print(f"  [WARN] exit_code={result.exit_code}")

    print(f"\n{'='*70}")
    print(f"[DONE] Evaluated {len(to_run)} checkpoint(s)")
    print(f"  registry : {registry_path}")
    print(f"  csv      : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sequential E0–E5 driver with skip-complete + resume. Never writes Branch A/B/C artifacts.

Default profile (`reduced`) is the thesis-feasible ladder on one GPU:
E0 five seeds @ 224 → E1 seed 42 @ 384 and 456 → pick val winner →
E2/E3 seed 42 at that resolution → soft maps → E4 → compare on val →
E5 only if E4 val AUC beats E3.

Canonical test stays locked. E6/E7 stay blocked. Profile `e8` runs source-balanced E8 + LOSO.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.classify.next_architecture.config import load_next_config, verify_canonical_split
from src.utils.common import ROOT

STAGES = ("audit", "e0", "e1", "pick_resolution", "e2", "e3", "maps", "e4", "compare", "e5")
OOM_MARKERS = (
    "out of memory",
    "mps backend out of memory",
    "cuda out of memory",
    "not enough memory",
)
BUSY_NEEDLES = (
    "src.classify.next_architecture train",
    "src.segmentation.infer_masks",
)
FORBIDDEN_TEST_FLAG = "--allow-test-evaluation"


@dataclass(frozen=True)
class TrainJob:
    experiment: str
    resolution: int
    seed: int
    allow_oom_retry: bool = False


def prob_map_path(prob_dir: Path, image_path: str) -> Path:
    """Must match `unique_mask_name` stem in src.segmentation.infer_masks."""
    stem = Path(image_path).stem
    digest = hashlib.md5(image_path.encode("utf-8")).hexdigest()[:8]
    return Path(prob_dir) / f"{stem}_{digest}.npy"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ladder_dir(cfg: dict[str, Any]) -> Path:
    path = Path(cfg["paths"]["results_dir"]) / "ladder"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_dir(cfg: dict[str, Any]) -> Path:
    path = Path(cfg["paths"]["results_dir"]) / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def override_key(experiment: str, resolution: int, seed: int) -> str:
    return f"{experiment}:res{resolution}:seed{seed}"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=str) + "\n")


def find_run_dirs(results_root: Path, experiment: str, resolution: int, seed: int) -> list[Path]:
    parent = results_root / experiment / f"res{resolution}" / f"seed{seed}"
    if not parent.exists():
        return []
    return sorted(
        [p for p in parent.glob("foldofficial_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )


def run_status(results_root: Path, experiment: str, resolution: int, seed: int) -> tuple[Path | None, str]:
    """Return (dir, complete|resume|missing). Prefer a dir that already has results.json."""
    dirs = find_run_dirs(results_root, experiment, resolution, seed)
    complete = [p for p in dirs if (p / "results.json").exists()]
    if complete:
        return complete[-1], "complete"
    resumable = [p for p in dirs if (p / "train.pt").exists()]
    if resumable:
        return resumable[-1], "resume"
    if dirs:
        return dirs[-1], "missing"
    return None, "missing"


def read_val_auc(run_dir: Path) -> float | None:
    payload = load_json(run_dir / "results.json", {})
    auc = (payload.get("val_raw") or {}).get("auc")
    try:
        return float(auc)
    except (TypeError, ValueError):
        return None


def pick_e1_resolution(results_root: Path, seed: int = 42) -> dict[str, Any]:
    candidates = []
    for res in (384, 456):
        path, status = run_status(results_root, "E1", res, seed)
        auc = read_val_auc(path) if path and status == "complete" else None
        candidates.append({"resolution": res, "status": status, "path": str(path) if path else None, "val_auc": auc})
    finished = [c for c in candidates if c["val_auc"] is not None]
    if not finished:
        raise RuntimeError("E1 has no completed val AUC at 384 or 456. Cannot pick resolution.")
    finished.sort(key=lambda row: (-float(row["val_auc"]), row["resolution"]))
    winner = finished[0]
    return {
        "winner_resolution": int(winner["resolution"]),
        "winner_val_auc": winner["val_auc"],
        "seed": seed,
        "rule": "max E1 val_raw.auc; tie → lower resolution",
        "candidates": candidates,
        "development_only": True,
        "canonical_test_used": False,
    }


def e4_beats_e3(e3_auc: float, e4_auc: float) -> bool:
    return float(e4_auc) > float(e3_auc)


def missing_probability_maps(index_csv: Path, prob_dir: Path) -> list[str]:
    if not index_csv.exists():
        raise FileNotFoundError(index_csv)
    frame = pd.read_csv(index_csv)
    missing = []
    for image_path in frame["image_path"].astype(str):
        npy = prob_map_path(prob_dir, image_path)
        if not npy.exists():
            missing.append(image_path)
    return missing


def look_busy(needles: tuple[str, ...] = BUSY_NEEDLES) -> list[tuple[int, str]]:
    try:
        raw = subprocess.check_output(["ps", "-ax", "-o", "pid=,command="], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    found: list[tuple[int, str]] = []
    my_pid = os.getpid()
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, cmd = line.partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == my_pid:
            continue
        if "run_next_architecture_ladder" in cmd:
            continue
        if any(needle in cmd for needle in needles):
            found.append((pid, cmd.strip()))
    return found


def wait_until_idle(*, fail_if_busy: bool, poll_s: int = 30) -> None:
    busy = look_busy()
    if not busy:
        return
    if fail_if_busy:
        pretty = "; ".join(f"{pid} {cmd}" for pid, cmd in busy[:4])
        raise SystemExit(f"GPU job already running. Pass without --fail-if-busy to wait. {pretty}")
    print(f"[wait] {len(busy)} existing train/infer process(es). Waiting until they finish.")
    while True:
        time.sleep(poll_s)
        busy = look_busy()
        if not busy:
            print("[wait] GPU free. Continuing.")
            return
        print(f"[wait] still busy pid={busy[0][0]}")


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise SystemExit(f"ladder already running (lock {path})") from exc
    handle.write(f"{os.getpid()} {now_iso()}\n")
    handle.flush()
    return handle


def is_oom(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in OOM_MARKERS)


def python_cmd(explicit: str | None) -> str:
    if explicit:
        return explicit
    venv = ROOT / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


def _run_logged(cmd: list[str], log_path: Path, *, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = " ".join(cmd)
    print(f"[cmd] {pretty}")
    print(f"[log] {log_path}")
    if dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n# {now_iso()} {pretty}\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, env=env)
    return int(proc.returncode)


def _tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def train_command(
    python: str,
    config: Path,
    job: TrainJob,
    override: dict[str, int] | None,
) -> list[str]:
    cmd = [
        python,
        "-m",
        "src.classify.next_architecture",
        "train",
        "--experiment",
        job.experiment,
        "--resolution",
        str(job.resolution),
        "--seed",
        str(job.seed),
        "--config",
        str(config),
    ]
    if override:
        if override.get("batch_size"):
            cmd.extend(["--batch-size", str(override["batch_size"])])
        if override.get("grad_accumulation"):
            cmd.extend(["--grad-accumulation", str(override["grad_accumulation"])])
    return cmd


def oom_retry_plan() -> list[dict[str, int] | None]:
    return [
        None,
        {"batch_size": 4, "grad_accumulation": 4},
        {"batch_size": 2, "grad_accumulation": 8},
    ]


def run_train_job(
    *,
    cfg: dict[str, Any],
    python: str,
    config_path: Path,
    job: TrainJob,
    overrides: dict[str, Any],
    dry_run: bool,
    events_path: Path,
) -> str:
    results_root = Path(cfg["paths"]["results_dir"])
    path, status = run_status(results_root, job.experiment, job.resolution, job.seed)
    key = override_key(job.experiment, job.resolution, job.seed)
    if status == "complete":
        auc = read_val_auc(path) if path else None
        print(f"[skip] {job.experiment} res={job.resolution} seed={job.seed} complete val_auc={auc} dir={path}")
        append_event(
            events_path,
            {"ts": now_iso(), "event": "skip_complete", "job": job.__dict__, "dir": str(path), "val_auc": auc},
        )
        return "skipped"
    log_path = log_dir(cfg) / f"ladder_{job.experiment}_res{job.resolution}_seed{job.seed}.log"
    saved = overrides.get(key)
    if saved:
        plans: list[dict[str, int] | None] = [saved]
    elif job.allow_oom_retry:
        plans = oom_retry_plan()
    else:
        plans = [None]
    last_rc = 1
    for plan in plans:
        if plan:
            overrides[key] = plan
        cmd = train_command(python, config_path, job, plan)
        if FORBIDDEN_TEST_FLAG in cmd:
            raise RuntimeError("ladder refuses --allow-test-evaluation")
        append_event(
            events_path,
            {
                "ts": now_iso(),
                "event": "train_start",
                "job": job.__dict__,
                "override": plan,
                "resume": status == "resume",
            },
        )
        last_rc = _run_logged(cmd, log_path, dry_run=dry_run)
        if dry_run:
            return "dry_run"
        if last_rc == 0:
            append_event(events_path, {"ts": now_iso(), "event": "train_done", "job": job.__dict__, "override": plan})
            return "ran"
        log_text = _tail(log_path, 80)
        if job.allow_oom_retry and is_oom(log_text) and plan != plans[-1]:
            print(f"[oom] {job.experiment} res={job.resolution} seed={job.seed}. Retry smaller batch.")
            append_event(
                events_path, {"ts": now_iso(), "event": "oom_retry", "job": job.__dict__, "failed_override": plan}
            )
            continue
        snippet = log_text[-2000:] if log_text else f"exit {last_rc}"
        print(snippet)
        raise SystemExit(f"train failed rc={last_rc} job={job} log={log_path}")
    raise SystemExit(f"train failed rc={last_rc} job={job} log={log_path}")


def seeds_for_profile(cfg: dict[str, Any], profile: str, after_e1: bool) -> list[int]:
    all_seeds = [int(s) for s in cfg.get("seeds", [42, 43, 44, 45, 46])]
    if profile == "e0":
        return all_seeds
    if profile == "full" and after_e1:
        return all_seeds
    if profile == "full" and not after_e1:
        return all_seeds
    if after_e1:
        return [int(cfg.get("seed", 42))]
    return all_seeds


def should_stop(current: str, stop_after: str | None) -> bool:
    if not stop_after:
        return False
    return STAGES.index(current) >= STAGES.index(stop_after)


def write_status(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_iso()
    dump_json(path, payload)


def run_ladder(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    cfg = load_next_config(config_path)
    results_root = Path(cfg["paths"]["results_dir"])
    results_root.mkdir(parents=True, exist_ok=True)
    state_dir = ladder_dir(cfg)
    events_path = state_dir / "events.jsonl"
    status_path = state_dir / "status.json"
    overrides_path = state_dir / "run_overrides.json"
    lock_path = log_dir(cfg) / "ladder.lock"
    python = python_cmd(args.python)
    lock = acquire_lock(lock_path)
    try:
        return _run_ladder_inner(
            args=args,
            cfg=cfg,
            config_path=config_path,
            python=python,
            results_root=results_root,
            events_path=events_path,
            status_path=status_path,
            overrides_path=overrides_path,
        )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _run_followup(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    config_path: Path,
    python: str,
    results_root: Path,
    events_path: Path,
    status_path: Path,
    overrides_path: Path,
) -> int:
    """After reduced ladder: 5-seed E2 at winner res, then LOSO. Never force E5. Test locked."""
    overrides = load_json(overrides_path, {})
    pick_path = ladder_dir(cfg) / "winner_resolution.json"
    pick = load_json(pick_path, {})
    winner_res = int(pick.get("winner_resolution") or 384)
    prev = load_json(status_path, {})
    status: dict[str, Any] = dict(prev)
    status.update(
        {
            "profile": "followup",
            "winner_resolution": winner_res,
            "allow_test_evaluation": False,
            "e5": "still_skipped_e4_did_not_beat_e3",
        }
    )
    status.setdefault("stages", {})
    write_status(status_path, status)
    wait_until_idle(fail_if_busy=bool(args.fail_if_busy))
    audit_cmd = [python, "-m", "src.classify.next_architecture", "audit", "--config", str(config_path)]
    rc = _run_logged(audit_cmd, log_dir(cfg) / "ladder_audit.log", dry_run=args.dry_run)
    if rc != 0 and not args.dry_run:
        raise SystemExit(f"audit failed rc={rc}")

    seeds = [int(s) for s in cfg.get("seeds", [42, 43, 44, 45, 46])]
    for seed in seeds:
        run_train_job(
            cfg=cfg,
            python=python,
            config_path=config_path,
            job=TrainJob("E2", winner_res, seed, allow_oom_retry=True),
            overrides=overrides,
            dry_run=args.dry_run,
            events_path=events_path,
        )
        dump_json(overrides_path, overrides)
    status["stages"]["e2_seeds"] = "ok"
    write_status(status_path, status)
    if args.stop_after == "e2":
        return 0

    loso_cmd = [
        python,
        "-m",
        "src.classify.next_architecture",
        "loso",
        "--run",
        "--experiment",
        "E2",
        "--resolution",
        str(winner_res),
        "--seed",
        "42",
        "--config",
        str(config_path),
    ]
    rc = _run_logged(loso_cmd, log_dir(cfg) / "ladder_loso.log", dry_run=args.dry_run)
    if rc != 0 and not args.dry_run:
        raise SystemExit(f"loso failed rc={rc}")
    status["stages"]["loso"] = "ok"
    status["blocked"] = {
        "E5": "E4 did not beat E3 on val",
        "E6": "anatomy-aware biomarker provenance absent",
        "E7": "MIL blocked: no eye_id/visit_id",
        "E8": "source-balanced E2 follow-up; run --profile e8",
    }
    write_status(status_path, status)
    return 0


def _run_e8(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    config_path: Path,
    python: str,
    results_root: Path,
    events_path: Path,
    status_path: Path,
    overrides_path: Path,
) -> int:
    """E8 = E2 architecture + inverse-frequency source sampler, then LOSO. Test locked. No E5."""
    overrides = load_json(overrides_path, {})
    pick_path = ladder_dir(cfg) / "winner_resolution.json"
    pick = load_json(pick_path, {})
    winner_res = int(pick.get("winner_resolution") or 384)
    prev = load_json(status_path, {})
    status: dict[str, Any] = dict(prev)
    status.update(
        {
            "profile": "e8",
            "winner_resolution": winner_res,
            "allow_test_evaluation": False,
            "e5": "still_skipped_e4_did_not_beat_e3",
        }
    )
    status.setdefault("stages", {})
    write_status(status_path, status)
    wait_until_idle(fail_if_busy=bool(args.fail_if_busy))
    audit_cmd = [python, "-m", "src.classify.next_architecture", "audit", "--config", str(config_path)]
    rc = _run_logged(audit_cmd, log_dir(cfg) / "ladder_audit.log", dry_run=args.dry_run)
    if rc != 0 and not args.dry_run:
        raise SystemExit(f"audit failed rc={rc}")

    run_train_job(
        cfg=cfg,
        python=python,
        config_path=config_path,
        job=TrainJob("E8", winner_res, int(cfg.get("seed", 42)), allow_oom_retry=True),
        overrides=overrides,
        dry_run=args.dry_run,
        events_path=events_path,
    )
    dump_json(overrides_path, overrides)
    status["stages"]["e8"] = "ok"
    write_status(status_path, status)
    if args.stop_after == "e8":
        return 0

    loso_cmd = [
        python,
        "-m",
        "src.classify.next_architecture",
        "loso",
        "--run",
        "--experiment",
        "E8",
        "--resolution",
        str(winner_res),
        "--seed",
        str(int(cfg.get("seed", 42))),
        "--config",
        str(config_path),
    ]
    rc = _run_logged(loso_cmd, log_dir(cfg) / "ladder_loso_e8.log", dry_run=args.dry_run)
    if rc != 0 and not args.dry_run:
        raise SystemExit(f"loso E8 failed rc={rc}")
    status["stages"]["loso_e8"] = "ok"
    status["blocked"] = {
        "E5": "E4 did not beat E3 on val",
        "E6": "anatomy-aware biomarker provenance absent",
        "E7": "MIL blocked: no eye_id/visit_id",
        "nested_cv": "not required unless claiming better than B test 0.928",
    }
    write_status(status_path, status)
    return 0


def _run_ladder_inner(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    config_path: Path,
    python: str,
    results_root: Path,
    events_path: Path,
    status_path: Path,
    overrides_path: Path,
) -> int:
    if args.profile == "e8":
        return _run_e8(
            args=args,
            cfg=cfg,
            config_path=config_path,
            python=python,
            results_root=results_root,
            events_path=events_path,
            status_path=status_path,
            overrides_path=overrides_path,
        )
    if args.profile == "followup":
        return _run_followup(
            args=args,
            cfg=cfg,
            config_path=config_path,
            python=python,
            results_root=results_root,
            events_path=events_path,
            status_path=status_path,
            overrides_path=overrides_path,
        )
    overrides = load_json(overrides_path, {})
    status: dict[str, Any] = {
        "profile": args.profile,
        "stop_after": args.stop_after,
        "dry_run": bool(args.dry_run),
        "allow_test_evaluation": False,
        "canonical_split_sha256": cfg.get("expected_split_sha256"),
        "stages": {},
        "notes": [
            "Canonical test is locked.",
            "Do not compare development val AUC to Branch B test 0.928.",
            "E5 trains only if E4 val AUC > E3 val AUC.",
        ],
    }
    write_status(status_path, status)
    wait_until_idle(fail_if_busy=bool(args.fail_if_busy))

    # audit
    audit_cmd = [python, "-m", "src.classify.next_architecture", "audit", "--config", str(config_path)]
    audit_log = log_dir(cfg) / "ladder_audit.log"
    append_event(events_path, {"ts": now_iso(), "event": "stage", "stage": "audit"})
    rc = _run_logged(audit_cmd, audit_log, dry_run=args.dry_run)
    if rc != 0 and not args.dry_run:
        raise SystemExit(f"audit failed rc={rc} log={audit_log}")
    if not args.dry_run:
        verify_canonical_split(cfg, allow_noncanonical=False)
    status["stages"]["audit"] = "ok"
    write_status(status_path, status)
    if should_stop("audit", args.stop_after):
        return 0

    e0_seeds = seeds_for_profile(cfg, args.profile, after_e1=False)
    for seed in e0_seeds:
        run_train_job(
            cfg=cfg,
            python=python,
            config_path=config_path,
            job=TrainJob("E0", 224, seed, allow_oom_retry=False),
            overrides=overrides,
            dry_run=args.dry_run,
            events_path=events_path,
        )
        dump_json(overrides_path, overrides)
    status["stages"]["e0"] = "ok"
    write_status(status_path, status)
    if args.profile == "e0" or should_stop("e0", args.stop_after):
        return 0

    scout_seed = int(cfg.get("seed", 42))
    for res in (384, 456):
        run_train_job(
            cfg=cfg,
            python=python,
            config_path=config_path,
            job=TrainJob("E1", res, scout_seed, allow_oom_retry=True),
            overrides=overrides,
            dry_run=args.dry_run,
            events_path=events_path,
        )
        dump_json(overrides_path, overrides)
    status["stages"]["e1"] = "ok"
    write_status(status_path, status)
    if should_stop("e1", args.stop_after):
        return 0

    if args.dry_run:
        winner_res = 384
        pick = {"winner_resolution": winner_res, "dry_run": True}
    else:
        pick = pick_e1_resolution(results_root, seed=scout_seed)
        winner_res = int(pick["winner_resolution"])
    dump_json(ladder_dir(cfg) / "winner_resolution.json", pick)
    status["stages"]["pick_resolution"] = pick
    write_status(status_path, status)
    print(f"[pick] E1 winner resolution={winner_res} val_auc={pick.get('winner_val_auc')}")
    if should_stop("pick_resolution", args.stop_after):
        return 0

    later_seeds = seeds_for_profile(cfg, args.profile, after_e1=True)
    for experiment, stop_name in (("E2", "e2"), ("E3", "e3")):
        for seed in later_seeds:
            run_train_job(
                cfg=cfg,
                python=python,
                config_path=config_path,
                job=TrainJob(experiment, winner_res, seed, allow_oom_retry=True),
                overrides=overrides,
                dry_run=args.dry_run,
                events_path=events_path,
            )
            dump_json(overrides_path, overrides)
        status["stages"][stop_name] = "ok"
        write_status(status_path, status)
        if should_stop(stop_name, args.stop_after):
            return 0

    maps_index = cfg["paths"]["splits_dir"] / "all.csv"
    prob_dir = Path(cfg["paths"]["vessel_prob_dir"])
    missing = [] if args.dry_run else missing_probability_maps(maps_index, prob_dir)
    if missing:
        maps_cmd = [
            python,
            "-m",
            "src.segmentation.infer_masks",
            "--save-probability-maps",
            "--probability-dir",
            str(prob_dir),
            "--probability-manifest",
            str(cfg["paths"]["probability_manifest"]),
            "--index",
            str(maps_index),
        ]
        maps_log = log_dir(cfg) / "ladder_maps.log"
        append_event(events_path, {"ts": now_iso(), "event": "maps_start", "missing": len(missing)})
        rc = _run_logged(maps_cmd, maps_log, dry_run=args.dry_run)
        if rc != 0 and not args.dry_run:
            raise SystemExit(f"infer_masks failed rc={rc} log={maps_log}")
        if not args.dry_run:
            still = missing_probability_maps(maps_index, prob_dir)
            if still:
                raise SystemExit(f"soft maps still missing after infer: n={len(still)} example={still[0]}")
    else:
        print("[skip] soft probability maps complete")
    status["stages"]["maps"] = "ok"
    write_status(status_path, status)
    if should_stop("maps", args.stop_after):
        return 0

    compare_seed = later_seeds[0]
    for seed in later_seeds:
        run_train_job(
            cfg=cfg,
            python=python,
            config_path=config_path,
            job=TrainJob("E4", winner_res, seed, allow_oom_retry=True),
            overrides=overrides,
            dry_run=args.dry_run,
            events_path=events_path,
        )
        dump_json(overrides_path, overrides)
    status["stages"]["e4"] = "ok"
    write_status(status_path, status)
    if should_stop("e4", args.stop_after):
        return 0

    if args.dry_run:
        status["stages"]["compare"] = {"dry_run": True, "e4_beats_e3": False}
        write_status(status_path, status)
        print("[dry-run] skip E5 gate")
        return 0

    e3_dir, e3_st = run_status(results_root, "E3", winner_res, compare_seed)
    e4_dir, e4_st = run_status(results_root, "E4", winner_res, compare_seed)
    if e3_st != "complete" or e4_st != "complete" or e3_dir is None or e4_dir is None:
        raise SystemExit(f"E3/E4 compare needs complete val preds. e3={e3_st} e4={e4_st}")
    compare_out = ladder_dir(cfg) / f"compare_E4_vs_E3_res{winner_res}_seed{compare_seed}.json"
    compare_cmd = [
        python,
        "-m",
        "src.classify.next_architecture",
        "compare",
        "--reference",
        str(e3_dir / "val_predictions.csv"),
        "--candidate",
        str(e4_dir / "val_predictions.csv"),
        "--development-only",
        "--out",
        str(compare_out),
    ]
    rc = _run_logged(compare_cmd, log_dir(cfg) / "ladder_compare.log", dry_run=False)
    if rc != 0:
        raise SystemExit(f"compare failed rc={rc}")
    e3_auc = read_val_auc(e3_dir)
    e4_auc = read_val_auc(e4_dir)
    beats = e4_beats_e3(e3_auc or 0.0, e4_auc or 0.0)
    gate = {
        "e3_val_auc": e3_auc,
        "e4_val_auc": e4_auc,
        "e4_beats_e3": beats,
        "compare": str(compare_out),
        "development_only": True,
        "canonical_test_used": False,
    }
    dump_json(ladder_dir(cfg) / "e4_vs_e3_gate.json", gate)
    status["stages"]["compare"] = gate
    write_status(status_path, status)
    print(f"[gate] E4 val_auc={e4_auc} vs E3 val_auc={e3_auc} beats={beats}")
    if should_stop("compare", args.stop_after):
        return 0

    if not beats and not args.force_e5:
        print("[skip] E5 not trained: E4 did not beat E3 on grouped val. Audit vessels first.")
        status["stages"]["e5"] = "skipped_e4_did_not_beat_e3"
        status["blocked"] = {
            "E6": "anatomy-aware biomarker provenance absent",
            "E7": "MIL blocked: no eye_id/visit_id",
            "E8": "LOSO not auto-started (multi-day; use loso CLI)",
        }
        write_status(status_path, status)
        return 0

    for seed in later_seeds:
        run_train_job(
            cfg=cfg,
            python=python,
            config_path=config_path,
            job=TrainJob("E5", winner_res, seed, allow_oom_retry=True),
            overrides=overrides,
            dry_run=args.dry_run,
            events_path=events_path,
        )
        dump_json(overrides_path, overrides)
    status["stages"]["e5"] = "ok"
    status["blocked"] = {
        "E6": "anatomy-aware biomarker provenance absent",
        "E7": "MIL blocked: no eye_id/visit_id",
        "E8": "LOSO not auto-started (multi-day; use loso CLI)",
    }
    write_status(status_path, status)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run next-architecture E0–E5 sequentially with resume. Canonical test stays locked."
    )
    p.add_argument("--config", default=str(ROOT / "configs" / "next_architecture.yaml"))
    p.add_argument("--python", default=None, help="Python interpreter (default: .venv/bin/python)")
    p.add_argument("--profile", choices=("reduced", "full", "e0", "followup", "e8"), default="reduced")
    p.add_argument("--stop-after", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fail-if-busy", action="store_true", help="Exit if another train/infer is running")
    p.add_argument(
        "--force-e5",
        action="store_true",
        help="Train E5 even if E4 does not beat E3. Default off (protocol).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        f"[ladder] profile={args.profile} stop_after={args.stop_after} "
        f"dry_run={args.dry_run} test_eval=LOCKED"
    )
    return run_ladder(args)

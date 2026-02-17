"""
SEASL+ - Scientific Evaluation & Stability Layer
Non-destructive. Deterministic. Fully guarded.

This module evaluates system performance WITHOUT modifying
any existing engine, audit, or data files.

Phases:
  0. File Integrity Validation
  1. Rolling Metrics (full edge control)
  2. Monte Carlo Baseline (reproducible)
  3. Variance Detector (safe division)
  4. Entropy Analysis (bias-aware)
  5. Adaptive Controller (hard bounds)

Design Philosophy:
  - Does not chase variance
  - Does not assume predictability
  - Does not overreact
  - Does not break reproducibility
  - Does not modify core engine
"""

import random
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import Counter
from math import comb

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PREDICTIONS_FILE = BASE_DIR / "predictions.csv"
HISTORICAL_FILE = BASE_DIR / "main_draws.csv"
PB_FILE = BASE_DIR / "powerball_draws.csv"
PARAM_LOG_FILE = BASE_DIR / ".seasl_params.json"
AUDIT_HISTORY_FILE = BASE_DIR / ".audit_history.json"

MAIN_NUMBER_COUNT = 6
MAIN_RANGE = (1, 40)
PB_RANGE = (1, 10)

PREDICTIONS_REQUIRED_COLS = [
    "prediction_date", "line_id", "n1", "n2", "n3", "n4", "n5", "n6", "powerball"
]
HISTORICAL_REQUIRED_COLS = ["date", "n1", "n2", "n3", "n4", "n5", "n6"]


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — FILE INTEGRITY VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_input_files() -> Dict[str, Any]:
    """
    Validate all input files before evaluation begins.

    Checks:
      - predictions.csv exists, non-empty, correct schema, no NaN in numbers,
        no duplicate tickets in same run
      - main_draws.csv (historical) exists, non-empty, correct schema

    Returns dict with:
      status: "OK" | "FILE_MISSING" | "EMPTY_DATA" | "INVALID_SCHEMA"
      warnings: List[str]
      predictions_rows: int
      historical_rows: int
    """
    result = {
        "status": "OK",
        "warnings": [],
        "predictions_rows": 0,
        "historical_rows": 0,
        "predictions_df": None,
        "historical_df": None,
    }

    # --- Check predictions.csv ---
    if not PREDICTIONS_FILE.exists():
        result["status"] = "FILE_MISSING"
        result["warnings"].append(f"predictions.csv not found at {PREDICTIONS_FILE}")
        return result

    try:
        pred_df = pd.read_csv(PREDICTIONS_FILE)
    except Exception as e:
        result["status"] = "INVALID_SCHEMA"
        result["warnings"].append(f"predictions.csv read error: {e}")
        return result

    if len(pred_df) == 0:
        result["status"] = "EMPTY_DATA"
        result["warnings"].append("predictions.csv is empty")
        return result

    missing = [c for c in PREDICTIONS_REQUIRED_COLS if c not in pred_df.columns]
    if missing:
        result["status"] = "INVALID_SCHEMA"
        result["warnings"].append(f"predictions.csv missing columns: {missing}")
        return result

    # NaN check on number columns
    num_cols = ["n1", "n2", "n3", "n4", "n5", "n6", "powerball"]
    nan_count = pred_df[num_cols].isna().sum().sum()
    if nan_count > 0:
        result["status"] = "INVALID_SCHEMA"
        result["warnings"].append(f"predictions.csv has {nan_count} NaN values in number columns")
        return result

    # Duplicate ticket check within same run
    if "run_id" in pred_df.columns:
        ticket_cols = ["run_id", "n1", "n2", "n3", "n4", "n5", "n6", "powerball"]
        dupes = pred_df.duplicated(subset=ticket_cols).sum()
        if dupes > 0:
            result["warnings"].append(f"Found {dupes} duplicate tickets in same run")

    result["predictions_rows"] = len(pred_df)
    result["predictions_df"] = pred_df

    # --- Check historical draws ---
    if not HISTORICAL_FILE.exists():
        result["status"] = "FILE_MISSING"
        result["warnings"].append(f"main_draws.csv not found at {HISTORICAL_FILE}")
        return result

    try:
        hist_df = pd.read_csv(HISTORICAL_FILE)
    except Exception as e:
        result["status"] = "INVALID_SCHEMA"
        result["warnings"].append(f"main_draws.csv read error: {e}")
        return result

    if len(hist_df) == 0:
        result["status"] = "EMPTY_DATA"
        result["warnings"].append("main_draws.csv is empty")
        return result

    missing_h = [c for c in HISTORICAL_REQUIRED_COLS if c not in hist_df.columns]
    if missing_h:
        result["status"] = "INVALID_SCHEMA"
        result["warnings"].append(f"main_draws.csv missing columns: {missing_h}")
        return result

    result["historical_rows"] = len(hist_df)
    result["historical_df"] = hist_df

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — ROLLING METRICS (FULL EDGE CONTROL)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rolling_metrics(
    run_history: Optional[List[float]],
    window_size: int = 30
) -> Dict[str, Any]:
    """
    Compute rolling match-rate statistics with full edge-case handling.

    Args:
        run_history: List of match rates per run (e.g. [0.03, 0.05, ...])
        window_size: Rolling window size (default 30)

    Returns dict with:
        mean, std, ci_lower, ci_upper, n, status
    """
    # Edge: None input
    if run_history is None:
        return {
            "mean": None, "std": None,
            "ci_lower": None, "ci_upper": None,
            "n": 0, "status": "NO_HISTORY"
        }

    n = len(run_history)

    # Edge: empty list
    if n == 0:
        return {
            "mean": None, "std": None,
            "ci_lower": None, "ci_upper": None,
            "n": 0, "status": "ZERO_SAMPLE"
        }

    # Use only the last `window_size` entries
    window = run_history[-window_size:]
    n = len(window)
    mean_val = sum(window) / n

    # Edge: 1-9 samples
    if n < 10:
        return {
            "mean": mean_val, "std": None,
            "ci_lower": None, "ci_upper": None,
            "n": n, "status": "LOW_SAMPLE_WARNING"
        }

    # n >= 10: full computation with unbiased std (n-1)
    variance = sum((x - mean_val) ** 2 for x in window) / (n - 1)
    std_val = math.sqrt(variance)

    # Edge: zero variance
    if std_val == 0:
        return {
            "mean": mean_val, "std": 0.0,
            "ci_lower": mean_val, "ci_upper": mean_val,
            "n": n, "status": "NO_VARIANCE"
        }

    # Normal case: CI = mean +/- 1.96 * (std / sqrt(n))
    margin = 1.96 * (std_val / math.sqrt(n))
    return {
        "mean": mean_val, "std": std_val,
        "ci_lower": mean_val - margin,
        "ci_upper": mean_val + margin,
        "n": n, "status": "OK"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — MONTE CARLO BASELINE (REPRODUCIBLE)
# ═══════════════════════════════════════════════════════════════════════════════

def monte_carlo_batch_simulation(
    ticket_count: int,
    actual_rate: float = 0.0,
    simulations: int = 100000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Deterministic Monte Carlo simulation for random baseline.

    Simulates `ticket_count` random 6/40 tickets per trial for `simulations`
    trials, counts 3+ main-number matches, and compares to `actual_rate`.

    Args:
        ticket_count: Number of tickets per simulation batch
        actual_rate: The observed match rate to rank against baseline
        simulations: Number of simulation runs (must be >= 1000)
        seed: RNG seed for full reproducibility

    Returns dict with:
        baseline_mean, baseline_std, percentile_rank, p_value, seed_used, status
    """
    if ticket_count == 0:
        return {
            "baseline_mean": None, "baseline_std": None,
            "percentile_rank": None, "p_value": None,
            "seed_used": seed, "status": "NO_TICKETS"
        }

    if simulations < 1000:
        raise ValueError("Simulations too low for statistical stability (min 1000)")

    # Lock seeds for determinism
    random.seed(seed)
    np.random.seed(seed)

    # Hypergeometric probability approach (much faster than brute force)
    # P(k matches | 6 drawn from 40, 6 in winning set)
    total_pool = MAIN_RANGE[1]  # 40
    draw_size = MAIN_NUMBER_COUNT  # 6

    # Pre-compute P(k>=3) using hypergeometric
    total_combinations = comb(total_pool, draw_size)
    p_3plus = 0.0
    for k in range(3, draw_size + 1):
        p_k = (comb(draw_size, k) * comb(total_pool - draw_size, draw_size - k)) / total_combinations
        p_3plus += p_k

    # Simulate batch rates using binomial (each ticket has p_3plus chance)
    sim_rates = np.random.binomial(ticket_count, p_3plus, size=simulations) / ticket_count

    baseline_mean = float(np.mean(sim_rates))
    baseline_std = float(np.std(sim_rates, ddof=1)) if simulations > 1 else 0.0

    # Percentile rank of actual_rate within simulated distribution
    percentile_rank = float(np.mean(sim_rates <= actual_rate) * 100)

    # p-value: proportion of simulations >= actual_rate
    p_value = float(np.mean(sim_rates >= actual_rate))

    return {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "percentile_rank": percentile_rank,
        "p_value": p_value,
        "seed_used": seed,
        "status": "OK"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — VARIANCE DETECTOR (SAFE DIVISION)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_variance_phase(
    current: float,
    mean: Optional[float],
    std: Optional[float]
) -> Dict[str, Any]:
    """
    Detect if the current match rate is statistically anomalous.

    Args:
        current: Current run match rate
        mean: Rolling mean from Phase 1
        std: Rolling std from Phase 1

    Returns dict with:
        z_score, phase, status
    """
    # Guard: insufficient data
    if mean is None or std is None or std == 0:
        return {
            "z_score": 0.0,
            "phase": "INSUFFICIENT_VARIANCE",
            "status": "INSUFFICIENT_VARIANCE"
        }

    # Safe division guaranteed by guard above
    z = (current - mean) / std

    # Classify phase (use epsilon for floating-point boundary safety)
    EPS = 1e-9
    if z <= -2.0 + EPS:
        phase = "STRONG_NEGATIVE"
    elif z >= 2.0 - EPS:
        phase = "STRONG_POSITIVE"
    elif z <= -1.0 + EPS:
        phase = "MODERATE_NEGATIVE"
    elif z >= 1.0 - EPS:
        phase = "MODERATE_POSITIVE"
    else:
        phase = "NORMAL"

    return {
        "z_score": round(z, 4),
        "phase": phase,
        "status": "OK"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — ENTROPY (BIAS-AWARE)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_historical_distribution(
    draws_df: pd.DataFrame,
    date_col: str = "date",
    num_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Build empirical frequency distribution from historical draws.

    Args:
        draws_df: DataFrame of historical draws
        date_col: Name of the date column
        num_cols: Number columns to analyze (default n1-n6)

    Returns dict with:
        distribution (dict number->prob), draw_count, warnings, status
    """
    if num_cols is None:
        num_cols = ["n1", "n2", "n3", "n4", "n5", "n6"]

    warnings = []

    # Validate date continuity
    try:
        dates = pd.to_datetime(draws_df[date_col], format='mixed').sort_values()
        gaps = dates.diff().dt.days
        large_gaps = (gaps > 10).sum()  # Draws ~2x/week, so >10 days = gap
        if large_gaps > 0:
            warnings.append(f"DATA_GAP_WARNING: {large_gaps} gaps > 10 days in history")
    except Exception:
        warnings.append("DATE_PARSE_WARNING: Could not validate date continuity")

    # Sample size check
    draw_count = len(draws_df)
    if draw_count < 200:
        warnings.append("HISTORICAL_SAMPLE_WEAK: Less than 200 draws")

    # Build frequency distribution
    all_nums = draws_df[num_cols].values.flatten()
    counter = Counter(all_nums)
    total = len(all_nums)

    if total == 0:
        return {
            "distribution": {},
            "draw_count": 0,
            "warnings": warnings,
            "status": "EMPTY_DATA"
        }

    distribution = {int(k): v / total for k, v in counter.items()}

    return {
        "distribution": distribution,
        "draw_count": draw_count,
        "warnings": warnings,
        "status": "OK"
    }


def compute_entropy(
    distribution: Dict[int, float],
    label: str = ""
) -> Dict[str, Any]:
    """
    Compute Shannon entropy H = -SUM(p(x) * log2(p(x))).

    Standard handling: if p(x) == 0, exclude from sum.

    Args:
        distribution: Dict mapping number -> probability
        label: Label for this distribution (e.g. "predictions", "historical")

    Returns dict with:
        entropy, n_symbols, status
    """
    if not distribution:
        return {"entropy": 0.0, "n_symbols": 0, "status": "EMPTY_DISTRIBUTION"}

    H = 0.0
    n_symbols = 0
    for num, p in distribution.items():
        if p > 0:
            H -= p * math.log2(p)
            n_symbols += 1

    return {
        "entropy": round(H, 6),
        "n_symbols": n_symbols,
        "label": label,
        "status": "OK"
    }


def check_entropy_drift(
    pred_entropy: float,
    hist_entropy: float,
    hist_weak: bool = False
) -> Dict[str, Any]:
    """
    Compare prediction entropy vs historical entropy.

    Guardrail: |H_pred - H_hist| > 0.15 => ENTROPY_DRIFT

    Args:
        pred_entropy: Shannon entropy of predictions
        hist_entropy: Shannon entropy of historical draws
        hist_weak: Whether historical sample was flagged weak

    Returns dict with:
        drift, threshold, status
    """
    drift = abs(pred_entropy - hist_entropy)
    threshold = 0.15

    if hist_weak:
        return {
            "drift": round(drift, 6),
            "threshold": threshold,
            "pred_entropy": round(pred_entropy, 6),
            "hist_entropy": round(hist_entropy, 6),
            "status": "LOW_CONFIDENCE"
        }

    if drift > threshold:
        return {
            "drift": round(drift, 6),
            "threshold": threshold,
            "pred_entropy": round(pred_entropy, 6),
            "hist_entropy": round(hist_entropy, 6),
            "status": "ENTROPY_DRIFT"
        }

    return {
        "drift": round(drift, 6),
        "threshold": threshold,
        "pred_entropy": round(pred_entropy, 6),
        "hist_entropy": round(hist_entropy, 6),
        "status": "OK"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — ADAPTIVE CONTROLLER (HARD BOUNDS)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_param_log() -> Dict[str, Any]:
    """Load parameter log from disk."""
    if PARAM_LOG_FILE.exists():
        try:
            with open(PARAM_LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # Default state
    return {
        "baseline_defaults": {},
        "current_values": {},
        "change_history": [],
        "total_drift_pct": {},
        "runs_since_last_change": 0,
        "consecutive_low_percentile": 0,
        "runs_since_change_for_revert": 0,
    }


def _save_param_log(log: Dict[str, Any]):
    """Save parameter log to disk."""
    try:
        with open(PARAM_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2, default=str)
    except Exception:
        pass  # Never crash workflow


def adaptive_controller(
    rolling_metrics: Dict[str, Any],
    mc_result: Dict[str, Any],
    current_run_index: int = 0
) -> Dict[str, Any]:
    """
    Adaptive parameter controller with hard-bound constraints.

    Rules:
      - Rolling sample >= 30 before any change
      - Percentile rank < 40% for 3 consecutive windows to trigger change
      - Max lifetime drift per parameter = 5% total
      - Auto revert if percentile < 30% for 5 runs after change
      - No cascading adjustments
      - Only 1 parameter adjustment per 10 runs

    Args:
        rolling_metrics: Output from compute_rolling_metrics()
        mc_result: Output from monte_carlo_batch_simulation()
        current_run_index: Current run number

    Returns dict with:
        action, reason, status, param_log
    """
    log = _load_param_log()

    n = rolling_metrics.get("n", 0)
    percentile = mc_result.get("percentile_rank")

    # Guard: insufficient sample
    if n < 30:
        log["runs_since_last_change"] = log.get("runs_since_last_change", 0) + 1
        _save_param_log(log)
        return {
            "action": "HOLD",
            "reason": f"Insufficient sample (n={n}, need 30)",
            "status": "WAITING",
            "adjustments_made": 0
        }

    # Guard: no percentile available
    if percentile is None:
        log["runs_since_last_change"] = log.get("runs_since_last_change", 0) + 1
        _save_param_log(log)
        return {
            "action": "HOLD",
            "reason": "No percentile data available",
            "status": "WAITING",
            "adjustments_made": 0
        }

    # Track consecutive low-percentile windows
    if percentile < 40:
        log["consecutive_low_percentile"] = log.get("consecutive_low_percentile", 0) + 1
    else:
        log["consecutive_low_percentile"] = 0

    # Auto-revert check: if percentile < 30% for 5 runs after a change
    runs_since_change = log.get("runs_since_change_for_revert", 0)
    if runs_since_change > 0 and runs_since_change <= 5:
        if percentile < 30:
            log["runs_since_change_for_revert"] = runs_since_change + 1
            if log["runs_since_change_for_revert"] >= 5:
                # Auto-revert
                log["current_values"] = dict(log.get("baseline_defaults", {}))
                log["total_drift_pct"] = {}
                log["runs_since_change_for_revert"] = 0
                log["change_history"].append({
                    "run": current_run_index,
                    "action": "AUTO_REVERT",
                    "reason": "Percentile < 30% for 5 consecutive runs post-change",
                    "timestamp": datetime.now().isoformat()
                })
                _save_param_log(log)
                return {
                    "action": "AUTO_REVERT",
                    "reason": "Percentile < 30% for 5 runs after last change",
                    "status": "REVERTED",
                    "adjustments_made": 0
                }
        else:
            log["runs_since_change_for_revert"] = 0
    elif runs_since_change > 5:
        log["runs_since_change_for_revert"] = 0

    # Rate limit: only 1 change per 10 runs
    runs_since_last = log.get("runs_since_last_change", 0)
    if runs_since_last < 10:
        log["runs_since_last_change"] = runs_since_last + 1
        _save_param_log(log)
        return {
            "action": "HOLD",
            "reason": f"Rate limited ({runs_since_last}/10 runs since last change)",
            "status": "RATE_LIMITED",
            "adjustments_made": 0
        }

    # Check if 3 consecutive low windows
    if log.get("consecutive_low_percentile", 0) < 3:
        log["runs_since_last_change"] = runs_since_last + 1
        _save_param_log(log)
        return {
            "action": "HOLD",
            "reason": f"Need 3 consecutive low windows (have {log['consecutive_low_percentile']})",
            "status": "MONITORING",
            "adjustments_made": 0
        }

    # All conditions met: eligible for 1 adjustment
    # But check lifetime drift cap (5%)
    total_drift = sum(abs(v) for v in log.get("total_drift_pct", {}).values())
    if total_drift >= 5.0:
        log["runs_since_last_change"] = runs_since_last + 1
        _save_param_log(log)
        return {
            "action": "HOLD",
            "reason": f"Lifetime drift cap reached ({total_drift:.1f}% / 5.0%)",
            "status": "DRIFT_CAP",
            "adjustments_made": 0
        }

    # Record that a change is eligible (but we don't actually change engine params
    # since this is evaluation-only layer)
    log["runs_since_last_change"] = 0
    log["runs_since_change_for_revert"] = 1
    log["change_history"].append({
        "run": current_run_index,
        "action": "ADJUSTMENT_ELIGIBLE",
        "percentile": percentile,
        "consecutive_low": log["consecutive_low_percentile"],
        "timestamp": datetime.now().isoformat()
    })
    _save_param_log(log)

    return {
        "action": "ADJUSTMENT_ELIGIBLE",
        "reason": "3 consecutive windows below 40th percentile",
        "status": "ELIGIBLE",
        "adjustments_made": 0
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT HISTORY LOADER (sole source of rolling data)
# ═══════════════════════════════════════════════════════════════════════════════

def load_audit_history(json_path=None) -> Dict[str, Any]:
    """
    Load rolling performance data from .audit_history.json.
    This is the SOLE source of rolling history — no CSV fallback.

    Handles both current structure ({runs: [{run_id, timestamp, success}]})
    and future-proofed entries with three_plus_rate/four_plus_rate fields.

    Returns dict with:
        status: "OK" | "NO_HISTORY_FILE" | "CORRUPTED_HISTORY_FILE" | "EMPTY_HISTORY"
        three_plus_rates: List[float]
        four_plus_rates: List[float]
        ticket_counts: List[int]
        total_runs: int
    """
    if json_path is None:
        json_path = AUDIT_HISTORY_FILE

    if not Path(json_path).exists():
        return {
            "status": "NO_HISTORY_FILE",
            "three_plus_rates": [],
            "four_plus_rates": [],
            "ticket_counts": [],
            "total_runs": 0
        }

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return {
            "status": "CORRUPTED_HISTORY_FILE",
            "three_plus_rates": [],
            "four_plus_rates": [],
            "ticket_counts": [],
            "total_runs": 0
        }

    # Handle both formats: {"runs": [...]} and bare [...]
    if isinstance(data, dict) and "runs" in data:
        entries = data["runs"]
    elif isinstance(data, list):
        entries = data
    else:
        return {
            "status": "EMPTY_HISTORY",
            "three_plus_rates": [],
            "four_plus_rates": [],
            "ticket_counts": [],
            "total_runs": 0
        }

    if len(entries) == 0:
        return {
            "status": "EMPTY_HISTORY",
            "three_plus_rates": [],
            "four_plus_rates": [],
            "ticket_counts": [],
            "total_runs": 0
        }

    three_plus = []
    four_plus = []
    tickets = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        # Future-proofed: read rate fields if present
        if "three_plus_rate" in entry:
            three_plus.append(float(entry["three_plus_rate"]))
        elif "three_plus" in entry:
            three_plus.append(float(entry["three_plus"]))

        if "four_plus_rate" in entry:
            four_plus.append(float(entry["four_plus_rate"]))
        elif "four_plus" in entry:
            four_plus.append(float(entry["four_plus"]))

        if "ticket_count" in entry:
            tickets.append(int(entry["ticket_count"]))

    return {
        "status": "OK",
        "three_plus_rates": three_plus,
        "four_plus_rates": four_plus,
        "ticket_counts": tickets,
        "total_runs": len(entries)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — FULL EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def run_evaluation(seed: int = 42) -> Dict[str, Any]:
    """
    Run complete SEASL+ evaluation pipeline.

    Args:
        seed: RNG seed for Monte Carlo reproducibility

    Returns structured report dict.
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "seed_used": seed,
    }

    # ── Phase 0: File Integrity ──
    file_check = validate_input_files()
    report["file_status"] = file_check["status"]
    report["file_warnings"] = file_check["warnings"]

    if file_check["status"] != "OK":
        report["evaluation_skipped"] = True
        report["skip_reason"] = file_check["status"]
        return report

    pred_df = file_check["predictions_df"]
    hist_df = file_check["historical_df"]

    # ── Load rolling history from .audit_history.json (SOLE source) ──
    history_data = load_audit_history()
    report["history_source"] = ".audit_history.json"
    report["history_status"] = history_data["status"]
    report["history_total_runs"] = history_data["total_runs"]

    if history_data["status"] != "OK":
        rolling_sample = []
    else:
        rolling_sample = history_data["three_plus_rates"]

    report["history_entries"] = len(rolling_sample)

    # ── Phase 1: Rolling Metrics ──
    rolling = compute_rolling_metrics(rolling_sample if rolling_sample else None)
    report["rolling_sample_size"] = rolling["n"]
    report["rolling_mean"] = rolling["mean"]
    report["rolling_std"] = rolling["std"]
    report["rolling_ci"] = (
        [rolling["ci_lower"], rolling["ci_upper"]]
        if rolling["ci_lower"] is not None else None
    )
    report["rolling_status"] = rolling["status"]

    # ── Phase 2: Monte Carlo ──
    actual_rate = rolling["mean"] if rolling["mean"] is not None else 0.0
    ticket_count = len(pred_df) if len(pred_df) > 0 else 1
    mc = monte_carlo_batch_simulation(
        ticket_count=ticket_count,
        actual_rate=actual_rate,
        simulations=100000,
        seed=seed
    )
    report["mc_percentile"] = mc["percentile_rank"]
    report["mc_p_value"] = mc["p_value"]
    report["mc_baseline_mean"] = mc["baseline_mean"]
    report["mc_status"] = mc["status"]

    # ── Phase 3: Variance Detector ──
    current_rate = rolling_sample[-1] if rolling_sample else 0.0
    variance = detect_variance_phase(current_rate, rolling["mean"], rolling["std"])
    report["variance_z"] = variance["z_score"]
    report["variance_phase"] = variance["phase"]

    # ── Phase 4: Entropy ──
    hist_dist_result = compute_historical_distribution(hist_df)
    hist_ent = compute_entropy(hist_dist_result["distribution"], label="historical")

    # Build prediction distribution
    pred_nums = pred_df[["n1", "n2", "n3", "n4", "n5", "n6"]].values.flatten()
    pred_counter = Counter(pred_nums)
    pred_total = len(pred_nums)
    pred_distribution = {int(k): v / pred_total for k, v in pred_counter.items()} if pred_total > 0 else {}
    pred_ent = compute_entropy(pred_distribution, label="predictions")

    hist_weak = any("HISTORICAL_SAMPLE_WEAK" in w for w in hist_dist_result.get("warnings", []))
    entropy_check = check_entropy_drift(pred_ent["entropy"], hist_ent["entropy"], hist_weak)
    report["entropy_pred"] = pred_ent["entropy"]
    report["entropy_hist"] = hist_ent["entropy"]
    report["entropy_drift"] = entropy_check["drift"]
    report["entropy_status"] = entropy_check["status"]
    report["data_integrity_flags"] = hist_dist_result.get("warnings", [])

    # ── Phase 5: Adaptive Controller ──
    adaptive = adaptive_controller(rolling, mc)
    report["adaptive_action"] = adaptive["action"]
    report["adaptive_status"] = adaptive["status"]
    report["adaptive_reason"] = adaptive["reason"]

    return report


def print_report(report: Dict[str, Any]):
    """Print the structured SCIENTIFIC EVALUATION REPORT."""
    print("")
    print("SCIENTIFIC EVALUATION REPORT")
    print("----------------------------")
    print(f"  File Status:          {report.get('file_status', 'UNKNOWN')}")
    print(f"  History Source:       {report.get('history_source', 'N/A')}")
    print(f"  History Entries:      {report.get('history_entries', 0)}")

    if report.get("evaluation_skipped"):
        print(f"  Evaluation skipped safely: {report.get('skip_reason')}")
        if report.get("file_warnings"):
            print(f"  Warnings:             {report['file_warnings']}")
        return

    # Rolling
    mean_str = f"{report['rolling_mean']:.4%}" if report.get('rolling_mean') is not None else "N/A"
    print(f"  Rolling Sample Size:  {report.get('rolling_sample_size', 'N/A')}")
    print(f"  Rolling Mean (3+):    {mean_str}")

    ci = report.get("rolling_ci")
    if ci:
        print(f"  95% CI:               [{ci[0]:.4%}, {ci[1]:.4%}]")
    else:
        print(f"  95% CI:               N/A")

    # Monte Carlo
    pct = report.get("mc_percentile")
    pval = report.get("mc_p_value")
    print(f"  MC Percentile:        {pct:.1f}%" if pct is not None else "  MC Percentile:        N/A")
    print(f"  MC p-value:           {pval:.4f}" if pval is not None else "  MC p-value:           N/A")

    # Variance
    print(f"  Variance Phase:       {report.get('variance_phase', 'UNKNOWN')}")

    # Entropy
    print(f"  Entropy Status:       {report.get('entropy_status', 'UNKNOWN')}")
    print(f"    Pred H:             {report.get('entropy_pred', 'N/A')}")
    print(f"    Hist H:             {report.get('entropy_hist', 'N/A')}")
    print(f"    Drift:              {report.get('entropy_drift', 'N/A')}")

    # Adaptive
    print(f"  Adaptive Status:      {report.get('adaptive_status', 'OFF')}")
    print(f"    Action:             {report.get('adaptive_action', 'N/A')}")
    print(f"    Reason:             {report.get('adaptive_reason', 'N/A')}")

    # Integrity flags
    flags = report.get("data_integrity_flags", [])
    if flags:
        print(f"  Data Integrity Flags: {flags}")
    else:
        print(f"  Data Integrity Flags: []")

    print(f"  Seed Used:            {report.get('seed_used', 'N/A')}")
    print("")

    # Critical failure summary
    warnings = report.get("file_warnings", [])
    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    - {w}")
        print("")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = run_evaluation(seed=42)
    print_report(result)

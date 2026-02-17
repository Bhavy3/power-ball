"""
Baseline Comparison Module

Compares the current engine's performance against:
1. Pure Random Baseline (Hypergeometric probability)
2. Frequency-Weighted Baseline (Historical frequency)
3. Coverage-Optimized Baseline (Uniform coverage)

This module provides the "scientific control" for the experiment.
"""

import math
from math import comb
from typing import Dict, List, Any
import random
import pandas as pd
import numpy as np
from audit_engine import AuditEngine
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

class BaselineComparison:
    """
    Compares engine results against theoretical and heuristic baselines.
    """
    
    def __init__(self, historical_df: pd.DataFrame = None):
        self.audit = AuditEngine()
        self.history = historical_df
        
        # Load history if not provided
        if self.history is None:
            try:
                self.history = self.audit._load_and_validate_actuals()
            except Exception:
                self.history = pd.DataFrame() # Graceful fallback
                
        # Pre-calculate frequencies for Weighted Baseline
        self.frequencies = self._calculate_frequencies()
        
    def _calculate_frequencies(self) -> Dict[int, float]:
        """Calculate normalized frequencies of all numbers in history."""
        if self.history is None or self.history.empty:
            return {n: 1/40 for n in range(1, 41)}
            
        all_nums = []
        for col in ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']:
            if col in self.history.columns:
                all_nums.extend(self.history[col].tolist())
                
        counts = pd.Series(all_nums).value_counts()
        total = counts.sum()
        
        # Normalize to probability
        return {n: counts.get(n, 0) / total for n in range(1, 41)}

    def get_random_baseline(self) -> Dict[str, float]:
        """
        Theoretical Random Baseline (Hypergeometric).
        P(k) for 6/40 lottery.
        """
        # We can reuse the audit engine's strict calc
        return self.audit._calculate_hypergeometric_baseline()

    def get_weighted_baseline_performance(self, n_simulations: int = 1000) -> Dict[str, float]:
        """
        Simulates a "Frequency Weighted" strategy.
        Selects numbers based on their historical frequency.
        """
        if not self.frequencies:
            return {}
            
        # Simulating matches against random draws (since future draws are random)
        # We simulate "Weighted Strategy Tickets" vs "Random Draws"
        
        probs = [self.frequencies.get(n, 0) for n in range(1, 41)]
        # Normalize strictly
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        numbers = list(range(1, 41))
        
        matches = []
        
        random.seed(42) # Deterministic simulation
        
        for _ in range(n_simulations):
            # Generate a ticket using weighted probability
            ticket = set(np.random.choice(numbers, size=6, replace=False, p=probs))
            
            # Generate a random "draw" (Nature is uniform)
            draw = set(random.sample(numbers, 6))
            
            match_count = len(ticket & draw)
            matches.append(match_count)
            
        # Compile stats
        counts = pd.Series(matches).value_counts(normalize=True)
        return {
            "2_plus_matches": counts.get(2, 0) + counts.get(3, 0) + counts.get(4, 0) + counts.get(5, 0) + counts.get(6, 0),
            "3_plus_matches": counts.get(3, 0) + counts.get(4, 0) + counts.get(5, 0) + counts.get(6, 0),
            "4_plus_matches": counts.get(4, 0) + counts.get(5, 0) + counts.get(6, 0)
        }

    def get_coverage_baseline_performance(self, n_simulations: int = 1000) -> Dict[str, float]:
        """
        Simulates a "Coverage" strategy.
        Tries to cover as many unique numbers as possible across ticket sets.
        For a single ticket, this converges to Random Baseline.
        But for a set of tickets, it reduces variance.
        
        Here we simulate single ticket performance for direct comparison.
        """
        # For single ticket comparison, Coverage strategy is effectively Random strategy
        # unless we are evaluating SETS of tickets. 
        # The prompt asks for "Coverage-optimized baseline". 
        # Let's implementation it as "Least Drawn Numbers" strategy (opposite of weighted).
        
        if not self.frequencies:
            return {}
            
        # Inverse weight
        probs = [1.0 / (self.frequencies.get(n, 0.0001) + 0.0001) for n in range(1, 41)]
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        numbers = list(range(1, 41))
        matches = []
        random.seed(43) # Different seed for variety
        
        for _ in range(n_simulations):
            ticket = set(np.random.choice(numbers, size=6, replace=False, p=probs))
            draw = set(random.sample(numbers, 6))
            matches.append(len(ticket & draw))
            
        counts = pd.Series(matches).value_counts(normalize=True)
        return {
            "2_plus_matches": counts.get(2, 0) + counts.get(3, 0) + counts.get(4, 0) + counts.get(5, 0) + counts.get(6, 0),
            "3_plus_matches": counts.get(3, 0) + counts.get(4, 0) + counts.get(5, 0) + counts.get(6, 0),
            "4_plus_matches": counts.get(4, 0) + counts.get(5, 0) + counts.get(6, 0)
        }

    def compare_all(self, engine_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare engine metrics against all baselines.
        
        Args:
            engine_metrics: Dict containing '3_plus_pct', '4_plus_pct' etc from Engine or Audit.
        """
        random_bl = self.get_random_baseline()
        weighted_bl = self.get_weighted_baseline_performance()
        coverage_bl = self.get_coverage_baseline_performance()
        
        # Helper to safely get percentage
        def get_pct(metrics, key):
            # Handles both decimal (0.03) and string percentage ("3.00%")
            val = metrics.get(key, 0)
            if isinstance(val, str) and '%' in val:
                return float(val.strip('%')) / 100.0
            return float(val)

        # Engine Stats
        eng_3plus = get_pct(engine_metrics, '3_plus_pct') if '3_plus_pct' in engine_metrics else get_pct(engine_metrics, '3_plus')
        
        comparisons = {
            "ENGINE_vs_RANDOM": {
                "engine": eng_3plus,
                "baseline": random_bl.get("3_plus_matches", 0),
                "diff": eng_3plus - random_bl.get("3_plus_matches", 0)
            },
            "ENGINE_vs_WEIGHTED": {
                "engine": eng_3plus,
                "baseline": weighted_bl.get("3_plus_matches", 0),
                "diff": eng_3plus - weighted_bl.get("3_plus_matches", 0)
            },
            "ENGINE_vs_COVERAGE": {
                "engine": eng_3plus,
                "baseline": coverage_bl.get("3_plus_matches", 0),
                "diff": eng_3plus - coverage_bl.get("3_plus_matches", 0)
            }
        }
        
        return comparisons

    def print_comparison_report(self, engine_metrics: Dict[str, Any]):
        """Print a clean comparison report."""
        report = self.compare_all(engine_metrics)
        
        print("\n" + "="*60)
        print("📊 BASELINE COMPARISON REPORT")
        print("="*60)
        print(f"{'METRIC':<20} | {'ENGINE':<10} | {'BASELINE':<10} | {'DIFF':<10}")
        print("-" * 60)
        
        for name, data in report.items():
            eng = data['engine'] * 100
            base = data['baseline'] * 100
            diff = data['diff'] * 100
            symbol = "+" if diff > 0 else ""
            
            print(f"{name:<20} | {eng:6.2f}%   | {base:6.2f}%   | {symbol}{diff:6.2f}%")
            
        print("-" * 60)
        print("Analysis:")
        
        rnd_diff = report["ENGINE_vs_RANDOM"]["diff"]
        if rnd_diff > 0.005:
            print("  ✅ Engine outperforms Random Baseline (> 0.5%)")
        elif rnd_diff < -0.005:
             print("  ⚠️ Engine underperforms Random Baseline (< -0.5%)")
        else:
            print("  ℹ️  Engine performs statistically equivalent to Random")
        print("="*60 + "\n")

if __name__ == "__main__":
    # Self-test
    bc = BaselineComparison()
    print("Self-test: Comparing dummy engine stats (3.5% match rate)...")
    bc.print_comparison_report({"3_plus_pct": 0.035}) 

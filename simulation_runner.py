"""
Simulation Runner - Research-Grade Validation Harnish

Performs large-scale Monte Carlo simulations to validate the engine's 
long-term statistical behavior.
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import time
from tqdm import tqdm # Optional, but good for CLI
from engine import LotteryEngine
from ticket_composer import LotteryTicket
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

class SimulationRunner:
    def __init__(self):
        self.engine = LotteryEngine(use_adaptive=True)
        # We need a source of "Truth" to simulate against.
        # Since we are simulating *future* draws, we use a uniform random generator
        # as the "Ground Truth" (Nature).
        # We test if the engine's "biased" numbers perform better/worse/same as random
        # against a neutral universe.
        
    def _generate_random_draw(self) -> set:
        """Simulates a fair lottery draw (Nature)."""
        return set(random.sample(range(1, 41), 6))
        
    def run_simulation(self, n_draws: int = 1000, seed: int = 42) -> Dict[str, Any]:
        """
        Run a full simulation loop.
        
        Args:
            n_draws: Number of virtual draws to simulate
            seed: Random seed for reproducibility
        """
        print(f"\n🧪 STARTING SIMULATION: {n_draws} Draws | Seed: {seed}")
        print("   (Comparing Engine Recommendations vs Uniform Random Nature)")
        
        random.seed(seed)
        np.random.seed(seed)
        
        results = {
            "draws": [],
            "matches_3plus": 0,
            "matches_4plus": 0,
            "total_matches": 0,
            "match_dist": {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        }
        
        start_time = time.time()
        
        # We can't use tqdm easily without adding dependency, so simple print
        print_interval = max(1, n_draws // 10)
        
        for i in range(1, n_draws + 1):
            # 1. Engine generates a ticket (prediction)
            # Note: In a real simulation, we might feed this result back into the engine
            # if adaptive learning is on. For now, we test "static" performance 
            # or "current state" performance to avoid long re-training loops.
            ticket = self.engine.generate_ticket()
            pred_numbers = set(ticket.main_numbers)
            
            # 2. Nature generates a draw
            actual_numbers = self._generate_random_draw()
            
            # 3. Compare
            match_count = len(pred_numbers & actual_numbers)
            
            # 4. Record
            results["match_dist"][match_count] += 1
            if match_count >= 3:
                results["matches_3plus"] += 1
            if match_count >= 4:
                results["matches_4plus"] += 1
                
            if i % print_interval == 0:
                print(f"   Progress: {i}/{n_draws} ({i/n_draws:.0%})")
                
        duration = time.time() - start_time
        
        results["duration_sec"] = duration
        results["3_plus_rate"] = results["matches_3plus"] / n_draws
        results["4_plus_rate"] = results["matches_4plus"] / n_draws
        
        return results
        
    def print_simulation_report(self, results: Dict[str, Any]):
        """Print detailed simulation findings."""
        n = sum(results["match_dist"].values())
        rate_3plus = results["3_plus_rate"]
        
        # Theoretical Baseline (Hypergeometric 3+ for 6/40) is ~3.1% (0.0309)
        baseline_3plus = 0.0309
        
        diff = rate_3plus - baseline_3plus
        pct_change = (diff / baseline_3plus) * 100
        
        print("\n" + "="*60)
        print(f"🧪 SIMULATION RESULTS (n={n})")
        print("="*60)
        print(f"Duration: {results['duration_sec']:.2f}s")
        print("-" * 60)
        print("MATCH DISTRIBUTION:")
        for k in range(7):
            count = results["match_dist"].get(k, 0)
            pct = count / n * 100
            print(f"  {k} Matches: {count:5d} ({pct:6.2f}%)")
            
        print("-" * 60)
        print(f"3+ Match Rate: {rate_3plus:.2%} (Target: >{baseline_3plus:.2%})")
        print(f"Diff vs Random: {diff:+.2%} ({pct_change:+.1f}%)")
        
        if diff > 0.005:
            print("\n✅ RESULT: POSITIVE SKEW (Engine > Random)")
        elif diff < -0.005:
            print("\n⚠️ RESULT: NEGATIVE SKEW (Engine < Random)")
        else:
            print("\nℹ️ RESULT: NEUTRAL (Engine ≈ Random)")
            
        print("="*60 + "\n")

if __name__ == "__main__":
    sim = SimulationRunner()
    res = sim.run_simulation(n_draws=100)
    sim.print_simulation_report(res)

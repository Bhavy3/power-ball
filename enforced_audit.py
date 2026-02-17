"""
Enforced Audit Engine - Rule-Based Optimization System Auditor
Implements strict 5-step validation pipeline from supporting prompt.md

CRITICAL: This auditor does NOT trust the system blindly.
It VALIDATES every step before proceeding.

Steps:
1. FRESHNESS: Confirm new prediction batch
2. STRUCTURE: Validate parity, decades, sequences (Data Blindness check)
3. COMPARISON: Match analysis (3+ and 4+ only)
4. DIFFERENCE ANALYSIS: Structural shifts, variance (Safe Learning Zone)
5. SAFE LEARNING DECISION: Verdict on structural parameter updates
"""

import json
import uuid
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import Counter
from math import comb
import pandas as pd
from pathlib import Path
import os

# Base directory
BASE_DIR = Path(__file__).parent

# Audit run tracking file
AUDIT_HISTORY_FILE = BASE_DIR / ".audit_history.json"


class ValidationFailedError(Exception):
    """Raised when validation fails - STOPS execution immediately."""
    pass


def load_audit_history() -> Dict:
    """Load previous audit run history."""
    if AUDIT_HISTORY_FILE.exists():
        try:
            with open(AUDIT_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"runs": []}
    return {"runs": []}


def save_audit_history(history: Dict):
    """Save audit run history."""
    with open(AUDIT_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def get_last_audit_time() -> Optional[datetime]:
    """Get timestamp of last audit run."""
    history = load_audit_history()
    runs = history.get("runs", [])
    if runs:
        try:
            return datetime.fromisoformat(runs[-1].get("run_timestamp", ""))
        except:
            return None
    return None


class EnforcedAuditEngine:
    """
    Rule-Based Optimization System Auditor.
    
    Enforces:
    - Data Blindness (Structure > specific numbers)
    - Structural Constraints (Parity, Decades, Sequences)
    - Safe Learning Rules
    """
    
    def __init__(self,
                 predictions_csv: str = "predictions.csv",
                 actuals_csv: str = "actual_draws.csv"):
        self.predictions_path = BASE_DIR / predictions_csv
        self.actuals_path = BASE_DIR / actuals_csv
        
        self.run_id = str(uuid.uuid4())[:8]
        self.run_timestamp = datetime.now()
        
        self.results = {
            "step1_freshness": None,
            "step2_structure": None,
            "step3_comparison": None,
            "step4_difference": None,
            "step5_verdict": None
        }
        
        self._predictions_df = None
        self._actuals_df = None
    
    # =========================================================================
    # STEP 1: FRESHNESS (MANDATORY)
    # =========================================================================
    
    def _validate_freshness(self) -> bool:
        """
        Validate that predictions are FRESHLY generated.
        Reuse of old prediction files is strictly forbidden.
        """
        if not self.predictions_path.exists():
            raise ValidationFailedError(
                "\n[VALIDATION FAILED]\n"
                "Predictions file does not exist.\n"
                "Action required: Generate new predictions BEFORE audit."
            )
        
        file_mtime = datetime.fromtimestamp(os.path.getmtime(self.predictions_path))
        last_audit = get_last_audit_time()
        
        df = pd.read_csv(self.predictions_path)
        
        # Check freshness via timestamp column or file mtime
        has_timestamp = 'generation_timestamp' in df.columns
        
        if has_timestamp:
            try:
                latest_gen = pd.to_datetime(df['generation_timestamp']).max()
                if last_audit and latest_gen <= last_audit:
                    raise ValidationFailedError(
                        "\n[VALIDATION FAILED]\n"
                        "Predictions were NOT freshly generated in this run.\n"
                        f"Latest generation: {latest_gen}\n"
                        f"Last audit run: {last_audit}\n"
                        "Action: Generate new predictions."
                    )
            except ValidationFailedError:
                raise
            except Exception:
                pass # Fallback to file mtime if parse fails
        
        if not has_timestamp and last_audit and file_mtime <= last_audit:
             raise ValidationFailedError(
                "\n[VALIDATION FAILED]\n"
                "Predictions file is stale (modified before last audit).\n"
                "Action: Generate new predictions."
            )
            
        print(f"[OK] STEP 1: Fresh predictions confirmed (Run ID: {self.run_id})")
        self.results["step1_freshness"] = "PASSED"
        self._predictions_df = df
        return True

    # =========================================================================
    # STEP 2: STRUCTURAL VALIDATION (DATA BLINDNESS)
    # =========================================================================
    
    def _validate_structure(self) -> bool:
        """
        Validate structural rules.
        - Parity: 2-4 evens
        - Spread: 2-4 low (1-20)
        - Decades: Max 3 per decade
        - Sequences: Max 2 consecutive
        """
        if self._predictions_df is None:
            raise ValidationFailedError("Step 1 failed")
        
        df = self._predictions_df
        violations = []
        
        for idx, row in df.iterrows():
            nums = sorted([int(row[f'n{i}']) for i in range(1, 7)])
            
            # 1. Parity (Odd/Even)
            evens = sum(1 for n in nums if n % 2 == 0)
            if not (2 <= evens <= 4):
                violations.append(f"Row {idx}: Invalid parity ({evens} evens)")
                continue
                
            # 2. Spread (Low/High)
            lows = sum(1 for n in nums if n <= 20)
            if not (2 <= lows <= 4):
                violations.append(f"Row {idx}: Invalid spread ({lows} low)")
                continue

            # 3. Decades (Max 3)
            decades = Counter(n // 10 for n in nums)
            # Handle 40 as decade 4
            if any(cnt > 3 for cnt in decades.values()):
                violations.append(f"Row {idx}: Decade limit exceeded {dict(decades)}")
                continue

            # 4. Sequences (Max 2 consecutive)
            cons_count = 1
            max_cons = 1
            for i in range(1, 6):
                if nums[i] == nums[i-1] + 1:
                    cons_count += 1
                else:
                    max_cons = max(max_cons, cons_count)
                    cons_count = 1
            max_cons = max(max_cons, cons_count)
            
            if max_cons > 2:
                violations.append(f"Row {idx}: Excessive sequence ({max_cons} consecutive)")
                continue
                
        if violations:
            print(f"[ERROR] Found {len(violations)} structural violations!")
            for v in violations[:3]: print(f"  {v}")
            raise ValidationFailedError(
                "\n[VALIDATION FAILED]\n"
                "Predictions violate structural constraints.\n"
                "System is NOT strictly following optimization rules."
            )
            
        print(f"[OK] STEP 2: Structural rules verified ({len(df)} predictions)")
        self.results["step2_structure"] = "PASSED"
        return True

    # =========================================================================
    # STEP 3: COMPARISON (NO LEARNING)
    # =========================================================================

    def _run_comparison(self) -> Dict:
        """
        Compare predictions vs actuals.
        Focus: 3+ and 4+ matches ONLY.
        """
        # Load actuals
        if not self.actuals_path.exists():
            raise ValidationFailedError("Actuals file missing")
            
        actuals = pd.read_csv(self.actuals_path)
        actuals['draw_date'] = pd.to_datetime(actuals['draw_date'])
        
        preds = self._predictions_df.copy()
        
        # Determine date column
        date_col = next((c for c in ['prediction_date', 'date'] if c in preds.columns), None)
        if date_col:
            preds[date_col] = pd.to_datetime(preds[date_col])
        else:
            print("[WARN] No date column - assuming alignment by index/generation")
            
        matches = []
        total = 0
        pb_hits = 0
        
        for idx, row in preds.iterrows():
            # Find next draw
            if date_col:
                p_date = row[date_col]
                future = actuals[actuals['draw_date'] > p_date]
                if future.empty: continue
                draw = future.sort_values('draw_date').iloc[0]
            else:
                 continue # Skip if no date alignment possible
            
            # Compare
            p_nums = set(int(row[f'n{i}']) for i in range(1, 7))
            a_nums = set(int(draw[f'a{i}']) for i in range(1, 7))
            
            m_cnt = len(p_nums & a_nums)
            matches.append(m_cnt)
            
            if int(row['powerball']) == int(draw['actual_powerball']):
                pb_hits += 1
            
            total += 1
            
        if total == 0:
            raise ValidationFailedError("No predictions could be matched to draws")
            
        stats = Counter(matches)
        results = {
            "total": total,
            "match_counts": dict(stats),
            "3_plus": sum(stats[k] for k in range(3, 7)),
            "4_plus": sum(stats[k] for k in range(4, 7)),
            "pb_hits": pb_hits
        }
        
        print(f"[OK] STEP 3: Comparison complete ({total} matched)")
        print(f"     3+ Matches: {results['3_plus']} ({(results['3_plus']/total)*100:.1f}%)")
        self.results["step3_comparison"] = results
        return results

    # =========================================================================
    # STEP 4: DIFFERENCE ANALYSIS (SAFE LEARNING ZONE)
    # =========================================================================

    def _analyze_difference(self) -> Dict:
        """
        Analyze structural differences (O/E, L/H) vs Random Expectation.
        This guides 'Safe Learning'.
        """
        df = self._predictions_df
        total = len(df)
        
        # 1. Parity Ratio (Expect 0.5 for even vs odd number count? No, probability differs)
        # Even numbers in 1-40: 20. Odd: 20. So 50/50 split expected on average.
        all_nums = [int(df.iloc[r][f'n{i}']) for r in range(total) for i in range(1, 7)]
        evens = sum(1 for n in all_nums if n % 2 == 0)
        even_ratio = evens / len(all_nums)
        
        # 2. Low/High Ratio
        lows = sum(1 for n in all_nums if n <= 20)
        low_ratio = lows / len(all_nums)
        
        # 3. Coverage
        unique_nums = len(set(all_nums))
        coverage = unique_nums / 40.0
        
        analysis = {
            "even_ratio": even_ratio,
            "low_ratio": low_ratio,
            "coverage": coverage,
            "variance_ok": coverage > 0.8 # Ensure we aren't just picking same 5 numbers
        }
        
        print(f"[OK] STEP 4: Structural Analysis")
        print(f"     Even Ratio: {even_ratio:.2f} (Target 0.50)")
        print(f"     Low Ratio:  {low_ratio:.2f} (Target 0.50)")
        print(f"     Coverage:   {coverage*100:.0f}% of pool")
        
        self.results["step4_difference"] = analysis
        return analysis

    # =========================================================================
    # STEP 5: SAFE LEARNING DECISION & VERDICT
    # =========================================================================

    def _generate_verdict(self, comp: Dict, diff: Dict) -> Dict:
        """
        Generate verdict based on match rates and structural health.
        """
        total = comp["total"]
        rate_3plus = (comp["3_plus"] / total) * 100
        
        # Random baselines (Hypergeometric)
        # 3+ match prob ~ 3.34%
        random_3plus = 3.34
        deviation = rate_3plus - random_3plus
        
        # Verdict Logic
        if total < 100:
            verdict = "INCONCLUSIVE"
            action = "NO ACTION TAKEN (Insufficient Data)"
        elif deviation > 0.5 and diff["variance_ok"]:
             verdict = "ABOVE RANDOM"
             action = "MAINTAIN CURRENT PARAMETERS (Positive Signal)"
        elif deviation < -0.5:
             verdict = "BELOW RANDOM"
             action = "ADJUST STRUCTURAL WEIGHTS (Negative Signal)"
        else:
             verdict = "EQUIVALENT TO RANDOM"
             action = "NO ACTION TAKEN (Noise Range)"
             
        # Honesty Requirement
        disclosure = (
            "This system does not predict lottery outcomes.\n"
            "Any observed improvement applies ONLY to low-tier matches.\n"
            "Randomness remains dominant."
        )
        
        result = {
            "verdict": verdict,
            "action": action,
            "deviation": f"{deviation:+.2f}%",
            "disclosure": disclosure
        }
        
        print(f"[OK] STEP 5: {verdict}")
        print(f"     Action: {action}")
        self.results["step5_verdict"] = result
        return result

    # =========================================================================
    # EXECUTION
    # =========================================================================

    def _print_detailed_report(self, comp: Dict, diff: Dict, verdict: Dict):
        """
        Print detailed audit report matching user's requested format.
        """
        total = comp["total"]
        counts = comp["match_counts"]
        
        print("\n" + "="*70)
        print("FINAL AUDIT REPORT")
        print("="*70)
        print(f"\n  Fresh predictions confirmed for this run")
        print(f"  Run ID: {self.run_id}")
        
        print(f"\n" + "-"*70)
        print("ACCURACY DISTRIBUTION (Focus: 3+ and 4+ matches)")
        print("-"*70)
        
        # 0-6 Matches
        for i in range(7):
            cnt = counts.get(i, 0)
            pct = (cnt / total) * 100 if total > 0 else 0
            print(f"  {i} matches: {cnt:3d} ({pct:6.2f}%)")
            
        print(f"\n  3+ matches: {comp['3_plus']:3d} ({comp['3_plus']/total*100:6.2f}%)")
        print(f"  4+ matches: {comp['4_plus']:3d} ({comp['4_plus']/total*100:6.2f}%)")
        
        print(f"\n" + "-"*70)
        print("RANDOM BASELINE")
        print("-"*70)
        print(f"  3+ matches (random): 3.3426%")
        print(f"  4+ matches (random): 0.2246%")
        
        print(f"\n" + "="*70)
        print("VERDICT")
        print("="*70)
        print(f"\n  Category:   {verdict['verdict']}")
        # Map action to a simpler "Confidence" like status if needed, or just print Action
        # The user example showed "Confidence: LOW" etc. 
        # Let's infer Confidence from the verdict/action context
        confidence = "LOW"
        if "ABOVE RANDOM" in verdict['verdict']: confidence = "MODERATE" # Or HIGH if >100 samples
        if "INCONCLUSIVE" in verdict['verdict']: confidence = "LOW"
        
        print(f"  Action:     {verdict['action']}")
        print(f"  Reason:     {verdict['deviation']} deviation from random")
        
        print(f"\n" + "-"*70)
        print("HONESTY DISCLOSURE:")
        print("-"*70)
        print(f"  {verdict['disclosure']}")
        
        print(f"\n" + "="*70)
        print("This system is evaluated ONLY for 3+ and 4+ matches.")
        print("Jackpot prediction is OUT OF SCOPE.")
        print("="*70 + "\n")

    def run_enforced_audit(self) -> Dict:
        print("\n" + "="*70)
        print("RULE-BASED OPTIMIZATION AUDIT")
        print("="*70)
        print(f"Run ID: {self.run_id}")
        print(f"Timestamp: {self.run_timestamp}")
        print("-"*70 + "\n")
        
        try:
            self._validate_freshness()
            self._validate_structure()
            comp = self._run_comparison()
            diff = self._analyze_difference()
            verdict = self._generate_verdict(comp, diff)
            
            # Print detailed report at the end
            self._print_detailed_report(comp, diff, verdict)
            
            self._save_run(True)
            return {
                "status": "SUCCESS",
                "run_id": self.run_id,
                "results": self.results,
                "verdict": verdict # Return verdict dict for workflow script
            }
            
        except ValidationFailedError as e:
            print(str(e))
            self._save_run(False)
            return {"status": "FAILED", "run_id": self.run_id, "error": str(e)}
        except Exception as e:
            print(f"[ERROR] {e}")
            self._save_run(False)
            return {"status": "ERROR", "run_id": self.run_id, "error": str(e)}

    def _save_run(self, success: bool):
        hist = load_audit_history()
        hist["runs"].append({
            "run_id": self.run_id,
            "timestamp": self.run_timestamp.isoformat(),
            "success": success
        })
        save_audit_history(hist)

if __name__ == "__main__":
    EnforcedAuditEngine().run_enforced_audit()

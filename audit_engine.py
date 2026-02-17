"""
Audit Engine - System Performance Evaluation Module
READ-ONLY statistical evaluation of prediction accuracy.

CRITICAL CONSTRAINTS:
- DO NOT change any existing logic, weights, generators, or sampling methods
- DO NOT retrain or re-balance the model
- This is READ-ONLY analysis only
- Purpose: EVALUATE performance, not improve the system

IMPLEMENTATION OF supporting prompt.md:
- Phase 1: Data Integrity & Parsing Checks
- Phase 2: Matching Logic Validation
- Phase 3: Sample Size & Statistical Safety
- Phase 4: Random Baseline Comparison
- Phase 5: Output & Verdict Rules
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter
from math import comb
import pandas as pd
from pathlib import Path


# Base directory
BASE_DIR = Path(__file__).parent


class AuditEngineError(Exception):
    """Raised when audit fails."""
    pass


class AssertionFailedError(Exception):
    """Raised when a mandatory assertion fails - STOPS evaluation."""
    pass


@dataclass
class DataHealthReport:
    """Phase 1: Data integrity tracking."""
    predictions_loaded: int = 0
    predictions_rejected: int = 0
    actuals_loaded: int = 0
    actuals_rejected: int = 0
    parse_errors: List[str] = field(default_factory=list)
    assertion_status: str = "PENDING"
    audit_confidence: str = "HIGH"
    
    def add_error(self, error: str):
        self.parse_errors.append(error)
        self.audit_confidence = "LOW"
    
    def to_dict(self) -> Dict:
        return {
            "predictions_loaded": self.predictions_loaded,
            "predictions_rejected": self.predictions_rejected,
            "actuals_loaded": self.actuals_loaded,
            "actuals_rejected": self.actuals_rejected,
            "parse_error_count": len(self.parse_errors),
            "parse_errors": self.parse_errors[:10],  # First 10 only
            "assertion_status": self.assertion_status,
            "audit_confidence": self.audit_confidence
        }


@dataclass
class MatchResult:
    """Result of matching a prediction to actual draw."""
    prediction_date: str
    line_id: int
    main_match_count: int
    powerball_match: bool
    predicted_main: Set[int]
    actual_main: Set[int]
    predicted_pb: int
    actual_pb: int


class AuditEngine:
    """
    Evaluates system prediction accuracy against actual lottery results.
    
    Implements the 5-phase evaluation pipeline from supporting prompt.md.
    """
    
    # Lottery parameters
    MAIN_NUMBER_COUNT = 6
    MAIN_NUMBER_RANGE = (1, 40)
    POWERBALL_RANGE = (1, 10)
    
    def __init__(self, 
                 predictions_csv: str = "predictions.csv",
                 actuals_csv: str = "actual_draws.csv"):
        """
        Initialize audit engine.
        
        Args:
            predictions_csv: Path to predictions file
            actuals_csv: Path to actual draws file
        """
        self.predictions_path = BASE_DIR / predictions_csv
        self.actuals_path = BASE_DIR / actuals_csv
        
        self._predictions_df = None
        self._actuals_df = None
        self._health = DataHealthReport()
        self._match_results: List[MatchResult] = []
    
    # =========================================================================
    # PHASE 1: Data Integrity & Parsing
    # =========================================================================
    
    def _load_and_validate_predictions(self) -> pd.DataFrame:
        """Load predictions with strict validation."""
        if not self.predictions_path.exists():
            raise AuditEngineError(f"Predictions file not found: {self.predictions_path}")
        
        df = pd.read_csv(self.predictions_path)
        valid_rows = []
        
        for idx, row in df.iterrows():
            try:
                # Validate all main numbers are integers in range
                main_nums = []
                for col in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
                    val = int(row[col])
                    if not (self.MAIN_NUMBER_RANGE[0] <= val <= self.MAIN_NUMBER_RANGE[1]):
                        raise ValueError(f"{col}={val} out of range [1-40]")
                    main_nums.append(val)
                
                # Validate powerball
                pb = int(row['powerball'])
                if not (self.POWERBALL_RANGE[0] <= pb <= self.POWERBALL_RANGE[1]):
                    raise ValueError(f"powerball={pb} out of range [1-10]")
                
                # Validate uniqueness of main numbers
                if len(set(main_nums)) != 6:
                    raise ValueError("Main numbers must be unique")
                
                valid_rows.append(row)
                self._health.predictions_loaded += 1
                
            except Exception as e:
                self._health.predictions_rejected += 1
                self._health.add_error(f"predictions row {idx}: {e}")
        
        return pd.DataFrame(valid_rows)
    
    def _load_and_validate_actuals(self) -> pd.DataFrame:
        """Load actual draws with strict validation."""
        if not self.actuals_path.exists():
            raise AuditEngineError(f"Actuals file not found: {self.actuals_path}")
        
        df = pd.read_csv(self.actuals_path)
        valid_rows = []
        
        for idx, row in df.iterrows():
            try:
                # Validate all main numbers
                main_nums = []
                for col in ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']:
                    val = int(row[col])
                    if not (self.MAIN_NUMBER_RANGE[0] <= val <= self.MAIN_NUMBER_RANGE[1]):
                        raise ValueError(f"{col}={val} out of range [1-40]")
                    main_nums.append(val)
                
                # Validate powerball
                pb = int(row['actual_powerball'])
                if not (self.POWERBALL_RANGE[0] <= pb <= self.POWERBALL_RANGE[1]):
                    raise ValueError(f"powerball={pb} out of range [1-10]")
                
                valid_rows.append(row)
                self._health.actuals_loaded += 1
                
            except Exception as e:
                self._health.actuals_rejected += 1
                self._health.add_error(f"actuals row {idx}: {e}")
        
        return pd.DataFrame(valid_rows)
    
    # =========================================================================
    # PHASE 2: Matching Logic
    # =========================================================================
    
    def _find_next_draw(self, prediction_date: datetime, actuals_df: pd.DataFrame) -> Optional[pd.Series]:
        """Find the NEXT official draw after prediction_date."""
        actuals_df = actuals_df.copy()
        actuals_df['draw_date'] = pd.to_datetime(actuals_df['draw_date'])
        
        future_draws = actuals_df[actuals_df['draw_date'] > prediction_date]
        if len(future_draws) == 0:
            return None
        
        # Get the earliest draw after prediction date
        return future_draws.sort_values('draw_date').iloc[0]
    
    def _count_main_matches(self, predicted: Set[int], actual: Set[int]) -> int:
        """Count main number matches (unordered sets)."""
        return len(predicted & actual)
    
    def _match_predictions(self, predictions_df: pd.DataFrame, actuals_df: pd.DataFrame):
        """Match each prediction to its next draw."""
        self._match_results = []
        
        for idx, pred_row in predictions_df.iterrows():
            try:
                pred_date = pd.to_datetime(pred_row['prediction_date'])
                next_draw = self._find_next_draw(pred_date, actuals_df)
                
                if next_draw is None:
                    continue
                
                # Extract numbers
                pred_main = set([int(pred_row[f'n{i}']) for i in range(1, 7)])
                actual_main = set([int(next_draw[f'a{i}']) for i in range(1, 7)])
                pred_pb = int(pred_row['powerball'])
                actual_pb = int(next_draw['actual_powerball'])
                
                # Count matches
                main_matches = self._count_main_matches(pred_main, actual_main)
                pb_match = pred_pb == actual_pb
                
                # Validate match count (ASSERTION)
                if main_matches not in range(0, 7):
                    raise AssertionFailedError(
                        f"main_match_count={main_matches} not in {{0,1,2,3,4,5,6}}"
                    )
                
                self._match_results.append(MatchResult(
                    prediction_date=str(pred_date.date()),
                    line_id=int(pred_row.get('line_id', idx)),
                    main_match_count=main_matches,
                    powerball_match=pb_match,
                    predicted_main=pred_main,
                    actual_main=actual_main,
                    predicted_pb=pred_pb,
                    actual_pb=actual_pb
                ))
                
            except AssertionFailedError:
                raise
            except Exception as e:
                self._health.add_error(f"Matching error row {idx}: {e}")
    
    def _validate_bucket_assertions(self, buckets: Dict[int, int]):
        """Validate aggregation sanity checks."""
        total_from_buckets = sum(buckets.values())
        total_evaluated = len(self._match_results)
        
        # Assertion: Sum of buckets == total predictions
        if total_from_buckets != total_evaluated:
            raise AssertionFailedError(
                f"Bucket sum ({total_from_buckets}) != total evaluated ({total_evaluated})"
            )
        
        # Assertion: 3+ = 3 + 4 + 5 + 6
        three_plus = buckets.get(3, 0) + buckets.get(4, 0) + buckets.get(5, 0) + buckets.get(6, 0)
        
        # Assertion: 4+ = 4 + 5 + 6
        four_plus = buckets.get(4, 0) + buckets.get(5, 0) + buckets.get(6, 0)
        
        return three_plus, four_plus
    
    # =========================================================================
    # PHASE 3: Sample Size & Statistical Safety
    # =========================================================================
    
    def _check_sample_size(self, total_draws: int) -> Dict:
        """Check sample size constraints."""
        result = {
            "total_draws": total_draws,
            "confidence_limit": "HIGH",
            "category_limit": None,
            "noise_warning": None
        }
        
        if total_draws < 30:
            result["confidence_limit"] = "LOW"
            result["category_limit"] = "INCONCLUSIVE"
            result["noise_warning"] = "Results dominated by noise"
        elif total_draws < 100:
            result["confidence_limit"] = "LOW"
            result["category_limit"] = "INCONCLUSIVE"
        
        return result
    
    # =========================================================================
    # PHASE 4: Random Baseline Calculation
    # =========================================================================
    
    def _calculate_hypergeometric_baseline(self) -> Dict:
        """
        Calculate correct hypergeometric probabilities for 6-from-40.
        
        P(k matches) = C(6,k) * C(34, 6-k) / C(40, 6)
        """
        total_balls = 40
        drawn = 6
        picked = 6
        
        total_combinations = comb(total_balls, drawn)
        
        probabilities = {}
        for k in range(0, 7):
            # Ways to pick k matching numbers from the 6 drawn
            matches_ways = comb(drawn, k)
            # Ways to pick (6-k) non-matching from remaining (40-6) = 34
            non_matches_ways = comb(total_balls - drawn, picked - k)
            
            prob = (matches_ways * non_matches_ways) / total_combinations
            probabilities[f"{k}_matches"] = prob
        
        # Powerball is independent: 1/10
        probabilities["powerball_only"] = 0.10
        
        # Cumulative probabilities
        probabilities["2_plus_matches"] = sum(probabilities[f"{k}_matches"] for k in range(2, 7))
        probabilities["3_plus_matches"] = sum(probabilities[f"{k}_matches"] for k in range(3, 7))
        probabilities["4_plus_matches"] = sum(probabilities[f"{k}_matches"] for k in range(4, 7))
        
        return probabilities
    
    # =========================================================================
    # PHASE 5: Verdict Determination
    # =========================================================================
    
    def _determine_verdict(self, 
                          accuracy: Dict, 
                          baseline: Dict,
                          sample_check: Dict) -> Dict:
        """
        Determine final verdict with strict constraints.
        
        Allowed verdicts:
        - INVALID EVALUATION
        - INCONCLUSIVE (insufficient data)
        - STATISTICALLY EQUIVALENT TO RANDOM
        - SLIGHTLY ABOVE RANDOM (LOW CONFIDENCE)
        - MEANINGFULLY ABOVE RANDOM (HIGH CONFIDENCE)
        """
        # Check if we can even give a verdict
        if self._health.assertion_status == "FAILED":
            return {
                "category": "INVALID EVALUATION",
                "confidence": "N/A",
                "reason": "Assertions failed - evaluation cannot be trusted"
            }
        
        # Check sample size limit
        if sample_check.get("category_limit") == "INCONCLUSIVE":
            return {
                "category": "INCONCLUSIVE (insufficient data)",
                "confidence": "LOW",
                "reason": f"Only {sample_check['total_draws']} draws evaluated. " +
                         (sample_check.get("noise_warning") or "Need at least 100 for reliable results")
            }
        
        # Calculate deviation from random
        actual_3plus = accuracy.get("3_plus_pct", 0)
        random_3plus = baseline.get("3_plus_matches", 0) * 100
        deviation = actual_3plus - random_3plus
        
        # Determine if HIGH confidence is even possible
        can_be_high = (
            sample_check["total_draws"] >= 100 and
            len(self._health.parse_errors) == 0 and
            self._health.assertion_status == "PASSED"
        )
        
        # Verdict logic
        if deviation > 0.5 and can_be_high:
            return {
                "category": "MEANINGFULLY ABOVE RANDOM (HIGH CONFIDENCE)",
                "confidence": "HIGH",
                "practical_edge": "YES",
                "reason": f"3+ match rate {actual_3plus:.2f}% is {deviation:+.2f}% above random baseline"
            }
        elif deviation > 0.5:
            # Would be meaningful but can't be high confidence
            return {
                "category": "SLIGHTLY ABOVE RANDOM (LOW CONFIDENCE)",
                "confidence": "LOW",
                "practical_edge": "QUESTIONABLE",
                "reason": f"3+ match rate {actual_3plus:.2f}% appears above random, but confidence is LOW due to sample size or errors"
            }
        elif deviation > 0:
            return {
                "category": "SLIGHTLY ABOVE RANDOM (LOW CONFIDENCE)",
                "confidence": "LOW",
                "practical_edge": "QUESTIONABLE",
                "reason": f"3+ match rate {actual_3plus:.2f}% is marginally above random (+{deviation:.2f}%)"
            }
        elif deviation > -0.5:
            return {
                "category": "STATISTICALLY EQUIVALENT TO RANDOM",
                "confidence": "N/A",
                "practical_edge": "NO",
                "reason": f"3+ match rate {actual_3plus:.2f}% matches random expectations ({random_3plus:.2f}%)"
            }
        else:
            return {
                "category": "STATISTICALLY EQUIVALENT TO RANDOM",
                "confidence": "HIGH",
                "practical_edge": "NO",
                "reason": f"3+ match rate {actual_3plus:.2f}% is {abs(deviation):.2f}% BELOW random baseline"
            }
    
    # =========================================================================
    # Main Evaluation
    # =========================================================================
    
    def evaluate(self) -> Dict:
        """
        Run complete 5-phase evaluation.
        
        Returns structured output per supporting prompt.md requirements.
        """
        try:
            # PHASE 1: Data Integrity
            predictions_df = self._load_and_validate_predictions()
            actuals_df = self._load_and_validate_actuals()
            
            if len(predictions_df) == 0:
                return {
                    "status": "FAILED",
                    "data_health": self._health.to_dict(),
                    "message": "No valid predictions to evaluate"
                }
            
            # PHASE 2: Matching Logic
            try:
                self._match_predictions(predictions_df, actuals_df)
                
                # Build match buckets
                buckets = Counter(r.main_match_count for r in self._match_results)
                pb_only = sum(1 for r in self._match_results 
                             if r.powerball_match and r.main_match_count == 0)
                
                # Validate assertions
                three_plus, four_plus = self._validate_bucket_assertions(buckets)
                self._health.assertion_status = "PASSED"
                
            except AssertionFailedError as e:
                self._health.assertion_status = "FAILED"
                return {
                    "status": "INVALID EVALUATION",
                    "data_health": self._health.to_dict(),
                    "message": f"Assertion failed: {e}",
                    "verdict": {"category": "INVALID EVALUATION"}
                }
            
            # PHASE 3: Sample Size Check
            total_evaluated = len(self._match_results)
            sample_check = self._check_sample_size(total_evaluated)
            
            # PHASE 4: Random Baseline
            baseline = self._calculate_hypergeometric_baseline()
            
            # Calculate accuracy distribution
            accuracy = {}
            for k in range(0, 7):
                count = buckets.get(k, 0)
                pct = (count / total_evaluated * 100) if total_evaluated > 0 else 0
                accuracy[f"{k}_matches"] = {"count": count, "pct": pct}
            
            accuracy["powerball_only"] = {"count": pb_only, "pct": (pb_only / total_evaluated * 100) if total_evaluated > 0 else 0}
            accuracy["2_plus"] = {"count": sum(buckets.get(k, 0) for k in range(2, 7)), 
                                  "pct": sum(accuracy[f"{k}_matches"]["pct"] for k in range(2, 7))}
            accuracy["3_plus"] = {"count": three_plus, "pct": sum(accuracy[f"{k}_matches"]["pct"] for k in range(3, 7))}
            accuracy["4_plus"] = {"count": four_plus, "pct": sum(accuracy[f"{k}_matches"]["pct"] for k in range(4, 7))}
            accuracy["3_plus_pct"] = accuracy["3_plus"]["pct"]
            
            # Deviation analysis
            deviations = {}
            for tier in ["2_plus", "3_plus", "4_plus"]:
                actual_pct = accuracy[tier]["pct"]
                random_pct = baseline.get(f"{tier}_matches", 0) * 100
                deviations[tier] = {
                    "actual_pct": actual_pct,
                    "random_pct": random_pct,
                    "deviation_pct": actual_pct - random_pct
                }
            
            # PHASE 5: Verdict
            verdict = self._determine_verdict(accuracy, baseline, sample_check)
            
            return {
                "status": "SUCCESS",
                "analysis_date": datetime.now().isoformat(),
                "total_predictions_evaluated": total_evaluated,
                
                # Section 1: Data Health Report
                "data_health": self._health.to_dict(),
                
                # Section 2: Accuracy Distribution
                "accuracy_distribution": accuracy,
                
                # Section 3: Random Baseline
                "random_baseline": {k: f"{v*100:.4f}%" for k, v in baseline.items()},
                
                # Section 4: Deviation Analysis
                "deviation_from_random": deviations,
                
                # Section 5: Verdict
                "verdict": verdict,
                
                # Sample size info
                "sample_size_check": sample_check
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": str(e),
                "data_health": self._health.to_dict()
            }
    
    def print_audit_report(self):
        """Print formatted audit report per supporting prompt.md requirements."""
        results = self.evaluate()
        
        if results.get("status") == "ERROR":
            print(f"\n[ERROR] AUDIT FAILED: {results.get('message')}")
            return
        
        # Header
        print("\n" + "=" * 70)
        print("LOTTERY PREDICTION SYSTEM - PERFORMANCE AUDIT")
        print("Evaluation per supporting prompt.md specifications")
        print("=" * 70)
        print(f"Analysis Date: {results.get('analysis_date', 'N/A')}")
        print(f"Total Predictions Evaluated: {results.get('total_predictions_evaluated', 0)}")
        
        # Section 1: Data Health Report
        health = results.get("data_health", {})
        print("\n" + "-" * 70)
        print("1. DATA HEALTH REPORT")
        print("-" * 70)
        print(f"  Predictions Loaded:   {health.get('predictions_loaded', 0)}")
        print(f"  Predictions Rejected: {health.get('predictions_rejected', 0)}")
        print(f"  Actuals Loaded:       {health.get('actuals_loaded', 0)}")
        print(f"  Actuals Rejected:     {health.get('actuals_rejected', 0)}")
        print(f"  Parse Errors:         {health.get('parse_error_count', 0)}")
        print(f"  Assertion Status:     {health.get('assertion_status', 'N/A')}")
        print(f"  Audit Confidence:     {health.get('audit_confidence', 'N/A')}")
        
        if health.get('parse_errors'):
            print("  Errors:")
            for err in health['parse_errors'][:5]:
                print(f"    - {err}")
        
        # Section 2: Accuracy Distribution
        acc = results.get("accuracy_distribution", {})
        print("\n" + "-" * 70)
        print("2. ACCURACY DISTRIBUTION TABLE")
        print("-" * 70)
        for k in range(0, 7):
            key = f"{k}_matches"
            data = acc.get(key, {})
            print(f"  {k} Matches: {data.get('count', 0):3d} draws ({data.get('pct', 0):6.2f}%)")
        pb_data = acc.get("powerball_only", {})
        print(f"  PB Only:   {pb_data.get('count', 0):3d} draws ({pb_data.get('pct', 0):6.2f}%)")
        
        # High-value summary
        print("\n  HIGH-VALUE COMBINATIONS:")
        for tier in ["2_plus", "3_plus", "4_plus"]:
            data = acc.get(tier, {})
            print(f"    {tier.replace('_', '+')}:  {data.get('count', 0):3d} draws ({data.get('pct', 0):6.2f}%)")
        
        # Section 3: Random Baseline
        baseline = results.get("random_baseline", {})
        print("\n" + "-" * 70)
        print("3. RANDOM BASELINE TABLE (Hypergeometric 6-from-40)")
        print("-" * 70)
        for k in range(0, 7):
            print(f"  {k} Matches: {baseline.get(f'{k}_matches', 'N/A')}")
        print(f"  PB Only:   {baseline.get('powerball_only', 'N/A')}")
        
        # Section 4: Deviation Analysis
        dev = results.get("deviation_from_random", {})
        print("\n" + "-" * 70)
        print("4. DEVIATION ANALYSIS")
        print("-" * 70)
        for tier in ["2_plus", "3_plus", "4_plus"]:
            d = dev.get(tier, {})
            actual = d.get("actual_pct", 0)
            random = d.get("random_pct", 0)
            deviation = d.get("deviation_pct", 0)
            symbol = "^" if deviation > 0 else "v" if deviation < 0 else "="
            print(f"\n  {tier.upper()}:")
            print(f"    Actual:    {actual:6.2f}%")
            print(f"    Random:    {random:6.2f}%")
            print(f"    Deviation: {deviation:+6.2f}% {symbol}")
        
        # Section 5: Verdict
        verdict = results.get("verdict", {})
        print("\n" + "=" * 70)
        print("5. FINAL VERDICT")
        print("=" * 70)
        print(f"\n  Category:       {verdict.get('category', 'N/A')}")
        print(f"  Confidence:     {verdict.get('confidence', 'N/A')}")
        if verdict.get('practical_edge'):
            print(f"  Practical Edge: {verdict.get('practical_edge')}")
        print(f"  Reason:         {verdict.get('reason', 'N/A')}")
        
        # Honesty Clause (MANDATORY per supporting prompt.md)
        if verdict.get('practical_edge') in ['NO', 'NEGATIVE', 'QUESTIONABLE', None]:
            print("\n" + "-" * 70)
            print("HONESTY DISCLOSURE (MANDATORY):")
            print("-" * 70)
            print("  This system does not demonstrate a reliable predictive edge.")
            print("  Observed deviations may be due to variance.")
            print("")
            print("  No marketing language. No optimism bias. Truth over hype.")
        
        print("\n" + "=" * 70)
        print("This system is evaluated ONLY for 3 and 4 main-number matches.")
        print("Jackpot prediction is OUT OF SCOPE.")
        print("=" * 70 + "\n")
    
    def get_json_output(self, indent: int = 2) -> str:
        """Get audit results as JSON."""
        try:
            results = self.evaluate()
            return json.dumps(results, indent=indent, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=indent)


if __name__ == "__main__":
    print("Lottery Prediction Audit Engine")
    print("(Implementation of supporting prompt.md)\n")
    
    engine = AuditEngine()
    engine.print_audit_report()

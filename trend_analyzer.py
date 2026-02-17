"""
Hot & Cold Number Trend Analysis Module
READ-ONLY, NON-INFLUENTIAL statistical trend reporting.

CRITICAL: This module is COMPLETELY ISOLATED from number generation.
It does NOT affect:
- Ticket generation logic
- Probability weights
- Confidence scoring
- Adaptive feedback mechanisms

This is for INFORMATIONAL and TRANSPARENCY purposes only.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import Counter
import pandas as pd
from data_loader import load_main_draws, load_powerball_draws


# Mandatory disclaimer (as per prompt specifications)
DISCLAIMER = (
    "Hot and Cold numbers describe past frequency only. "
    "Lottery draws are random events, and no pattern improves the odds of winning."
)


class TrendAnalyzer:
    """
    Analyzes hot and cold number trends from historical data.
    
    DESIGN PHILOSOPHY:
    - Enhances user trust
    - Provides system transparency
    - Promotes statistical literacy
    
    WITHOUT:
    - Gambler's fallacy
    - False prediction claims
    - Hidden bias
    
    This module is READ-ONLY and ISOLATED.
    """
    
    def __init__(self, lookback_days: int = 90):
        """
        Initialize trend analyzer.
        
        Args:
            lookback_days: Rolling window for analysis (default: 90 days)
        """
        self.lookback_days = lookback_days
        self._main_draws = None
        self._powerball_draws = None
    
    def _load_data(self):
        """Load draw data if not already loaded."""
        if self._main_draws is None:
            self._main_draws = load_main_draws()
        if self._powerball_draws is None:
            self._powerball_draws = load_powerball_draws()
    
    def _filter_by_date(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Filter DataFrame to last N days.
        
        Args:
            df: DataFrame with date column
            date_col: Name of date column
            
        Returns:
            Filtered DataFrame
        """
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        
        # Ensure date is datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], format='mixed')
        
        return df[df[date_col] >= cutoff_date]
    
    def _calculate_frequencies(self, numbers: List[int]) -> Counter:
        """
        Calculate frequency of each number.
        
        Args:
            numbers: List of numbers to count
            
        Returns:
            Counter with frequencies
        """
        return Counter(numbers)
    
    def _classify_hot_cold(self, 
                          frequencies: Counter, 
                          possible_numbers: range,
                          top_n: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """
        Classify numbers as hot or cold.
        
        Hot: Above average frequency (top 5)
        Cold: Below average frequency (bottom 5)
        
        Args:
            frequencies: Counter of number frequencies
            possible_numbers: Range of possible numbers
            top_n: Number of hot/cold to return (default: 5)
            
        Returns:
            Tuple of (hot_list, cold_list)
        """
        # Ensure all numbers have an entry (even zero)
        for num in possible_numbers:
            if num not in frequencies:
                frequencies[num] = 0
        
        # Calculate average
        total = sum(frequencies.values())
        avg = total / len(possible_numbers) if len(possible_numbers) > 0 else 0
        
        # Classify
        hot = []
        cold = []
        
        for num, freq in frequencies.items():
            entry = {"number": int(num), "frequency": int(freq)}
            if freq > avg:
                hot.append(entry)
            elif freq < avg:
                cold.append(entry)
        
        # Sort and select top N
        hot = sorted(hot, key=lambda x: x["frequency"], reverse=True)[:top_n]
        cold = sorted(cold, key=lambda x: x["frequency"])[:top_n]
        
        return hot, cold
    
    def analyze_main_numbers(self) -> Dict:
        """
        Analyze hot/cold trends for main numbers (1-40).
        
        Returns:
            Dict with 'hot' and 'cold' lists
        """
        self._load_data()
        
        # Filter to recent draws
        recent = self._filter_by_date(self._main_draws)
        
        if len(recent) == 0:
            return {"hot": [], "cold": [], "draws_analyzed": 0}
        
        # Flatten all 6 main numbers into single list
        number_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
        all_numbers = []
        for col in number_cols:
            all_numbers.extend(recent[col].tolist())
        
        # Calculate frequencies
        frequencies = self._calculate_frequencies(all_numbers)
        
        # Classify
        hot, cold = self._classify_hot_cold(frequencies, range(1, 41))
        
        return {
            "hot": hot,
            "cold": cold,
            "draws_analyzed": len(recent)
        }
    
    def analyze_powerball(self) -> Dict:
        """
        Analyze hot/cold trends for Powerball (1-10).
        
        Returns:
            Dict with 'hot' and 'cold' lists
        """
        self._load_data()
        
        # Filter to recent draws
        recent = self._filter_by_date(self._powerball_draws)
        
        if len(recent) == 0:
            return {"hot": [], "cold": [], "draws_analyzed": 0}
        
        # Get all powerball numbers
        all_numbers = recent['powerball'].tolist()
        
        # Calculate frequencies
        frequencies = self._calculate_frequencies(all_numbers)
        
        # Classify
        hot, cold = self._classify_hot_cold(frequencies, range(1, 11))
        
        return {
            "hot": hot,
            "cold": cold,
            "draws_analyzed": len(recent)
        }
    
    def get_full_analysis(self) -> Dict:
        """
        Get complete hot/cold analysis.
        
        Returns:
            Structured analysis result (as per prompt specification)
        """
        main_analysis = self.analyze_main_numbers()
        pb_analysis = self.analyze_powerball()
        
        return {
            "analysis_period": f"Last {self.lookback_days} days",
            "main_numbers": {
                "hot": main_analysis["hot"],
                "cold": main_analysis["cold"],
                "draws_analyzed": main_analysis["draws_analyzed"]
            },
            "powerball": {
                "hot": pb_analysis["hot"],
                "cold": pb_analysis["cold"],
                "draws_analyzed": pb_analysis["draws_analyzed"]
            },
            "disclaimer": DISCLAIMER
        }
    
    def get_json_output(self, indent: int = 2) -> str:
        """
        Get analysis as JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON formatted string
        """
        return json.dumps(self.get_full_analysis(), indent=indent)
    
    def print_report(self):
        """Print formatted trend analysis report."""
        analysis = self.get_full_analysis()
        
        print("\n" + "=" * 60)
        print("HOT & COLD NUMBER TREND ANALYSIS")
        print(f"Period: {analysis['analysis_period']}")
        print("=" * 60)
        
        # Main numbers
        print(f"\nMAIN NUMBERS (1-40) - {analysis['main_numbers']['draws_analyzed']} draws analyzed")
        print("-" * 40)
        
        print("\n  HOT (above average frequency):")
        for entry in analysis['main_numbers']['hot']:
            print(f"    Number {entry['number']:2d}: appeared {entry['frequency']} times")
        
        print("\n  COLD (below average frequency):")
        for entry in analysis['main_numbers']['cold']:
            print(f"    Number {entry['number']:2d}: appeared {entry['frequency']} times")
        
        # Powerball
        print(f"\nPOWERBALL (1-10) - {analysis['powerball']['draws_analyzed']} draws analyzed")
        print("-" * 40)
        
        print("\n  HOT (above average frequency):")
        for entry in analysis['powerball']['hot']:
            print(f"    Powerball {entry['number']:2d}: appeared {entry['frequency']} times")
        
        print("\n  COLD (below average frequency):")
        for entry in analysis['powerball']['cold']:
            print(f"    Powerball {entry['number']:2d}: appeared {entry['frequency']} times")
        
        # Mandatory disclaimer
        print("\n" + "=" * 60)
        print("DISCLAIMER:")
        print(DISCLAIMER)
        print("=" * 60)


if __name__ == "__main__":
    print("Testing Hot & Cold Trend Analyzer...")
    print("(This is a READ-ONLY, NON-INFLUENTIAL module)")
    
    analyzer = TrendAnalyzer(lookback_days=90)
    
    # Print formatted report
    analyzer.print_report()
    
    # Also show JSON output
    print("\n\nJSON OUTPUT:")
    print(analyzer.get_json_output())

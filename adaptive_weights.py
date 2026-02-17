"""
Adaptive Weights Module
Combines historical frequency data with user feedback signals.
Implements the 85/15 weighting formula: Historical dominant, feedback subordinate.
"""

from typing import Dict, Optional
from data_loader import load_main_frequencies, load_powerball_frequencies, get_frequency_weights
from feedback_store import FeedbackStore


# Weight constants (as per system prompt)
HISTORICAL_WEIGHT = 0.85
FEEDBACK_WEIGHT = 0.15


class AdaptiveWeightCalculator:
    """
    Calculates adaptive weights combining historical data and user feedback.
    
    Formula:
        Final Weight = 0.85 * Historical Distribution + 0.15 * User Feedback Signal
        
    Historical data ALWAYS dominates.
    """
    
    def __init__(self, feedback_store: FeedbackStore = None):
        """
        Initialize calculator.
        
        Args:
            feedback_store: FeedbackStore instance (creates new if None)
        """
        self.feedback_store = feedback_store or FeedbackStore()
        
        # Load historical frequencies
        main_freq = load_main_frequencies()
        pb_freq = load_powerball_frequencies()
        
        self.historical_main = get_frequency_weights(main_freq, 'number')
        self.historical_pb = get_frequency_weights(pb_freq, 'powerball')
    
    def get_main_weights(self, feedback_days: int = 90) -> Dict[int, float]:
        """
        Get weights for main numbers.
        
        CRITICAL UPDATE:
        - Removes outcome-based learning (Number Blindness).
        - Returns purely historical frequency weights.
        - No feedback adjustment for specific numbers.
        """
        # PURE BLINDNESS: Return historical weights only
        # We do NOT look at which numbers won recently
        return self.historical_main.copy()
    
    def get_powerball_weights(self, feedback_days: int = 90) -> Dict[int, float]:
        """
        Get weights for Powerball.
        
        CRITICAL UPDATE:
        - Removes outcome-based learning.
        - Returns purely historical frequency weights.
        """
        # PURE BLINDNESS: Return historical weights only
        return self.historical_pb.copy()
    
    def get_weight_comparison(self) -> dict:
        """
        Get comparison of historical vs adaptive weights.
        
        Returns:
            Dict with 'main' and 'powerball' comparisons
        """
        adaptive_main = self.get_main_weights()
        adaptive_pb = self.get_powerball_weights()
        
        main_comparison = {}
        for num in range(1, 41):
            hist = self.historical_main.get(num, 0) * 100
            adapt = adaptive_main.get(num, 0) * 100
            diff = adapt - hist
            main_comparison[num] = {
                'historical': f"{hist:.2f}%",
                'adaptive': f"{adapt:.2f}%",
                'difference': f"{diff:+.2f}%"
            }
        
        pb_comparison = {}
        for num in range(1, 11):
            hist = self.historical_pb.get(num, 0) * 100
            adapt = adaptive_pb.get(num, 0) * 100
            diff = adapt - hist
            pb_comparison[num] = {
                'historical': f"{hist:.2f}%",
                'adaptive': f"{adapt:.2f}%",
                'difference': f"{diff:+.2f}%"
            }
        
        return {
            'main_numbers': main_comparison,
            'powerball': pb_comparison,
            'feedback_count': self.feedback_store.count()
        }


if __name__ == "__main__":
    print("Testing Adaptive Weight Calculator...")
    
    calc = AdaptiveWeightCalculator()
    
    print(f"\nFeedback entries: {calc.feedback_store.count()}")
    
    # Get weights
    main_weights = calc.get_main_weights()
    pb_weights = calc.get_powerball_weights()
    
    print(f"\nMain number weights sum: {sum(main_weights.values()):.6f}")
    print(f"Powerball weights sum: {sum(pb_weights.values()):.6f}")
    
    # Show top 5 main numbers by weight
    print("\nTop 5 main numbers by adaptive weight:")
    sorted_main = sorted(main_weights.items(), key=lambda x: x[1], reverse=True)
    for num, weight in sorted_main[:5]:
        hist = calc.historical_main.get(num, 0) * 100
        print(f"  Number {num:2d}: {weight*100:.2f}% (historical: {hist:.2f}%)")
    
    # Show powerball weights
    print("\nPowerball adaptive weights:")
    for num in sorted(pb_weights.keys()):
        weight = pb_weights[num] * 100
        hist = calc.historical_pb.get(num, 0) * 100
        print(f"  Powerball {num:2d}: {weight:.2f}% (historical: {hist:.2f}%)")

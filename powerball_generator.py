"""
Powerball Generator Module
Generates Powerball numbers (1-10) using weighted random sampling.
Uses only post-2001 historical data.
"""

import random
from typing import Dict
from data_loader import load_powerball_frequencies, get_frequency_weights


class PowerballGenerator:
    """
    Generates statistically-aligned Powerball numbers.
    
    Uses post-2001 historical frequency data for weighted sampling.
    Probability Space B (independent from main numbers).
    """
    
    def __init__(self, weights: Dict[int, float] = None):
        """
        Initialize generator with frequency weights.
        
        Args:
            weights: Dict mapping powerball (1-10) to probability weight.
                    If None, loads from powerball_frequencies.csv
        """
        if weights is None:
            freq_df = load_powerball_frequencies()
            self.weights = get_frequency_weights(freq_df, 'powerball')
        else:
            self.weights = weights
        
        self.numbers = list(self.weights.keys())
        self.probabilities = [self.weights[n] for n in self.numbers]
    
    def generate(self) -> int:
        """
        Generate a single Powerball number.
        
        Returns:
            Integer between 1-10 (weighted by historical frequency)
        """
        r = random.random()
        cumulative = 0
        
        for number, prob in zip(self.numbers, self.probabilities):
            cumulative += prob
            if r <= cumulative:
                return number
        
        # Fallback (should never reach)
        return self.numbers[-1]
    
    def generate_multiple(self, count: int) -> list:
        """
        Generate multiple Powerball numbers.
        
        Args:
            count: Number of Powerballs to generate
            
        Returns:
            List of Powerball numbers
        """
        return [self.generate() for _ in range(count)]
    
    def get_distribution_info(self) -> dict:
        """
        Get probability distribution information.
        
        Returns:
            Dict with number -> percentage string
        """
        return {
            num: f"{prob * 100:.1f}%"
            for num, prob in self.weights.items()
        }


if __name__ == "__main__":
    print("Testing Powerball Generator...")
    
    generator = PowerballGenerator()
    
    # Show weights
    print("\nPowerball probability distribution:")
    for num, pct in sorted(generator.get_distribution_info().items()):
        print(f"  Powerball {num}: {pct}")
    
    # Generate samples
    print("\nGenerated Powerballs (20 samples):")
    samples = generator.generate_multiple(20)
    print(f"  {samples}")
    
    # Distribution test
    print("\nDistribution test (1000 samples):")
    from collections import Counter
    large_sample = generator.generate_multiple(1000)
    counts = Counter(large_sample)
    
    for num in sorted(counts.keys()):
        expected = generator.weights[num] * 100
        actual = counts[num] / 10
        diff = actual - expected
        print(f"  Powerball {num}: {actual:.1f}% (expected {expected:.1f}%, diff {diff:+.1f}%)")

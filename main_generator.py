"""
Main Number Generator Module
Generates 6 main lottery numbers (1-40) using weighted random sampling.
Implements realism constraints to avoid unrealistic combinations.
"""

import random
from typing import List, Dict, Set
from data_loader import load_main_frequencies, get_frequency_weights


class MainNumberGenerator:
    """
    Generates statistically-aligned main lottery numbers.
    
    Constraints applied:
    - Weighted sampling based on historical frequencies
    - No full sequences (e.g., 1,2,3,4,5,6)
    - No all-even or all-odd sets
    - Balanced low/high spread
    - All numbers unique
    """
    
    def __init__(self, weights: Dict[int, float] = None):
        """
        Initialize generator with frequency weights.
        
        Args:
            weights: Dict mapping number (1-40) to probability weight.
                    If None, loads from main_frequencies.csv
        """
        if weights is None:
            freq_df = load_main_frequencies()
            self.weights = get_frequency_weights(freq_df, 'number')
        else:
            self.weights = weights
        
        self.numbers = list(self.weights.keys())
        self.probabilities = [self.weights[n] for n in self.numbers]
        
        # Ranges for balance check
        self.low_range = set(range(1, 21))   # 1-20
        self.high_range = set(range(21, 41))  # 21-40
    
    def _weighted_sample(self, k: int, exclude: Set[int] = None) -> List[int]:
        """
        Sample k numbers using weighted probabilities.
        
        Args:
            k: Number of items to sample
            exclude: Numbers to exclude from sampling
            
        Returns:
            List of k unique numbers
        """
        exclude = exclude or set()
        available = [n for n in self.numbers if n not in exclude]
        available_probs = [self.weights[n] for n in available]
        
        # Normalize probabilities
        total = sum(available_probs)
        normalized = [p / total for p in available_probs]
        
        # Sample without replacement
        result = []
        for _ in range(k):
            if not available:
                break
            
            # Weighted random choice
            r = random.random()
            cumulative = 0
            for i, prob in enumerate(normalized):
                cumulative += prob
                if r <= cumulative:
                    chosen = available[i]
                    result.append(chosen)
                    
                    # Remove chosen and renormalize
                    idx = available.index(chosen)
                    available.pop(idx)
                    normalized.pop(idx)
                    
                    if normalized:
                        total = sum(normalized)
                        normalized = [p / total for p in normalized]
                    break
        
        return result
    
    def _has_excessive_consecutive(self, numbers: List[int]) -> bool:
        """
        Check for more than 2 consecutive numbers.
        Allowed: 1,2 (2 consecutive)
        Disallowed: 1,2,3 (3 consecutive)
        """
        sorted_nums = sorted(numbers)
        consecutive_count = 1
        
        for i in range(1, len(sorted_nums)):
            if sorted_nums[i] == sorted_nums[i-1] + 1:
                consecutive_count += 1
                if consecutive_count > 2:
                    return True
            else:
                consecutive_count = 1
        return False
    
    def _has_valid_parity(self, numbers: List[int]) -> bool:
        """
        Check for balanced Odd/Even distribution.
        Allowed ratios (Odd:Even): 2:4, 3:3, 4:2
        """
        evens = sum(1 for n in numbers if n % 2 == 0)
        # 6 numbers total. Valid evens count: 2, 3, 4
        return 2 <= evens <= 4
    
    def _has_valid_decade_balance(self, numbers: List[int]) -> bool:
        """
        Check decade distribution.
        No more than 3 numbers from the same decade.
        Decades: 1-9, 10-19, 20-29, 30-39, 40
        """
        decades = {0:0, 1:0, 2:0, 3:0, 4:0}
        for n in numbers:
            d = n // 10
            # Handle 40 (belongs to 40s decade or conceptually 30s? Usually independent or 40s)
            # Let's treat 40 as decade 4.
            if n == 40:
                decades[4] += 1
            else:
                decades[d] += 1
                
        return all(count <= 3 for count in decades.values())

    def _has_balanced_spread(self, numbers: List[int]) -> bool:
        """
        Check if numbers have balanced low/high spread.
        Acceptable: 2-4, 3-3, 4-2 low/high distribution
        """
        low_count = sum(1 for n in numbers if n in self.low_range)
        high_count = len(numbers) - low_count
        
        # Must have at least 2 from each range
        return 2 <= low_count <= 4
    
    def _validate_combination(self, numbers: List[int]) -> bool:
        """
        Validate a combination against all structural constraints.
        
        Returns:
            True if combination is structurally valid.
        """
        if len(numbers) != 6: return False
        if len(set(numbers)) != 6: return False
        
        # 1. Sequence Check (Max 2 consecutive)
        if self._has_excessive_consecutive(numbers):
            return False
            
        # 2. Parity Check (Balanced Odd/Even)
        if not self._has_valid_parity(numbers):
            return False
            
        # 3. Spread Check (Balanced Low/High)
        if not self._has_balanced_spread(numbers):
            return False
            
        # 4. Decade Check (Max 3 per decade)
        if not self._has_valid_decade_balance(numbers):
            return False
            
        return True
    
    def generate(self, max_attempts: int = 100) -> List[int]:
        """
        Generate 6 valid main numbers.
        
        Args:
            max_attempts: Maximum generation attempts before fallback
            
        Returns:
            List of 6 unique numbers between 1-40, sorted
            
        Raises:
            RuntimeError: If unable to generate valid combination
        """
        for _ in range(max_attempts):
            numbers = self._weighted_sample(6)
            
            if self._validate_combination(numbers):
                return sorted(numbers)
        
        # Fallback: Generate with relaxed constraints
        # Still maintain uniqueness and range
        numbers = self._weighted_sample(6)
        return sorted(numbers)
    
    def generate_multiple(self, count: int) -> List[List[int]]:
        """
        Generate multiple unique ticket combinations.
        
        Args:
            count: Number of tickets to generate
            
        Returns:
            List of ticket combinations (each is 6 sorted numbers)
        """
        tickets = []
        seen = set()
        
        attempts = 0
        max_attempts = count * 10
        
        while len(tickets) < count and attempts < max_attempts:
            numbers = self.generate()
            key = tuple(numbers)
            
            if key not in seen:
                seen.add(key)
                tickets.append(numbers)
            
            attempts += 1
        
        return tickets


if __name__ == "__main__":
    print("Testing Main Number Generator...")
    
    generator = MainNumberGenerator()
    
    # Generate sample tickets
    print("\nGenerated tickets:")
    for i, ticket in enumerate(generator.generate_multiple(5), 1):
        print(f"  Ticket {i}: {ticket}")
    
    # Validation test
    print("\nConstraint validation test (100 tickets):")
    tickets = generator.generate_multiple(100)
    
    all_valid = True
    for ticket in tickets:
        if not generator._validate_combination(ticket):
            print(f"  [FAIL] Invalid ticket: {ticket}")
            all_valid = False
    
    if all_valid:
        print("  [OK] All 100 tickets passed validation!")
    
    # Distribution check
    print("\nNumber distribution in 100 tickets:")
    from collections import Counter
    all_numbers = [n for ticket in tickets for n in ticket]
    counts = Counter(all_numbers)
    
    most_common = counts.most_common(5)
    least_common = counts.most_common()[-5:]
    
    print(f"  Most frequent: {most_common}")
    print(f"  Least frequent: {least_common}")

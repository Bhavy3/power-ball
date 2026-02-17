"""
Ticket Composer Module
Combines main numbers and Powerball into complete lottery tickets.
Provides statistical alignment explanations and confidence scores.
"""

from typing import List, Dict, Tuple
from main_generator import MainNumberGenerator
from powerball_generator import PowerballGenerator
from data_loader import load_main_frequencies, load_powerball_frequencies


class LotteryTicket:
    """Represents a complete lottery ticket with metadata."""
    
    def __init__(self, main_numbers: List[int], powerball: int):
        """
        Create a lottery ticket.
        
        Args:
            main_numbers: 6 main numbers (1-40)
            powerball: Powerball number (1-10)
        """
        self.main_numbers = sorted(main_numbers)
        self.powerball = powerball
        self.confidence_score = 0.0
        self.explanation = ""
    
    def __str__(self) -> str:
        nums = " ".join(f"{n:2d}" for n in self.main_numbers)
        return f"[{nums}] + PB: {self.powerball}"
    
    def to_dict(self) -> dict:
        """Convert ticket to dictionary."""
        return {
            'main_numbers': self.main_numbers,
            'powerball': self.powerball,
            'confidence_score': self.confidence_score,
            'explanation': self.explanation
        }


class TicketComposer:
    """
    Composes complete lottery tickets with explanations.
    
    Maintains strict separation between:
    - Probability Space A: Main numbers (1-40)
    - Probability Space B: Powerball (1-10)
    """
    
    RANDOMNESS_DISCLAIMER = (
        "Note: Lottery draws are random and independent events. "
        "This ticket is statistically aligned with historical patterns "
        "but does not guarantee any outcome."
    )
    
    def __init__(self):
        """Initialize composers for both probability spaces."""
        self.main_gen = MainNumberGenerator()
        self.pb_gen = PowerballGenerator()
        
        # Load frequency data for analysis
        self.main_freq = load_main_frequencies()
        self.pb_freq = load_powerball_frequencies()
    
    def _calculate_confidence(self, ticket: LotteryTicket) -> float:
        """
        Calculate confidence score based on historical similarity.
        
        Score components:
        - Frequency alignment (how well numbers match historical distribution)
        - Balance score (low/high, odd/even distribution)
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0
        
        # Frequency alignment for main numbers
        freq_dict = dict(zip(self.main_freq['number'], self.main_freq['frequency']))
        avg_freq = sum(freq_dict.values()) / len(freq_dict)
        
        freq_scores = []
        for num in ticket.main_numbers:
            num_freq = freq_dict.get(num, 0)
            # Score based on how close to average
            deviation = abs(num_freq - avg_freq) / avg_freq
            freq_scores.append(max(0, 1 - deviation))
        
        freq_alignment = sum(freq_scores) / len(freq_scores)
        
        # Balance scores
        evens = sum(1 for n in ticket.main_numbers if n % 2 == 0)
        odds = 6 - evens
        even_odd_balance = 1 - abs(evens - 3) / 3  # Best at 3-3
        
        lows = sum(1 for n in ticket.main_numbers if n <= 20)
        highs = 6 - lows
        low_high_balance = 1 - abs(lows - 3) / 3  # Best at 3-3
        
        # Powerball frequency alignment
        pb_dict = dict(zip(self.pb_freq['powerball'], self.pb_freq['frequency']))
        pb_avg = sum(pb_dict.values()) / len(pb_dict)
        pb_freq = pb_dict.get(ticket.powerball, 0)
        pb_alignment = max(0, 1 - abs(pb_freq - pb_avg) / pb_avg)
        
        # Combined score (weighted)
        score = (
            0.40 * freq_alignment +
            0.20 * even_odd_balance +
            0.20 * low_high_balance +
            0.20 * pb_alignment
        )
        
        return round(min(1.0, max(0.0, score)), 2)
    
    def _generate_explanation(self, ticket: LotteryTicket) -> str:
        """Generate statistical alignment explanation for a ticket."""
        # Analyze main numbers
        evens = sum(1 for n in ticket.main_numbers if n % 2 == 0)
        lows = sum(1 for n in ticket.main_numbers if n <= 20)
        
        # Build explanation
        parts = [
            f"Main numbers: {evens} even, {6-evens} odd",
            f"Range distribution: {lows} low (1-20), {6-lows} high (21-40)",
            f"Powerball {ticket.powerball}: {self._get_pb_percentile(ticket.powerball)}",
            f"Confidence: {ticket.confidence_score:.0%}",
        ]
        
        return " | ".join(parts)
    
    def _get_pb_percentile(self, powerball: int) -> str:
        """Get percentile description for a powerball number."""
        pb_dict = dict(zip(self.pb_freq['powerball'], self.pb_freq['frequency']))
        freq = pb_dict.get(powerball, 0)
        total = sum(pb_dict.values())
        pct = (freq / total) * 100
        
        if pct >= 12:
            return "high frequency"
        elif pct >= 9:
            return "average frequency"
        else:
            return "lower frequency"
    
    def compose(self) -> LotteryTicket:
        """
        Compose a complete lottery ticket.
        
        Returns:
            LotteryTicket with main numbers, powerball, confidence, and explanation
        """
        main_numbers = self.main_gen.generate()
        powerball = self.pb_gen.generate()
        
        ticket = LotteryTicket(main_numbers, powerball)
        ticket.confidence_score = self._calculate_confidence(ticket)
        ticket.explanation = self._generate_explanation(ticket)
        
        return ticket
    
    def compose_multiple(self, count: int) -> List[LotteryTicket]:
        """
        Compose multiple unique tickets.
        
        Args:
            count: Number of tickets to generate
            
        Returns:
            List of unique LotteryTicket objects
        """
        tickets = []
        seen_main = set()
        
        attempts = 0
        max_attempts = count * 10
        
        while len(tickets) < count and attempts < max_attempts:
            ticket = self.compose()
            key = tuple(ticket.main_numbers)
            
            if key not in seen_main:
                seen_main.add(key)
                tickets.append(ticket)
            
            attempts += 1
        
        return tickets
    
    def format_ticket_output(self, ticket: LotteryTicket) -> str:
        """Format a ticket for display with full details."""
        lines = [
            "=" * 55,
            f"LOTTERY TICKET",
            "=" * 55,
            f"Main Numbers: {' '.join(f'{n:2d}' for n in ticket.main_numbers)}",
            f"Powerball:    {ticket.powerball}",
            "-" * 55,
            f"Statistical Analysis:",
            f"  {ticket.explanation}",
            "-" * 55,
            f"{self.RANDOMNESS_DISCLAIMER}",
            "=" * 55,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    print("Testing Ticket Composer...")
    
    composer = TicketComposer()
    
    # Generate single ticket with full output
    print("\n" + "=" * 60)
    print("SINGLE TICKET GENERATION")
    print("=" * 60)
    
    ticket = composer.compose()
    print(composer.format_ticket_output(ticket))
    
    # Generate multiple tickets
    print("\n" + "=" * 60)
    print("BATCH GENERATION (5 tickets)")
    print("=" * 60)
    
    tickets = composer.compose_multiple(5)
    for i, t in enumerate(tickets, 1):
        print(f"\nTicket {i}: {t}")
        print(f"         Confidence: {t.confidence_score:.0%}")
        print(f"         {t.explanation}")

"""
Lottery Statistical Optimization Engine
Main entry point for ticket generation with adaptive learning.

This is NOT a prediction system. It provides statistically-aligned,
realistic lottery combinations based on historical distributions.
"""

import argparse
from typing import List
from ticket_composer import TicketComposer, LotteryTicket
from main_generator import MainNumberGenerator
from powerball_generator import PowerballGenerator
from adaptive_weights import AdaptiveWeightCalculator
from feedback_store import FeedbackStore


class LotteryEngine:
    """
    Main engine for lottery ticket generation.
    
    Features:
    - Weighted sampling from historical frequencies
    - Realism constraints (no sequences, balanced distribution)
    - Adaptive learning from user feedback
    - Statistical alignment explanations
    
    This system respects the fundamental truth that lottery draws
    are random and independent events.
    """
    
    SYSTEM_DISCLAIMER = """
================================================================================
                    LOTTERY STATISTICAL OPTIMIZATION ENGINE
================================================================================

IMPORTANT DISCLAIMER:
- This is NOT a prediction system and does NOT guarantee wins
- Lottery draws are random and independent events
- Historical data provides descriptive, not predictive, insight
- This system improves selection strategy quality, not probability

The engine generates statistically-aligned combinations based on:
- Historical frequency distributions (primary)
- User feedback signals (secondary, max 15% influence)
- Realism constraints (balanced, non-sequential)

================================================================================
"""
    
    def __init__(self, use_adaptive: bool = True):
        """
        Initialize the lottery engine.
        
        Args:
            use_adaptive: Whether to use adaptive weights from feedback
        """
        self.use_adaptive = use_adaptive
        self.feedback_store = FeedbackStore()
        
        if use_adaptive:
            # Use adaptive weights
            calc = AdaptiveWeightCalculator(self.feedback_store)
            main_weights = calc.get_main_weights()
            pb_weights = calc.get_powerball_weights()
            
            self.main_gen = MainNumberGenerator(weights=main_weights)
            self.pb_gen = PowerballGenerator(weights=pb_weights)
        else:
            # Pure historical weights
            self.main_gen = MainNumberGenerator()
            self.pb_gen = PowerballGenerator()
        
        self.composer = TicketComposer()
    
    def generate_ticket(self) -> LotteryTicket:
        """Generate a single lottery ticket."""
        return self.composer.compose()
    
    def generate_tickets(self, count: int) -> List[LotteryTicket]:
        """Generate multiple unique tickets."""
        return self.composer.compose_multiple(count)
    
    def submit_feedback(self, 
                        main_numbers: List[int],
                        powerball: int,
                        draw_date: str,
                        match_category: str):
        """
        Submit user feedback to improve future generations.
        
        Args:
            main_numbers: User's selected main numbers
            powerball: User's selected powerball
            draw_date: Date of the draw (YYYY-MM-DD)
            match_category: "0-1", "2-3", or "4+"
        """
        self.feedback_store.add_feedback(
            main_numbers=main_numbers,
            powerball=powerball,
            draw_date=draw_date,
            match_category=match_category
        )
        print(f"Feedback recorded. Total entries: {self.feedback_store.count()}")
    
    def print_ticket(self, ticket: LotteryTicket):
        """Print a single ticket with full details."""
        print(self.composer.format_ticket_output(ticket))
    
    def print_tickets_summary(self, tickets: List[LotteryTicket]):
        """Print a summary of multiple tickets."""
        print("\n" + "=" * 60)
        print("GENERATED LOTTERY TICKETS")
        print("=" * 60)
        
        for i, ticket in enumerate(tickets, 1):
            print(f"\n  Ticket {i}: {ticket}")
            print(f"           Confidence: {ticket.confidence_score:.0%}")
        
        print("\n" + "-" * 60)
        print("Note: All selections are statistically aligned with")
        print("historical patterns. Randomness remains dominant.")
        print("=" * 60)
    
    def get_stats(self) -> dict:
        """Get engine statistics."""
        return {
            'adaptive_mode': self.use_adaptive,
            'feedback_entries': self.feedback_store.count(),
            'historical_weight': '85%' if self.use_adaptive else '100%',
            'feedback_weight': '15%' if self.use_adaptive else '0%'
        }


def main():
    """Command-line interface for the lottery engine."""
    parser = argparse.ArgumentParser(
        description="Lottery Statistical Optimization Engine",
        epilog="Note: This is NOT a prediction system."
    )
    
    parser.add_argument(
        '-n', '--count',
        type=int,
        default=1,
        help="Number of tickets to generate (default: 1)"
    )
    
    parser.add_argument(
        '--pure',
        action='store_true',
        help="Use pure historical weights (no adaptive learning)"
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help="Show engine statistics only"
    )
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = LotteryEngine(use_adaptive=not args.pure)
    
    # Print disclaimer
    print(LotteryEngine.SYSTEM_DISCLAIMER)
    
    if args.stats:
        # Show stats only
        stats = engine.get_stats()
        print("Engine Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    # Generate tickets
    if args.count == 1:
        ticket = engine.generate_ticket()
        engine.print_ticket(ticket)
    else:
        tickets = engine.generate_tickets(args.count)
        engine.print_tickets_summary(tickets)


if __name__ == "__main__":
    main()

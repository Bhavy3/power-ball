"""
Feedback Store Module
Stores and manages user feedback for adaptive weight calculation.
Implements time-aware feedback storage and retrieval.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


# Feedback file path
FEEDBACK_FILE = Path(__file__).parent / "user_feedback.json"


class FeedbackEntry:
    """Represents a single user feedback entry."""
    
    MATCH_CATEGORIES = ["0-1", "2-3", "4+"]
    
    def __init__(self, 
                 main_numbers: List[int],
                 powerball: int,
                 draw_date: str,
                 match_category: str,
                 submitted_at: str = None):
        """
        Create a feedback entry.
        
        Args:
            main_numbers: User's selected main numbers
            powerball: User's selected powerball
            draw_date: Date of the draw
            match_category: "0-1", "2-3", or "4+"
            submitted_at: Timestamp (auto-generated if not provided)
        """
        self.main_numbers = sorted(main_numbers)
        self.powerball = powerball
        self.draw_date = draw_date
        self.match_category = match_category
        self.submitted_at = submitted_at or datetime.now().isoformat()
        
        # Validate match category
        if match_category not in self.MATCH_CATEGORIES:
            raise ValueError(f"Invalid match category. Must be one of {self.MATCH_CATEGORIES}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage."""
        return {
            'main_numbers': self.main_numbers,
            'powerball': self.powerball,
            'draw_date': self.draw_date,
            'match_category': self.match_category,
            'submitted_at': self.submitted_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FeedbackEntry':
        """Create from dictionary."""
        return cls(
            main_numbers=data['main_numbers'],
            powerball=data['powerball'],
            draw_date=data['draw_date'],
            match_category=data['match_category'],
            submitted_at=data.get('submitted_at')
        )
    
    def get_weight_signal(self) -> float:
        """
        Get weight signal based on match category.
        
        Returns:
            Signal value:
            - 4+ matches: 1.0 (strong positive)
            - 2-3 matches: 0.5 (moderate)
            - 0-1 matches: 0.1 (weak)
        """
        signals = {
            "4+": 1.0,
            "2-3": 0.5,
            "0-1": 0.1
        }
        return signals.get(self.match_category, 0.1)


class FeedbackStore:
    """
    Manages persistent storage of user feedback.
    
    Feedback is stored as a JSON file and used for adaptive weight calculation.
    Historical data always dominates (85% historical, 15% feedback).
    """
    
    def __init__(self, filepath: Path = None):
        """
        Initialize feedback store.
        
        Args:
            filepath: Path to JSON file (default: user_feedback.json)
        """
        self.filepath = filepath or FEEDBACK_FILE
        self._entries: List[FeedbackEntry] = []
        self._load()
    
    def _load(self):
        """Load feedback from file."""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self._entries = [FeedbackEntry.from_dict(e) for e in data]
            except (json.JSONDecodeError, KeyError):
                self._entries = []
        else:
            self._entries = []
    
    def _save(self):
        """Save feedback to file."""
        with open(self.filepath, 'w') as f:
            json.dump([e.to_dict() for e in self._entries], f, indent=2)
    
    def add(self, entry: FeedbackEntry):
        """
        Add a feedback entry.
        
        Args:
            entry: FeedbackEntry to add
        """
        self._entries.append(entry)
        self._save()
    
    def add_feedback(self,
                     main_numbers: List[int],
                     powerball: int,
                     draw_date: str,
                     match_category: str):
        """
        Convenience method to add feedback directly.
        
        Args:
            main_numbers: User's selected main numbers
            powerball: User's selected powerball
            draw_date: Date of the draw (YYYY-MM-DD)
            match_category: "0-1", "2-3", or "4+"
        """
        entry = FeedbackEntry(main_numbers, powerball, draw_date, match_category)
        self.add(entry)
    
    def get_all(self) -> List[FeedbackEntry]:
        """Get all feedback entries."""
        return self._entries.copy()
    
    def get_recent(self, days: int = 90) -> List[FeedbackEntry]:
        """
        Get feedback from the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of recent feedback entries
        """
        cutoff = datetime.now().timestamp() - (days * 86400)
        recent = []
        
        for entry in self._entries:
            try:
                entry_time = datetime.fromisoformat(entry.submitted_at).timestamp()
                if entry_time >= cutoff:
                    recent.append(entry)
            except (ValueError, TypeError):
                pass
        
        return recent
    
    def get_number_signals(self, days: int = 90) -> Dict[int, float]:
        """
        Get aggregated weight signals for each main number.
        
        Args:
            days: Number of days to consider
            
        Returns:
            Dict mapping number (1-40) to aggregated signal
        """
        signals = {i: 0.0 for i in range(1, 41)}
        counts = {i: 0 for i in range(1, 41)}
        
        for entry in self.get_recent(days):
            signal = entry.get_weight_signal()
            for num in entry.main_numbers:
                signals[num] += signal
                counts[num] += 1
        
        # Average signals
        for num in signals:
            if counts[num] > 0:
                signals[num] /= counts[num]
        
        return signals
    
    def get_powerball_signals(self, days: int = 90) -> Dict[int, float]:
        """
        Get aggregated weight signals for each powerball.
        
        Args:
            days: Number of days to consider
            
        Returns:
            Dict mapping powerball (1-10) to aggregated signal
        """
        signals = {i: 0.0 for i in range(1, 11)}
        counts = {i: 0 for i in range(1, 11)}
        
        for entry in self.get_recent(days):
            signal = entry.get_weight_signal()
            signals[entry.powerball] += signal
            counts[entry.powerball] += 1
        
        # Average signals
        for num in signals:
            if counts[num] > 0:
                signals[num] /= counts[num]
        
        return signals
    
    def count(self) -> int:
        """Get total number of feedback entries."""
        return len(self._entries)
    
    def clear(self):
        """Clear all feedback (use with caution)."""
        self._entries = []
        self._save()


if __name__ == "__main__":
    print("Testing Feedback Store...")
    
    # Create temporary store for testing
    test_store = FeedbackStore(Path("test_feedback.json"))
    test_store.clear()
    
    # Add sample feedback
    print("\nAdding sample feedback...")
    test_store.add_feedback([1, 5, 12, 23, 31, 40], 7, "2025-01-15", "2-3")
    test_store.add_feedback([3, 8, 15, 22, 35, 38], 4, "2025-01-18", "0-1")
    test_store.add_feedback([2, 10, 18, 25, 33, 39], 9, "2025-01-20", "4+")
    
    print(f"   Total entries: {test_store.count()}")
    
    # Show entries
    print("\nAll entries:")
    for entry in test_store.get_all():
        print(f"  {entry.main_numbers} + PB:{entry.powerball} = {entry.match_category}")
    
    # Show signals
    print("\nMain number signals (top 5):")
    signals = test_store.get_number_signals()
    sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
    for num, signal in sorted_signals[:5]:
        if signal > 0:
            print(f"  Number {num}: {signal:.2f}")
    
    print("\nPowerball signals:")
    pb_signals = test_store.get_powerball_signals()
    for pb, signal in sorted(pb_signals.items()):
        if signal > 0:
            print(f"  Powerball {pb}: {signal:.2f}")
    
    # Cleanup
    Path("test_feedback.json").unlink(missing_ok=True)
    print("\n[OK] Test complete!")

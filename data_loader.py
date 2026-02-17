"""
Data Loader Module
Loads and validates all CSV files for the Lottery Statistical Optimization Engine.
Enforces schema contracts and handles data integrity.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

# Base directory
BASE_DIR = Path(__file__).parent

# File paths
MAIN_DRAWS_PATH = BASE_DIR / "main_draws.csv"
MAIN_FREQ_PATH = BASE_DIR / "main_frequencies.csv"
PB_DRAWS_PATH = BASE_DIR / "powerball_draws.csv"
PB_FREQ_PATH = BASE_DIR / "powerball_frequencies.csv"

# Constants
MAIN_NUMBER_RANGE = (1, 40)
POWERBALL_RANGE = (1, 10)
POWERBALL_START_YEAR = 2001


class DataLoadError(Exception):
    """Raised when data loading or validation fails."""
    pass


def load_main_draws() -> pd.DataFrame:
    """
    Load main draws data.
    
    Returns:
        DataFrame with columns: date, draw_number, n1-n6
        
    Raises:
        DataLoadError: If file missing or validation fails
    """
    if not MAIN_DRAWS_PATH.exists():
        raise DataLoadError(f"Main draws file not found: {MAIN_DRAWS_PATH}")
    
    df = pd.read_csv(MAIN_DRAWS_PATH)
    
    # Validate required columns
    required_cols = ['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise DataLoadError(f"Missing columns in main_draws: {missing}")
    
    # Validate number ranges
    number_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
    for col in number_cols:
        if df[col].min() < MAIN_NUMBER_RANGE[0] or df[col].max() > MAIN_NUMBER_RANGE[1]:
            raise DataLoadError(f"Column {col} has values outside range {MAIN_NUMBER_RANGE}")
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    
    return df


def load_main_frequencies() -> pd.DataFrame:
    """
    Load main number frequencies.
    
    Returns:
        DataFrame with columns: number, frequency
    """
    if not MAIN_FREQ_PATH.exists():
        raise DataLoadError(f"Main frequencies file not found: {MAIN_FREQ_PATH}")
    
    df = pd.read_csv(MAIN_FREQ_PATH)
    
    # Validate
    if 'number' not in df.columns or 'frequency' not in df.columns:
        raise DataLoadError("main_frequencies.csv must have 'number' and 'frequency' columns")
    
    if len(df) != 40:
        raise DataLoadError(f"Expected 40 numbers in frequencies, got {len(df)}")
    
    return df


def load_powerball_draws() -> pd.DataFrame:
    """
    Load Powerball draws data.
    
    Returns:
        DataFrame with columns: date, powerball
        
    Note:
        Only contains data from 2001 onwards (when Powerball was introduced)
    """
    if not PB_DRAWS_PATH.exists():
        raise DataLoadError(f"Powerball draws file not found: {PB_DRAWS_PATH}")
    
    df = pd.read_csv(PB_DRAWS_PATH)
    
    # Validate columns
    if 'date' not in df.columns or 'powerball' not in df.columns:
        raise DataLoadError("powerball_draws.csv must have 'date' and 'powerball' columns")
    
    # Validate range
    if df['powerball'].min() < POWERBALL_RANGE[0] or df['powerball'].max() > POWERBALL_RANGE[1]:
        raise DataLoadError(f"Powerball values outside range {POWERBALL_RANGE}")
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    
    return df


def load_powerball_frequencies() -> pd.DataFrame:
    """
    Load Powerball frequencies.
    
    Returns:
        DataFrame with columns: powerball, frequency
    """
    if not PB_FREQ_PATH.exists():
        raise DataLoadError(f"Powerball frequencies file not found: {PB_FREQ_PATH}")
    
    df = pd.read_csv(PB_FREQ_PATH)
    
    # Validate
    if 'powerball' not in df.columns or 'frequency' not in df.columns:
        raise DataLoadError("powerball_frequencies.csv must have 'powerball' and 'frequency' columns")
    
    if len(df) != 10:
        raise DataLoadError(f"Expected 10 powerball numbers, got {len(df)}")
    
    return df


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all data files at once.
    
    Returns:
        Tuple of (main_draws, main_frequencies, powerball_draws, powerball_frequencies)
    """
    return (
        load_main_draws(),
        load_main_frequencies(),
        load_powerball_draws(),
        load_powerball_frequencies()
    )


def get_frequency_weights(frequencies_df: pd.DataFrame, 
                          number_col: str, 
                          freq_col: str = 'frequency') -> dict:
    """
    Convert frequency DataFrame to normalized probability weights.
    
    Args:
        frequencies_df: DataFrame with number and frequency columns
        number_col: Name of the number column
        freq_col: Name of the frequency column
        
    Returns:
        Dict mapping number -> probability weight (sums to 1.0)
    """
    total = frequencies_df[freq_col].sum()
    weights = {}
    
    for _, row in frequencies_df.iterrows():
        weights[int(row[number_col])] = row[freq_col] / total
    
    return weights


if __name__ == "__main__":
    # Test loading
    print("Testing data loader...")
    
    try:
        main_draws, main_freq, pb_draws, pb_freq = load_all_data()
        
        print(f"[OK] Main draws: {len(main_draws)} records")
        print(f"[OK] Main frequencies: {len(main_freq)} numbers")
        print(f"[OK] Powerball draws: {len(pb_draws)} records")
        print(f"[OK] Powerball frequencies: {len(pb_freq)} numbers")
        
        # Test weight generation
        main_weights = get_frequency_weights(main_freq, 'number')
        pb_weights = get_frequency_weights(pb_freq, 'powerball')
        
        print(f"\n[OK] Main number weights sum: {sum(main_weights.values()):.4f}")
        print(f"[OK] Powerball weights sum: {sum(pb_weights.values()):.4f}")
        
        print("\nAll tests passed!")
        
    except DataLoadError as e:
        print(f"[ERROR] {e}")

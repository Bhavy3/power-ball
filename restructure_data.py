"""
Data Restructuring Script
Standardizes column names and creates proper frequency files.
Run once to prepare data for the Lottery Statistical Optimization Engine.
"""

import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
MAIN_DRAWS_PATH = BASE_DIR / "main_draws.csv"
MAIN_FREQ_PATH = BASE_DIR / "main_frequencies.csv"
PB_OLD_PATH = BASE_DIR / "powerball_frequencies.csv"  # Will become powerball_draws
PB_DRAWS_PATH = BASE_DIR / "powerball_draws.csv"
PB_FREQ_PATH = BASE_DIR / "powerball_frequencies_new.csv"

def restructure_main_draws():
    """Standardize main_draws.csv column names."""
    print("[1/4] Restructuring main_draws.csv...")
    
    df = pd.read_csv(MAIN_DRAWS_PATH)
    
    # Rename columns: 1,2,3,4,5,6 -> n1,n2,n3,n4,n5,n6
    rename_map = {
        'Date': 'date',
        'Draw Number': 'draw_number',
        '1': 'n1', '2': 'n2', '3': 'n3',
        '4': 'n4', '5': 'n5', '6': 'n6'
    }
    
    df = df.rename(columns=rename_map)
    
    # Keep only required columns (exclude Bonus Number)
    required_cols = ['date', 'draw_number', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']
    df = df[required_cols]
    
    # Save
    df.to_csv(MAIN_DRAWS_PATH, index=False)
    print(f"   [OK] Saved: {len(df)} records")
    print(f"   [OK] Columns: {list(df.columns)}")
    return df

def restructure_main_frequencies():
    """Standardize main_frequencies.csv column names."""
    print("\n[2/4] Restructuring main_frequencies.csv...")
    
    df = pd.read_csv(MAIN_FREQ_PATH)
    
    # Rename columns to match prompt
    rename_map = {
        'Winning Number': 'number',
        'Frequency': 'frequency'
    }
    df = df.rename(columns=rename_map)
    
    # Ensure lowercase column names
    df.columns = df.columns.str.lower()
    
    # Save
    df.to_csv(MAIN_FREQ_PATH, index=False)
    print(f"   [OK] Saved: {len(df)} records")
    print(f"   [OK] Columns: {list(df.columns)}")
    return df

def restructure_powerball():
    """
    Current powerball_frequencies.csv is actually draw-level data.
    1. Rename it to powerball_draws.csv
    2. Create proper aggregated powerball_frequencies.csv
    """
    print("\n[3/4] Restructuring Powerball files...")
    
    # Read the current file (it's actually draws, not frequencies)
    df = pd.read_csv(PB_OLD_PATH)
    
    # Standardize column names
    rename_map = {
        'Date': 'date',
        'Powerball Number': 'powerball'
    }
    df = df.rename(columns=rename_map)
    df.columns = df.columns.str.lower()
    
    # Save as powerball_draws.csv
    df.to_csv(PB_DRAWS_PATH, index=False)
    print(f"   [OK] Created powerball_draws.csv: {len(df)} records")
    
    # Create aggregated frequencies
    freq_df = df['powerball'].value_counts().reset_index()
    freq_df.columns = ['powerball', 'frequency']
    freq_df = freq_df.sort_values('powerball').reset_index(drop=True)
    
    # Save new frequencies file
    freq_df.to_csv(PB_FREQ_PATH, index=False)
    print(f"   [OK] Created powerball_frequencies_new.csv: {len(freq_df)} entries")
    print(f"   Frequency distribution:")
    for _, row in freq_df.iterrows():
        print(f"      Powerball {int(row['powerball'])}: {int(row['frequency'])} occurrences")
    
    return df, freq_df

def validate_data():
    """Validate all data files after restructuring."""
    print("\n[4/4] VALIDATION...")
    
    # Validate main_draws
    main_df = pd.read_csv(MAIN_DRAWS_PATH)
    assert all(col in main_df.columns for col in ['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']), "main_draws columns invalid"
    assert main_df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].min().min() >= 1, "Main numbers below 1"
    assert main_df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].max().max() <= 40, "Main numbers above 40"
    print("   [OK] main_draws.csv: VALID")
    
    # Validate main_frequencies
    main_freq = pd.read_csv(MAIN_FREQ_PATH)
    assert 'number' in main_freq.columns and 'frequency' in main_freq.columns, "main_frequencies columns invalid"
    assert len(main_freq) == 40, f"Expected 40 numbers, got {len(main_freq)}"
    print("   [OK] main_frequencies.csv: VALID")
    
    # Validate powerball_draws
    pb_draws = pd.read_csv(PB_DRAWS_PATH)
    assert 'date' in pb_draws.columns and 'powerball' in pb_draws.columns, "powerball_draws columns invalid"
    assert pb_draws['powerball'].min() >= 1, "Powerball below 1"
    assert pb_draws['powerball'].max() <= 10, "Powerball above 10"
    print("   [OK] powerball_draws.csv: VALID")
    
    # Validate powerball_frequencies
    pb_freq = pd.read_csv(PB_FREQ_PATH)
    assert 'powerball' in pb_freq.columns and 'frequency' in pb_freq.columns, "powerball_frequencies columns invalid"
    assert len(pb_freq) == 10, f"Expected 10 powerball numbers, got {len(pb_freq)}"
    print("   [OK] powerball_frequencies_new.csv: VALID")
    
    print("\n*** ALL DATA VALIDATED SUCCESSFULLY! ***")

def main():
    print("=" * 60)
    print("LOTTERY DATA RESTRUCTURING SCRIPT")
    print("=" * 60)
    
    restructure_main_draws()
    restructure_main_frequencies()
    restructure_powerball()
    validate_data()
    
    print("\n" + "=" * 60)
    print("DATA RESTRUCTURING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("   1. Delete old powerball_frequencies.csv")
    print("   2. Rename powerball_frequencies_new.csv -> powerball_frequencies.csv")
    print("   3. Run the core engine")

if __name__ == "__main__":
    main()

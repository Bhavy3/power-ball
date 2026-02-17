"""
Data Preparation Script for Audit Pipeline
Converts existing CSV formats to the required audit format.

Converts:
1. predicted_results.csv -> predictions.csv (with n1-n6 columns)
2. main_draws.csv + powerball_draws.csv -> actual_draws.csv
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent


def prepare_predictions():
    """
    Convert predicted_results.csv to predictions.csv format.
    
    Input format:
        draw_date, predicted_main_numbers (comma-separated), predicted_powerball
    
    Output format:
        prediction_date, line_id, n1, n2, n3, n4, n5, n6, powerball
    """
    input_path = BASE_DIR / "predicted_results.csv"
    output_path = BASE_DIR / "predictions.csv"
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return False
    
    df = pd.read_csv(input_path)
    
    # Parse and expand main numbers
    rows = []
    for idx, row in df.iterrows():
        try:
            # Parse main numbers
            main_nums_str = str(row['predicted_main_numbers']).strip('"[]')
            main_nums = [int(n.strip()) for n in main_nums_str.split(',')]
            
            if len(main_nums) != 6:
                print(f"[WARN] Row {idx}: Expected 6 main numbers, got {len(main_nums)}")
                continue
            
            # Build new row
            new_row = {
                'prediction_date': row['draw_date'],
                'line_id': idx + 1,
                'n1': main_nums[0],
                'n2': main_nums[1],
                'n3': main_nums[2],
                'n4': main_nums[3],
                'n5': main_nums[4],
                'n6': main_nums[5],
                'powerball': int(row['predicted_powerball'])
            }
            rows.append(new_row)
        except Exception as e:
            print(f"[WARN] Row {idx}: Parse error - {e}")
            continue
    
    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_path, index=False)
    print(f"[OK] Created predictions.csv with {len(output_df)} rows")
    return True


def prepare_actual_draws():
    """
    Merge main_draws.csv and powerball_draws.csv into actual_draws.csv.
    
    Output format:
        draw_date, a1, a2, a3, a4, a5, a6, actual_powerball
    """
    main_path = BASE_DIR / "main_draws.csv"
    pb_path = BASE_DIR / "powerball_draws.csv"
    output_path = BASE_DIR / "actual_draws.csv"
    
    if not main_path.exists():
        print(f"[ERROR] Main draws file not found: {main_path}")
        return False
    
    if not pb_path.exists():
        print(f"[ERROR] Powerball draws file not found: {pb_path}")
        return False
    
    # Load data
    main_df = pd.read_csv(main_path)
    pb_df = pd.read_csv(pb_path)
    
    # Standardize date columns
    main_df['date'] = pd.to_datetime(main_df['date'], format='mixed')
    pb_df['date'] = pd.to_datetime(pb_df['date'], format='mixed')
    
    # Merge on date
    merged = main_df.merge(pb_df, on='date', how='left')
    
    # Build output format
    output_df = pd.DataFrame({
        'draw_date': merged['date'],
        'a1': merged['n1'],
        'a2': merged['n2'],
        'a3': merged['n3'],
        'a4': merged['n4'],
        'a5': merged['n5'],
        'a6': merged['n6'],
        'actual_powerball': merged['powerball']
    })
    
    # Keep only rows with valid powerball (post-2001)
    output_df = output_df.dropna(subset=['actual_powerball'])
    output_df['actual_powerball'] = output_df['actual_powerball'].astype(int)
    
    output_df.to_csv(output_path, index=False)
    print(f"[OK] Created actual_draws.csv with {len(output_df)} rows")
    return True


def validate_data():
    """Validate the created files."""
    predictions_path = BASE_DIR / "predictions.csv"
    actual_path = BASE_DIR / "actual_draws.csv"
    
    errors = []
    
    # Validate predictions
    if predictions_path.exists():
        df = pd.read_csv(predictions_path)
        for idx, row in df.iterrows():
            # Check main number range
            for col in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
                val = row[col]
                if not (1 <= val <= 40):
                    errors.append(f"predictions.csv row {idx}: {col}={val} out of range [1-40]")
            
            # Check powerball range
            if not (1 <= row['powerball'] <= 10):
                errors.append(f"predictions.csv row {idx}: powerball={row['powerball']} out of range [1-10]")
    
    # Validate actual draws
    if actual_path.exists():
        df = pd.read_csv(actual_path)
        for idx, row in df.iterrows():
            # Check main number range
            for col in ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']:
                val = row[col]
                if not (1 <= val <= 40):
                    errors.append(f"actual_draws.csv row {idx}: {col}={val} out of range [1-40]")
            
            # Check powerball range
            if not (1 <= row['actual_powerball'] <= 10):
                errors.append(f"actual_draws.csv row {idx}: powerball={row['actual_powerball']} out of range [1-10]")
    
    if errors:
        print(f"\n[ERROR] Validation found {len(errors)} errors:")
        for err in errors[:10]:
            print(f"   {err}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")
        return False
    
    print("[OK] All data validated successfully")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("DATA PREPARATION FOR AUDIT PIPELINE")
    print("=" * 60)
    print()
    
    print("Step 1: Converting predictions...")
    prepare_predictions()
    print()
    
    print("Step 2: Merging actual draws...")
    prepare_actual_draws()
    print()
    
    print("Step 3: Validating data...")
    validate_data()
    print()
    
    print("=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


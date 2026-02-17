"""
Generate and Audit - Complete Workflow
Generates fresh predictions and then runs enforced audit.

This script ensures:
1. Fresh predictions are generated with proper timestamps
2. Predictions are saved with freshness tracking
3. Enforced audit runs immediately after generation

This is the CORRECT way to run the audit pipeline.
"""

import argparse
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import uuid

# Import system components
from engine import LotteryEngine
from enforced_audit import EnforcedAuditEngine

BASE_DIR = Path(__file__).parent


def append_audit_history(three_plus_rate, four_plus_rate, ticket_count):
    """Persist audit metrics to .audit_history.json after every run."""
    history_file = os.path.join(str(BASE_DIR), ".audit_history.json")

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "three_plus_rate": float(three_plus_rate),
        "four_plus_rate": float(four_plus_rate),
        "ticket_count": int(ticket_count)
    }

    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                data = json.load(f)
            # Handle both {"runs": [...]} and bare [...] formats
            if isinstance(data, dict) and "runs" in data:
                data["runs"].append(entry)
            elif isinstance(data, list):
                data.append(entry)
            else:
                data = [entry]
        except Exception:
            data = [entry]
    else:
        data = [entry]

    with open(history_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  [HISTORY] Appended to .audit_history.json")
    print(f"    three_plus_rate: {three_plus_rate:.4f}")
    print(f"    four_plus_rate:  {four_plus_rate:.4f}")
    print(f"    ticket_count:    {ticket_count}")


def generate_fresh_predictions(count: int = 10, use_historical_dates: bool = True) -> pd.DataFrame:
    """
    Generate fresh predictions with proper tracking metadata.
    
    Args:
        count: Number of predictions to generate
        use_historical_dates: If True, backdate predictions to match actual draws
                            (for testing/audit purposes)
    
    Returns DataFrame with:
    - prediction_date: Date predictions are made for
    - generation_timestamp: When predictions were generated
    - run_id: Unique identifier for this generation run
    - n1-n6: Main numbers
    - powerball: Powerball number
    """
    engine = LotteryEngine(use_adaptive=True)
    
    # Generate run metadata
    run_id = str(uuid.uuid4())[:8]
    generation_time = datetime.now()
    
    print(f"\n[GENERATING] Fresh predictions...")
    print(f"  Run ID: {run_id}")
    print(f"  Timestamp: {generation_time}")
    print(f"  Count: {count}")
    
    # Generate tickets
    tickets = engine.generate_tickets(count)
    
    # If using historical dates, get dates from actual_draws.csv
    prediction_dates = []
    if use_historical_dates:
        try:
            actuals_path = BASE_DIR / "actual_draws.csv"
            if actuals_path.exists():
                actuals_df = pd.read_csv(actuals_path)
                actuals_df['draw_date'] = pd.to_datetime(actuals_df['draw_date'])
                actuals_df = actuals_df.sort_values('draw_date', ascending=False)
                
                # Get dates BEFORE the draws (predictions made day before)
                for i, row in actuals_df.head(count * 2).iterrows():
                    draw_date = pd.to_datetime(row['draw_date'])
                    pred_date = draw_date - timedelta(days=1)  # Predict day before
                    prediction_dates.append(pred_date)
                    if len(prediction_dates) >= count:
                        break
                
                print(f"  [INFO] Using historical dates for backtesting")
        except Exception as e:
            print(f"  [WARN] Could not load historical dates: {e}")
    
    # Calculate structural metrics of generated batch
    all_nums = [n for t in tickets for n in t.main_numbers]
    evens = sum(1 for n in all_nums if n % 2 == 0)
    lows = sum(1 for n in all_nums if n <= 20)
    total_nums = len(all_nums)
    
    print(f"  [METRICS] Structure Analysis:")
    print(f"    Even/Odd Ratio: {evens/total_nums:.2f} (Target 0.50)")
    print(f"    Low/High Ratio: {lows/total_nums:.2f} (Target 0.50)")
    
    # Fill remaining with current date if needed
    while len(prediction_dates) < count:
        prediction_dates.append(datetime.now())
    
    # Convert to DataFrame format
    rows = []
    for i, ticket in enumerate(tickets):
        main_nums = sorted(ticket.main_numbers)
        rows.append({
            'prediction_date': prediction_dates[i].strftime('%Y-%m-%d'),
            'generation_timestamp': generation_time.isoformat(),
            'run_id': run_id,
            'line_id': i + 1,
            'n1': main_nums[0],
            'n2': main_nums[1],
            'n3': main_nums[2],
            'n4': main_nums[3],
            'n5': main_nums[4],
            'n6': main_nums[5],
            'powerball': ticket.powerball
        })
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_path = BASE_DIR / "predictions.csv"
    df.to_csv(output_path, index=False)
    
    print(f"[OK] Generated {count} fresh predictions")
    print(f"     Saved to: {output_path}")
    
    return df


def run_complete_workflow(prediction_count: int = 10):
    """
    Run complete generate-and-audit workflow.
    
    This is the proper way to run the audit:
    1. Generate fresh predictions with timestamps
    2. Run enforced audit (which validates freshness)
    """
    print("\n" + "=" * 70)
    print("RULE-BASED OPTIMIZATION SYSTEM - WORKFLOW")
    print("=" * 70)
    print(f"Workflow started: {datetime.now()}")
    
    # Step 1: Generate fresh predictions
    print("\n" + "-" * 70)
    print("PHASE 1: GENERATING FRESH PREDICTIONS")
    print("-" * 70)
    
    predictions_df = generate_fresh_predictions(count=prediction_count)
    
    # Step 2: Run enforced audit
    print("\n" + "-" * 70)
    print("PHASE 2: RUNNING ENFORCED AUDIT")
    print("-" * 70)
    
    auditor = EnforcedAuditEngine()
    result = auditor.run_enforced_audit()
    
    # Summary
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    
    status = result.get('status', 'UNKNOWN')
    print(f"  Status: {status}")
    
    if status == 'SUCCESS':
        results = result.get('results', {})
        verdict = results.get('step5_verdict', {})
        print(f"  Verdict:    {verdict.get('verdict', 'N/A')}")
        print(f"  Action:     {verdict.get('action', 'N/A')}")
        print(f"  Deviation:  {verdict.get('deviation', 'N/A')}")
        
        # Persist audit metrics to .audit_history.json
        comparison = results.get('step3_comparison', {})
        total = comparison.get('total', 0)
        if total > 0:
            three_plus_rate = comparison.get('3_plus', 0) / total
            four_plus_rate = comparison.get('4_plus', 0) / total
        else:
            three_plus_rate = 0.0
            four_plus_rate = 0.0
        append_audit_history(three_plus_rate, four_plus_rate, prediction_count)
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")
        # Still persist a zero-rate entry so rolling sample grows
        append_audit_history(0.0, 0.0, prediction_count)
    
    return result


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate fresh predictions and run enforced audit"
    )
    
    parser.add_argument(
        '-n', '--count',
        type=int,
        default=10,
        help="Number of predictions to generate (default: 10)"
    )
    
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help="Only generate predictions, don't run audit"
    )
    
    parser.add_argument(
        '--audit-only',
        action='store_true',
        help="Only run audit (predictions must exist)"
    )
    
    args = parser.parse_args()
    
    if args.generate_only:
        generate_fresh_predictions(count=args.count)
    elif args.audit_only:
        auditor = EnforcedAuditEngine()
        auditor.run_enforced_audit()
    else:
        run_complete_workflow(prediction_count=args.count)


if __name__ == "__main__":
    main()

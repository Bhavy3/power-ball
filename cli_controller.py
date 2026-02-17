"""
CLI Controller - Central Entry Point
Upgraded Powerball System (Research-Grade)
"""

import argparse
import sys
import json
from audit_engine import AuditEngine
from seasl_evaluation import run_evaluation
from evaluation_tests import run_all_tests
from system_health_check import SystemDiagnostic
from baseline_comparison import BaselineComparison
from simulation_runner import SimulationRunner
from report_export import ReportExporter
from engine import LotteryEngine as CoreEngine

# Fix for Windows Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(
        description="Powerball Research & Optimization Framework (CLI)",
        epilog="Deterministic. Auditable. Research-Grade."
    )

    # Arguments
    parser.add_argument('--engine', action='store_true', help="Run ticket generation engine")
    parser.add_argument('--test', action='store_true', help="Run internal unit tests (SEASL+)")
    parser.add_argument('--audit', action='store_true', help="Run historical audit")
    parser.add_argument('--evaluate', action='store_true', help="Run SEASL+ evaluation")
    parser.add_argument('--full-report', action='store_true', help="Generate comprehensive system report")
    parser.add_argument('--simulate', type=int, metavar='N', help="Run Monte Carlo simulation with N draws")
    parser.add_argument('--health', action='store_true', help="Run system health diagnostic")
    parser.add_argument('--export-report', action='store_true', help="Export latest report to file")
    
    # Engine args
    parser.add_argument('-n', '--count', type=int, default=1, help="Number of tickets for --engine")

    args = parser.parse_args()

    # Route commands
    if args.engine:
        run_engine(args.count)
    elif args.test:
        run_tests()
    elif args.audit:
        run_audit()
    elif args.evaluate:
        run_evaluate()
    elif args.full_report:
        run_full_report()
    elif args.simulate:
        run_simulation(args.simulate)
    elif args.health:
        run_health()
    elif args.export_report:
        run_export()
    else:
        parser.print_help()

# ==============================================================================
# Command Handlers
# ==============================================================================

def run_engine(count):
    print("\n🚀 STARTING ENGINE...")
    engine = CoreEngine(use_adaptive=True)
    if count == 1:
        ticket = engine.generate_ticket()
        engine.print_ticket(ticket)
    else:
        tickets = engine.generate_tickets(count)
        engine.print_tickets_summary(tickets)

def run_tests():
    print("\n🧪 RUNNING SEASL+ TESTS...")
    success = run_all_tests()
    if not success:
        sys.exit(1)

def run_audit():
    print("\n📋 RUNNING AUDIT ENGINE...")
    engine = AuditEngine()
    engine.print_audit_report()

def run_evaluate():
    print("\n🔬 RUNNING SEASL+ EVALUATION...")
    report = run_evaluation(seed=42)
    print(json.dumps(report, indent=2, default=str))

def run_health():
    print("\n🏥 RUNNING SYSTEM DIAGNOSTIC...")
    diag = SystemDiagnostic()
    diag.run()

def run_simulation(n):
    sim = SimulationRunner()
    res = sim.run_simulation(n_draws=n)
    sim.print_simulation_report(res)

def run_full_report():
    print("\n" + "="*70)
    print("📑 GENERATING FULL SYSTEM REPORT")
    print("="*70)
    
    # 1. Health
    print("\n>>> SYSTEM HEALTH")
    diag = SystemDiagnostic()
    diag.run()
    
    # 2. Audit
    print("\n>>> AUDIT STATUS")
    audit = AuditEngine()
    audit_results = audit.evaluate()
    audit.print_audit_report()
    
    # 3. SEASL+ Evaluation
    print("\n>>> SEASL+ EVALUATION")
    eval_report = run_evaluation(seed=42)
    # Summarize SEASL output
    print(f"  Status: {eval_report.get('file_status')}")
    print(f"  History Source: {eval_report.get('history_source')}")
    
    # 4. Baseline Comparison
    print("\n>>> BASELINE BENCHMARKING")
    bc = BaselineComparison()
    # Extract metrics from audit for comparison
    metrics = audit_results.get("accuracy_distribution", {}).get("3_plus", {})
    # Flatten metric for comparison
    flat_metrics = {"3_plus_pct": metrics.get("pct", 0) / 100.0} # Normalize
    bc.print_comparison_report(flat_metrics)
    
    print("\n✅ FULL REPORT COMPLETE")

def run_export():
    print("\n💾 EXPORTING REPORT...")
    # Gather data
    audit = AuditEngine()
    results = audit.evaluate()
    
    exporter = ReportExporter()
    exporter.export_json(results)
    
    # Flatten a bit for CSV (Accuracy Distribution)
    acc = results.get("accuracy_distribution", {})
    csv_rows = []
    for k, v in acc.items():
        if isinstance(v, dict):
            row = {"metric": k, "count": v.get("count"), "pct": v.get("pct")}
            csv_rows.append(row)
            
    if csv_rows:
        exporter.export_csv(csv_rows)
        
    # Chart
    exporter.try_generate_chart(results)

if __name__ == "__main__":
    main()

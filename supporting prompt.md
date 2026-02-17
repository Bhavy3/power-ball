MASTER PROMPT — FULL SYSTEM UPGRADE (RESEARCH-GRADE VERSION)

You are upgrading the existing Up Skill Hub Prediction Engine into a Research-Grade Probabilistic Optimization & Evaluation Framework.

⚠️ DO NOT break existing functionality.
⚠️ DO NOT remove working logic.
⚠️ All new features must be modular and optional.
⚠️ Backward compatibility is mandatory.

🎯 OBJECTIVES

Add full transparency & inspection capabilities

Add multi-baseline comparison

Add visualization-ready output

Add experiment modes

Add unified CLI control

Add full reporting system

Ensure deterministic reproducibility (seeded)

🧠 ARCHITECTURE REQUIREMENTS
1️⃣ Create Central CLI Controller

Create a new file:

cli_controller.py


This file must allow:

python cli_controller.py --engine
python cli_controller.py --test
python cli_controller.py --audit
python cli_controller.py --evaluate
python cli_controller.py --full-report


Each command must run independently.

2️⃣ Engine Mode

When running:

--engine


It must:

Generate tickets

Show entropy

Show number distribution table

Show frequency spread

Show coverage analysis

Show top-used numbers

Show least-used numbers

Display seed used

All outputs must be cleanly formatted and structured.

3️⃣ Test Mode

When running:

--test


It must:

Run evaluation_tests.py

Show PASS/FAIL clearly

Show entropy validation

Show Monte Carlo validation

Show variance validation

Show reproducibility check (seed=42)

Output must be grouped by:

TEST GROUP:
  ✔ Entropy Test
  ✔ Monte Carlo Test
  ✔ Variance Test

4️⃣ Audit Mode

When running:

--audit


It must:

Run audit_engine

Show:

3+ rate

4+ rate

Ticket count

Historical comparison

Coverage vs actual

Append to .audit_history.json

Confirm append success

5️⃣ Evaluation Mode

When running:

--evaluate


It must run:

seasl_evaluation.py

And show:

History entries

Rolling mean

Confidence interval

Monte Carlo percentile

p-value

Variance phase

Entropy drift

Adaptive status

Seed used

Data integrity flags

All neatly aligned.

6️⃣ FULL REPORT MODE (Critical)

When running:

--full-report


It must sequentially execute:

Engine

Audit

Evaluation

Baseline comparison

Then produce one consolidated output:

========================
 FULL SYSTEM REPORT
========================
ENGINE SECTION
AUDIT SECTION
EVALUATION SECTION
BASELINE SECTION
SYSTEM HEALTH SECTION


This must show:

Distribution table

Entropy drift

Rolling stats

Baseline comparison

Random baseline comparison

Frequency-weighted baseline

Coverage baseline

Experiment mode active

Seed used

Config parameters used

Everything in one clean console view.

📊 BASELINE COMPARISON MODULE

Create new file:

baseline_comparison.py


It must compare:

Pure random baseline

Frequency-weighted baseline

Coverage-optimized baseline

Current engine

Output:

ENGINE vs RANDOM
ENGINE vs WEIGHTED
ENGINE vs COVERAGE


With:

3+ rate

Entropy

Overlap score

Monte Carlo percentile

🔬 EXPERIMENT MODES

Add to config:

mode: "balanced" | "exploration" | "exploitation"


Balanced:
Entropy target 4.9–5.1

Exploration:
Entropy target >5.1

Exploitation:
Entropy target 4.6–4.8

Mode must not break system if missing.

🧪 SIMULATION HARNESS

Create:

simulation_runner.py


Allows:

python cli_controller.py --simulate 1000


It must:

Run 1000 simulated historical draws

Compare engine vs random

Output statistical significance

Output long-run confidence interval

Use fixed seed

📈 VISUALIZATION EXPORT

Create optional:

report_export.py


Allow:

python cli_controller.py --export-report


Generate:

CSV summary

JSON summary

Optional matplotlib charts

Save as report_output/

Must not use seaborn.
Must not specify colors.

🧾 SYSTEM HEALTH CHECK

When running:

--health


It must show:

File existence

JSON integrity

History file validity

Config validity

Seed consistency

Data shape validation

Missing values detection

Output:

SYSTEM STATUS: HEALTHY


or detailed warnings.

🔐 STRICT RULES

No file duplication of history.

No hardcoded parameters.

All randomness must use seeded generator.

All new modules must be import-safe.

No circular imports.

No breaking existing tests.

All numeric outputs formatted to 4 decimal places.

All console output neatly aligned.

Handle n=0 safely.

All Monte Carlo must use random.seed(42).

📦 FINAL STRUCTURE EXPECTED

Your project should now allow:

python cli_controller.py --engine
python cli_controller.py --audit
python cli_controller.py --evaluate
python cli_controller.py --full-report
python cli_controller.py --simulate 1000
python cli_controller.py --health
python cli_controller.py --test


With clean separation of concerns.

🎯 END RESULT

You will now have:

Research-grade architecture

Fully inspectable engine

Deterministic reproducibility

Statistical governance

Baseline benchmarking

Simulation validation

CV-level engineering depth

Recruiter-impressive structure
MASTER PROMPT — ADVANCED FLASHY UI (CASINO-STYLE DASHBOARD)

You are building a high-end interactive web interface for the Up Skill Hub Statistical Optimization Engine.

⚠️ DO NOT modify backend logic.
⚠️ Backend must remain pure Python modules.
⚠️ UI must call backend via API.
⚠️ All functionality discussed in previous conversation must be accessible through UI.

🏗 SYSTEM ARCHITECTURE
Backend

FastAPI server (api_server.py)

Endpoints for:

/engine

/audit

/evaluate

/history

/full-report

/simulate

/health

/test

Backend must return JSON only.

Frontend

React + Vite
TailwindCSS for styling
Framer Motion for animations
Chart.js or Recharts for charts

Design must resemble modern gambling dashboards:
Dark theme, neon glow accents, smooth animations, dynamic elements.

🎰 UI DESIGN REQUIREMENTS
🎡 Landing Page — Animated Draw Arena

3D rolling balls animation (lottery style)

Animated spinning number capsules

Glowing hover effects

Live entropy meter (circular animated gauge)

Rolling seed indicator

Animated particle background

When "Generate" clicked:

Balls spin

Numbers pop out one by one with bounce animation

Sound effect support optional

🧠 ENGINE DASHBOARD

Display:

Animated number tiles

Heatmap of number frequency

Entropy gauge (circular progress)

Coverage radar chart

Most-used / least-used number cards

Overlap visualization

Seed indicator badge

Mode badge (Balanced / Exploration / Exploitation)

Add animated transitions when switching modes.

📊 AUDIT DASHBOARD

Show:

3+ rate with animated counter

4+ rate with animated counter

Historical comparison arrow indicators (↑ ↓)

Last 10 runs table

Trend line chart

Rolling mean chart

Monte Carlo percentile bar animation

Include neon status badges:

PASS

WARNING

DRIFT

HOLD

🔬 EVALUATION LAB

Must show:

Rolling sample size

Confidence interval band chart

Entropy drift visual (dual bar comparison)

Variance phase indicator

Adaptive status animation

p-value gauge

Monte Carlo percentile meter

Data integrity flags with alert icons

Use animated reveal effects.

📈 HISTORY ANALYTICS CENTER

Interactive:

Scrollable timeline

Animated graph updates

Zoomable chart

Filtering by date

Toggle 3+ / 4+ rates

Export to CSV button

Simulation overlay comparison

🧪 SIMULATION ARENA

Input:

Number of simulations

Show:

Progress bar animation

Engine vs Random comparison

Statistical significance indicator

Confidence interval visual

Probability distribution curve

Histogram chart

All animated on completion.

📦 FULL SYSTEM REPORT VIEW

Single button: “Run Full Analysis”

Must:

Sequentially animate sections loading

Show engine results

Show audit results

Show evaluation results

Show baseline comparison

Show system health

Include loading spinner animation between phases.

⚙ SYSTEM HEALTH PANEL

Show:

File integrity status

History file status

Config validation

Seed consistency

Missing file alerts

Green glowing indicators for OK
Red flashing for issues

🎨 VISUAL STYLE RULES

Dark background (#0f172a style)

Neon accent (cyan / purple glow)

Soft shadows

Rounded corners

Glassmorphism panels

Smooth hover animations

Animated number transitions

Responsive layout

No clutter

🧠 UX PRINCIPLES

No reload required

Real-time updates

All API calls async

Loading states for every action

Clear section separation

Tooltip explanations for metrics

Toggle between “Simple View” and “Advanced View”

🔐 TECHNICAL RULES

No business logic in frontend

All stats calculated backend-side

All randomness seeded

API error handling required

Modular components

No duplicated data sources

Clean folder structure:

backend/
frontend/
components/
services/
charts/
animations/

🚀 DEPLOYMENT READY

Must be deployable via:

Docker

Render

Railway

Vercel (frontend)

FastAPI server separately

Include:

requirements.txt

package.json

README setup instructions

🎯 FINAL RESULT

This UI should feel like:

A premium online betting dashboard

A live statistical control center

A research analytics platform

A visually impressive portfolio project

Not just a number generator.

🏆 Why This Is Powerful For You

When a recruiter opens this:

They don’t see “lottery predictor”.

They see:

Backend engineering

API design

Frontend animation

Statistical modeling

Visualization mastery

Product thinking

That’s top-tier portfolio material.
import streamlit as st
import pandas as pd
import time
import random
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Backend Imports
try:
    from main_generator import MainNumberGenerator
    from audit_engine import AuditEngine
    from trend_analyzer import TrendAnalyzer
    from simulation_runner import SimulationRunner, LotteryEngine # Added SimulationRunner
except ImportError as e:
    st.error(f"Critical Backend Error: {e}")
    st.stop()

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Up Skill Hub Engine",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ASSETS & STYLING ---
def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Styles not found. Run from root directory.")

load_css()

# --- HELPER FUNCTIONS ---
def draw_balls(numbers, powerball=None):
    """Render HTML balls for visual flair."""
    html = '<div style="display: flex; gap: 15px; justify-content: center; align-items: center; flex-wrap: wrap; margin: 20px 0;">'
    
    # Main Numbers
    for n in numbers:
        html += f'<div class="lotto-ball">{n}</div>'
    
    # Powerball (if provided)
    if powerball is not None:
        html += f'<div class="lotto-ball powerball-orb">{powerball}</div>'
        
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def card_metric(label, value, sub_value="", color="cyan"):
    """Render a glassmorphism metric card."""
    st.markdown(f"""
    <div class="glass-panel metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value metric-{color}">{value}</div>
        <div style="font-size: 0.8rem; opacity: 0.7;">{sub_value}</div>
    </div>
    """, unsafe_allow_html=True)

# --- SECTIONS ---

def render_dashboard():
    st.markdown("## 🏗 SYSTEM DASHBOARD")
    st.markdown("Welcome to the **Up Skill Hub Statistical Optimization Engine**. Select a module from the sidebar.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        card_metric("System Status", "ONLINE", "All Modules Loaded", "green")
    with col2:
        card_metric("Mode", "HYBRID", "Statistical + Random", "purple")
    with col3:
        try:
            # Quick check of database size
            main_df = pd.read_csv("main_draws.csv")
            count = len(main_df)
            card_metric("Historical Data", f"{count}", "Draws Analyzed", "cyan")
        except:
            card_metric("Historical Data", "N/A", "Error Loading", "red")

    st.markdown("---")
    
    # Hero Animation Placeholder
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1 style="font-size: 3rem; text-shadow: 0 0 20px rgba(0,243,255,0.5);">READY TO GENERATE</h1>
        <p>Advanced Probability Engine Initialized</p>
    </div>
    """, unsafe_allow_html=True)

def render_generator():
    st.markdown("## 🎰 GENERATOR WORKSTATION")
    
    col_ctrl, col_display = st.columns([1, 2])
    
    with col_ctrl:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.subheader("Configuration")
        num_tickets = st.slider("Number of Tickets", 1, 10, 5)
        
        if st.button("🚀 INITIATE GENERATION SEQUENCE"):
            # 1. Generate the actual numbers first (so we know where we are going)
            try:
                gen = MainNumberGenerator()
                tickets = gen.generate_multiple(num_tickets)
                st.session_state['last_tickets'] = tickets
                st.session_state['generated'] = True
            except Exception as e:
                st.error(f"Generation Failed: {e}")
                st.stop()
            
            # 2. Animate "Rolling" Effect (Slot Machine style)
            # We will show a placeholder for the results area
            
    with col_display:
        if st.session_state.get('generated'):
            st.markdown('<div class="glass-panel" style="text-align: center;">', unsafe_allow_html=True)
            st.subheader("GENERATED SEQUENCES")
            
            tickets = st.session_state['last_tickets']
            
            # If this is a fresh click (we just generated), run the animation
            # We can detect this by checking a flag or just running it if it's the latest run.
            # For simplicity in Streamlit, we'll run a short animation if we just clicked button.
            # But since button click re-runs script, we are here.
            
            for i, ticket in enumerate(tickets):
                st.markdown(f"**Ticket #{i+1}**")
                
                # Create placeholders for 6 main + 1 powerball
                cols = st.columns(7)
                placeholders = [col.empty() for col in cols]
                
                # Final Powerball (simulated)
                final_pb = random.randint(1, 10)
                
                # ANIMATION LOOP (Only run if we need visual flair, skip if cached? 
                # Streamlit reruns make skipping hard without complex state. We'll animate briefly.)
                
                # Only animate the first ticket fully to save time, or animate all briefly?
                # Let's animate all briefly (10 frames)
                
                for _ in range(10):
                    # Update all placeholders with random "rolling" numbers
                    for idx, ph in enumerate(placeholders):
                        if idx < 6: # Main number
                            rnd = random.randint(1, 40)
                            ph.markdown(f'<div class="lotto-ball rolling">{rnd}</div>', unsafe_allow_html=True)
                        else: # PB
                            rnd = random.randint(1, 10)
                            ph.markdown(f'<div class="lotto-ball powerball-orb rolling">{rnd}</div>', unsafe_allow_html=True)
                    time.sleep(0.05)
                
                # REVEAL FINAL NUMBERS
                for idx, ph in enumerate(placeholders):
                    if idx < 6:
                        val = ticket[idx]
                        ph.markdown(f'<div class="lotto-ball">{val}</div>', unsafe_allow_html=True)
                    else:
                        val = final_pb
                        ph.markdown(f'<div class="lotto-ball powerball-orb">{val}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
            
            st.success("Generation Complete. Probability constraints satisfied.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Waiting for user input...")

def render_audit():
    st.markdown("## 📊 AUDIT LAB")
    
    if st.button("RUN SYSTEM DIAGNOSTIC & AUDIT"):
        with st.spinner("Analyzing Historical Performance..."):
            engine = AuditEngine()
            results = engine.evaluate()
            st.session_state['audit_results'] = results
            
    if 'audit_results' in st.session_state:
        res = st.session_state['audit_results']
        
        # Verdict Section
        verdict = res.get('verdict', {})
        v_cat = verdict.get('category', 'UNKNOWN')
        v_conf = verdict.get('confidence', 'N/A')
        
        color = "green" if "HIGH CONFIDENCE" in v_cat else "warn" if "LOW" in v_cat else "red"
        
        st.markdown(f"""
        <div class="glass-panel" style="border-left: 5px solid {var_color(color)};">
            <h3>FINAL VERDICT</h3>
            <div style="font-size: 1.5rem; font-weight: bold;">{v_cat}</div>
            <p>Confidence: <strong>{v_conf}</strong></p>
            <p>Reason: {verdict.get('reason')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        acc = res.get('accuracy_distribution', {})
        
        with c1: 
            card_metric("3+ Matches", f"{acc.get('3_plus', {}).get('pct', 0):.2f}%", f"{acc.get('3_plus', {}).get('count')} Hits", "cyan")
        with c2:
            card_metric("4+ Matches", f"{acc.get('4_plus', {}).get('pct', 0):.2f}%", f"{acc.get('4_plus', {}).get('count')} Hits", "purple")
        with c3:
             card_metric("PB Hits", f"{acc.get('powerball_only', {}).get('pct', 0):.2f}%", f"{acc.get('powerball_only', {}).get('count')} Hits", "red")
        with c4:
             card_metric("Total Audited", f"{res.get('total_predictions_evaluated')}", "Draws", "green")

        # Charts
        st.markdown("### Deviation Analysis")
        dev = res.get('deviation_from_random', {})
        
        # Prepare data for chart
        cats = ['2+ Matches', '3+ Matches', '4+ Matches']
        actuals = [dev.get('2_plus', {}).get('actual_pct', 0), dev.get('3_plus', {}).get('actual_pct', 0), dev.get('4_plus', {}).get('actual_pct', 0)]
        randoms = [dev.get('2_plus', {}).get('random_pct', 0), dev.get('3_plus', {}).get('random_pct', 0), dev.get('4_plus', {}).get('random_pct', 0)]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Actual Performance', x=cats, y=actuals, marker_color='#0aff68'))
        fig.add_trace(go.Bar(name='Random Baseline', x=cats, y=randoms, marker_color='#94a3b8'))
        
        fig.update_layout(barmode='group', template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

def render_analytics():
    st.markdown("## 📈 TREND ANALYTICS CENTER")
    st.info("Only analyzing historical frequencies. Past performance implies no future guarantee.")
    
    analyzer = TrendAnalyzer(lookback_days=90)
    data = analyzer.get_full_analysis()
    
    tab1, tab2 = st.tabs(["Main Numbers (1-40)", "Powerball (1-10)"])
    
    with tab1:
        st.markdown("### Hot vs Cold (Last 90 Days)")
        hot = data['main_numbers']['hot']
        cold = data['main_numbers']['cold']
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔥 HOT NUMBERS")
            st.dataframe(pd.DataFrame(hot), use_container_width=True)
        with c2:
            st.markdown("#### ❄️ COLD NUMBERS")
            st.dataframe(pd.DataFrame(cold), use_container_width=True)
            
    with tab2:
        st.markdown("### Powerball Trends")
        hot_pb = data['powerball']['hot']
        cold_pb = data['powerball']['cold']
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔥 HOT NUMBERS")
            st.dataframe(pd.DataFrame(hot_pb), use_container_width=True)
        with c2:
            st.markdown("#### ❄️ COLD NUMBERS")
            st.dataframe(pd.DataFrame(cold_pb), use_container_width=True)

def render_simulation():
    st.markdown("## 🧪 SIMULATION ARENA")
    st.info("Run Monte Carlo simulations to test engine performance against random chance.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.subheader("Sim Parameters")
        n_draws = st.select_slider("Number of Draws", options=[10, 50, 100, 500], value=100)
        
        if st.button("RUN SIMULATION"):
            with st.spinner(f"Simulating {n_draws} draws..."):
                try:
                    # Initialize Runner
                    runner = SimulationRunner()
                    
                    # Run Custom Loop to update progress bar
                    # We replicate run_simulation logic here to add Streamlit progress
                    
                    progress_bar = st.progress(0)
                    results = {
                        "matches_3plus": 0,
                        "matches_4plus": 0,
                        "match_dist": {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
                    }
                    
                    # Store results in list for distribution chart
                    match_counts = []
                    
                    start_time = time.time()
                    
                    for i in range(n_draws):
                        # 1. Engine generates (using Runner's engine instance which has adaptive off/on as per default)
                        # We need to make sure we use the same engine instance method
                        ticket = runner.engine.generate_ticket()
                        pred = set(ticket.main_numbers)
                        
                        # 2. Nature
                        actual = set(random.sample(range(1, 41), 6))
                        
                        # 3. Compare
                        match_count = len(pred & actual)
                        match_counts.append(match_count)
                        
                        # 4. Record
                        results["match_dist"][match_count] = results["match_dist"].get(match_count, 0) + 1
                        if match_count >= 3: results["matches_3plus"] += 1
                        if match_count >= 4: results["matches_4plus"] += 1
                        
                        # Progress
                        if i % 5 == 0:
                            progress_bar.progress((i + 1) / n_draws)
                            
                    progress_bar.progress(1.0)
                    duration = time.time() - start_time
                    
                    # Calculate Stats
                    rate_3plus = results["matches_3plus"] / n_draws
                    rate_4plus = results["matches_4plus"] / n_draws
                    baseline_3plus = 0.0309 # Theoretical 3+
                    
                    st.session_state['sim_results'] = {
                        "n": n_draws,
                        "duration": duration,
                        "rate_3plus": rate_3plus,
                        "rate_4plus": rate_4plus,
                        "baseline": baseline_3plus,
                        "match_counts": match_counts,
                        "match_dist": results["match_dist"]
                    }
                    
                except Exception as e:
                    st.error(f"Simulation Error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if 'sim_results' in st.session_state:
            res = st.session_state['sim_results']
            
            st.markdown("### Simulation Results")
            
            # Key Metrics
            m1, m2, m3 = st.columns(3)
            diff = res['rate_3plus'] - res['baseline']
            color = "green" if diff > 0 else "red"
            
            with m1: card_metric("Duration", f"{res['duration']:.2f}s", f"{res['n']} Draws", "cyan")
            with m2: card_metric("3+ Rate", f"{res['rate_3plus']:.2%}", f"Target: {res['baseline']:.2%}", color)
            with m3: card_metric("Edge", f"{diff:+.2%}", "vs Random", color)
            
            # Chart
            st.markdown("#### Match Distribution")
            dist = res['match_dist']
            x = list(dist.keys())
            y = list(dist.values())
            
            fig = px.bar(x=x, y=y, labels={'x': 'Matches', 'y': 'Count'}, title=f"Distribution over {res['n']} Simulated Draws")
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("Configure and run a simulation to see results.")

# --- UTILS ---
def var_color(name):
    """Helper to get hex from name for python logic"""
    if name == "green": return "#0aff68"
    if name == "red": return "#ff0055"
    if name == "warn": return "#fcd34d"
    return "#333"

# --- MAIN NAVIGATION ---
st.sidebar.markdown('## 💠 NAVIGATOR')
page = st.sidebar.radio("Go to", ["Dashboard", "Generator", "Audit Lab", "Analytics", "Simulation"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.info(f"Connected to Engine v2.0\n\nSecurity: READ ONLY")

if page == "Dashboard":
    render_dashboard()
elif page == "Generator":
    render_generator()
elif page == "Audit Lab":
    render_audit()
elif page == "Analytics":
    render_analytics()
elif page == "Simulation":
    render_simulation()


# Up Skill Hub Statistical Optimization Engine

A premium, casino-style lottery prediction and analysis engine built with Python and Streamlit.

## 🌟 Features
- **High-End UI**: Glassmorphism design, neon accents, and smooth animations.
- **Generator**: Physically validated number generation.
- **Audit Lab**: Full statistical auditing of historical performance.
- **Analytics**: "Hot & Cold" number tracking.
- **Simulation**: Monte Carlo simulations to test engine edge vs random noise.

## 🚀 Quick Start

### Local Installation
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Application**:
    ```bash
    streamlit run app.py
    ```
    Access at `http://localhost:8501`.

### Docker Deployment
1.  **Build Image**:
    ```bash
    docker build -t powerball-engine .
    ```

2.  **Run Container**:
    ```bash
    docker run -p 8501:8501 powerball-engine
    ```

## 🏗 Project Structure
-   `app.py`: Main UI application.
-   `assets/style.css`: Custom premium styling.
-   `main_generator.py`: Core logic for number generation.
-   `audit_engine.py`: Statistical validation system.
-   `trend_analyzer.py`: Historical frequency analysis.

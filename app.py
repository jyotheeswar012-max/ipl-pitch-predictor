import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import train_model, predict_best_players
from data_generator import generate_ipl_dataset

st.set_page_config(
    page_title="IPL Pitch Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;
    }
    .main-header h1 { color: #e94560; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #a8dadc; font-size: 1.1rem; margin: 5px 0 0; }
    .player-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #e94560; border-radius: 10px; padding: 15px; margin: 8px 0;
    }
    .player-rank  { color: #ffd700; font-size: 1.4rem; font-weight: bold; }
    .player-name  { color: #ffffff; font-size: 1.1rem; }
    .player-score { color: #00d4aa; font-size: 1rem; }
    .metric-box {
        background: #1e1e2e; border-left: 4px solid #e94560;
        padding: 12px; border-radius: 6px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='main-header'>
    <h1>🏏 IPL Pitch Predictor</h1>
    <p>ML-powered player recommendations based on pitch conditions &amp; match context</p>
</div>
""", unsafe_allow_html=True)

# ── Data & Model (cache safely) ──────────────────────────────
@st.cache_data
def load_data():
    return generate_ipl_dataset()

@st.cache_data
def load_model_cached():
    """Train model independently so cache hash is stable."""
    _df = generate_ipl_dataset()
    return train_model(_df)

df = load_data()
model, label_encoders, feature_cols = load_model_cached()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("⚙️ Match Conditions")

PITCH_TYPES = ["Flat/Batting", "Hard/True Bounce", "Green/Grassy", "Slow/Dry Spin", "Wet/Dew Heavy"]
VENUES = [
    "Wankhede Stadium, Mumbai",
    "M Chinnaswamy Stadium, Bengaluru",
    "Eden Gardens, Kolkata",
    "Narendra Modi Stadium, Ahmedabad",
    "Arun Jaitley Stadium, Delhi",
    "MA Chidambaram Stadium, Chennai",
    "Rajiv Gandhi Stadium, Hyderabad",
    "Sawai Mansingh Stadium, Jaipur",
]
ROLES = ["Batsman", "Bowler", "All-Rounder", "Wicketkeeper-Batter"]

pitch_type  = st.sidebar.selectbox("🏙️ Pitch Type", PITCH_TYPES)
venue       = st.sidebar.selectbox("📍 Venue", VENUES)
toss        = st.sidebar.radio("🪙 Toss Decision", ["Bat First", "Chase"])
match_phase = st.sidebar.multiselect("⏱️ Match Phase Focus",
                ["Powerplay", "Middle Overs", "Death Overs"],
                default=["Powerplay", "Death Overs"])
top_n       = st.sidebar.slider("🔝 Show Top N Players", 5, 20, 10)
role_filter = st.sidebar.multiselect("🎯 Player Role Filter", ROLES, default=ROLES)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Pitch Intelligence")
pitch_info = {
    "Flat/Batting":     {"color": "#00d4aa", "icon": "🟢", "tip": "Favors top-order power hitters."},
    "Hard/True Bounce": {"color": "#ffd700", "icon": "🟡", "tip": "Benefits aggressive batters & pace bowlers."},
    "Green/Grassy":     {"color": "#4caf50", "icon": "🌿", "tip": "Seam & swing bowlers thrive."},
    "Slow/Dry Spin":    {"color": "#ff7043", "icon": "🟠", "tip": "Spinners dominate. Anchor batters build totals."},
    "Wet/Dew Heavy":    {"color": "#2196f3", "icon": "💧", "tip": "Chasing teams benefit. Yorkers crucial at death."},
}
pi = pitch_info[pitch_type]
st.sidebar.markdown(f"""
<div style='background:#1e1e2e;padding:10px;border-radius:8px;border-left:4px solid {pi['color']}'>
<b style='color:{pi['color']}'>{pi['icon']} {pitch_type}</b><br>
<span style='color:#ccc;font-size:0.85rem'>{pi['tip']}</span>
</div>
""", unsafe_allow_html=True)

# ── Predictions ──────────────────────────────────────────────
predictions = predict_best_players(
    df, model, label_encoders, feature_cols,
    pitch_type=pitch_type, venue=venue,
    toss=toss, top_n=top_n, role_filter=role_filter
)

tab1, tab2, tab3, tab4 = st.tabs(["🏆 Top Players", "📊 Analytics", "🗺️ Venue Stats", "ℹ️ How It Works"])

# ── Tab 1: Top Players ───────────────────────────────────────
with tab1:
    st.markdown(f"### 🏆 Best Players for **{pitch_type}** at {venue}")
    st.markdown(f"*Toss: {toss} | Phases: {', '.join(match_phase) if match_phase else 'All'}*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-box'><b style='color:#e94560;font-size:1.5rem'>{len(predictions)}</b><br><span style='color:#aaa'>Players Ranked</span></div>", unsafe_allow_html=True)
    with col2:
        avg_score = predictions['suitability_score'].mean() if not predictions.empty else 0
        st.markdown(f"<div class='metric-box'><b style='color:#00d4aa;font-size:1.5rem'>{avg_score:.1f}</b><br><span style='color:#aaa'>Avg Score</span></div>", unsafe_allow_html=True)
    with col3:
        top_player = predictions.iloc[0]['player_name'] if not predictions.empty else 'N/A'
        st.markdown(f"<div class='metric-box'><b style='color:#ffd700;font-size:1rem'>{top_player}</b><br><span style='color:#aaa'>Top Pick</span></div>", unsafe_allow_html=True)
    with col4:
        top_role = predictions.iloc[0]['role'] if not predictions.empty else 'N/A'
        st.markdown(f"<div class='metric-box'><b style='color:#a78bfa;font-size:1rem'>{top_role}</b><br><span style='color:#aaa'>Top Role</span></div>", unsafe_allow_html=True)

    st.markdown("---")

    for i, row in predictions.iterrows():
        rank = i + 1
        score_bar = "█" * int(row['suitability_score'] / 10) + "░" * (10 - int(row['suitability_score'] / 10))
        reason = row.get('reason', 'Strong historical performance on this surface.')
        ds_tag = row.get('data_source', 'real_ipl')
        ds_color = {'real_ipl': '#00d4aa', 'estimated_domestic': '#ffd700', 'role_based_fallback': '#ff7043'}.get(ds_tag, '#aaa')
        ds_label = {'real_ipl': '✅ Real IPL', 'estimated_domestic': '📊 Estimated', 'role_based_fallback': '📌 Fallback'}.get(ds_tag, ds_tag)
        st.markdown(f"""
<div class='player-card'>
  <span class='player-rank'>#{rank}</span>
  <span class='player-name' style='margin-left:10px;font-size:1.15rem;font-weight:bold'>{row['player_name']}</span>
  <span style='background:#e94560;color:#fff;padding:2px 8px;border-radius:12px;font-size:0.8rem;margin-left:8px'>{row['role']}</span>
  <span style='background:#0f3460;color:#a8dadc;padding:2px 8px;border-radius:12px;font-size:0.8rem;margin-left:4px'>{row['team']}</span>
  <span style='background:{ds_color}22;color:{ds_color};padding:2px 8px;border-radius:12px;font-size:0.75rem;margin-left:4px'>{ds_label}</span>
  <br><br>
  <span class='player-score'>🎯 Suitability Score: <b>{row['suitability_score']:.1f}/100</b></span>
  &nbsp;|&nbsp; Avg: <b>{row['batting_avg']:.1f}</b>
  &nbsp;|&nbsp; SR: <b>{row['strike_rate']:.1f}</b>
  &nbsp;|&nbsp; Economy: <b>{row['economy']:.2f}</b>
  <br>
  <span style='color:#888;font-family:monospace'>{score_bar}</span>
  <br><span style='color:#a8dadc;font-size:0.88rem'>💡 {reason}</span>
</div>
""", unsafe_allow_html=True)

# ── Tab 2: Analytics ─────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Player Suitability Analytics")
    if not predictions.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(
                predictions, x='suitability_score', y='player_name',
                orientation='h', color='suitability_score',
                color_continuous_scale='RdYlGn',
                title=f'Suitability Scores — {pitch_type}',
                labels={'suitability_score': 'Score', 'player_name': 'Player'},
            )
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450, template='plotly_dark')
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            fig_scatter = px.scatter(
                predictions, x='batting_avg', y='strike_rate',
                size='suitability_score', color='role',
                hover_name='player_name',
                title='Avg vs Strike Rate (bubble = suitability)',
                template='plotly_dark',
            )
            fig_scatter.update_layout(height=450)
            st.plotly_chart(fig_scatter, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            role_counts = predictions['role'].value_counts().reset_index()
            role_counts.columns = ['Role', 'Count']
            fig_pie = px.pie(role_counts, names='Role', values='Count',
                             title='Role Distribution in Top Picks',
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             template='plotly_dark')
            st.plotly_chart(fig_pie, use_container_width=True)
        with col4:
            fig_radar = go.Figure()
            for _, row in predictions.head(5).iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[
                        min(row['batting_avg'] / 60 * 100, 100),
                        min(row['strike_rate'] / 200 * 100, 100),
                        max(0, (1 - row['economy'] / 12) * 100),
                        row['suitability_score'],
                        min(row['wickets_per_match'] * 50, 100)
                    ],
                    theta=['Batting Avg', 'Strike Rate', 'Economy', 'Suitability', 'Wickets/Match'],
                    fill='toself', name=row['player_name']
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True, template='plotly_dark', height=400,
                title='Top 5 Player Radar'
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# ── Tab 3: Venue Stats ───────────────────────────────────────
with tab3:
    st.markdown("### 🗺️ Venue Performance Analysis")
    venue_df = df[df['venue'] == venue]
    if not venue_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            avg_score_venue = venue_df.groupby('pitch_type')['match_score'].mean().reset_index()
            fig_v = px.bar(avg_score_venue, x='pitch_type', y='match_score',
                           title=f'Avg Match Score by Pitch Type at {venue.split(",")[0]}',
                           template='plotly_dark', color='match_score',
                           color_continuous_scale='Blues')
            st.plotly_chart(fig_v, use_container_width=True)
        with col2:
            role_perf = venue_df.groupby('role')['suitability_score'].mean().reset_index()
            fig_r = px.bar(role_perf, x='role', y='suitability_score',
                           title=f'Avg Suitability by Role at {venue.split(",")[0]}',
                           template='plotly_dark', color='suitability_score',
                           color_continuous_scale='Reds')
            st.plotly_chart(fig_r, use_container_width=True)

        show_cols = ['player_name', 'role', 'team', 'data_source', 'pitch_type',
                     'batting_avg', 'strike_rate', 'economy', 'suitability_score']
        # Only include data_source column if it exists
        show_cols = [c for c in show_cols if c in venue_df.columns]
        st.dataframe(
            venue_df[show_cols]
            .sort_values('suitability_score', ascending=False)
            .head(20)
            .style.background_gradient(subset=['suitability_score'], cmap='YlOrRd'),
            use_container_width=True
        )

# ── Tab 4: How It Works ──────────────────────────────────────
with tab4:
    st.markdown("### ℹ️ How the ML Model Works")
    st.markdown("""
    #### 🧠 Model Architecture
    This app uses a **Random Forest Regressor** trained on IPL 2025/2026 data enriched with:
    - Real batting avg, strike rate, economy, wickets per match
    - Pitch type encoding (5 types)
    - Venue encoding (8 venues)
    - Toss decision (bat/chase)

    #### 📐 Feature Engineering
    | Feature | Description |
    |---|---|
    | `batting_avg` | Historical average runs per innings |
    | `strike_rate` | Runs per 100 balls |
    | `economy` | Runs conceded per over |
    | `wickets_per_match` | Average wickets taken per match |
    | `pitch_encoded` | Label-encoded pitch type |
    | `venue_encoded` | Label-encoded venue |
    | `toss_encoded` | Bat first = 1, Chase = 0 |
    | `suitability_score` | Target variable (0–100) |

    #### 🎯 Data Source Tags
    | Tag | Meaning |
    |---|---|
    | ✅ Real IPL | Verified IPL 2025/2026 season stats |
    | 📊 Estimated | Domestic T20 / Ranji-based estimates |
    | 📌 Fallback | New/uncapped player — role-average profile |

    #### 📦 Coverage
    All **10 IPL 2026 teams** | **200+ players** | **~64,000 records**
    """)

st.markdown("---")
st.markdown("<center><span style='color:#555'>Built by Jyotheeswar Reddy | IPL Pitch Predictor v2.0 | Powered by RandomForest + Streamlit</span></center>", unsafe_allow_html=True)

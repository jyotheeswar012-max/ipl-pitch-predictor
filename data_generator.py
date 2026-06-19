import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# REAL IPL STATS (IPL 2025 + IPL 2026 seasons combined)
# Sources: Cricbuzz, NDTV Sports, ESPN Cricinfo, Business Standard
# batting_avg, strike_rate = batting stats
# economy = bowling economy (bowlers/all-rounders), else career T20 avg
# wickets_per_match = avg wickets per IPL match
# ─────────────────────────────────────────────────────────────────────────────
REAL_PLAYER_STATS = {
    # ── DOMESTIC INDIAN BATSMEN ─────────────────────────────────────────────
    "Vaibhav Sooryavanshi":  {"batting_avg": 45.33, "strike_rate": 237.30, "economy": 9.80, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "RR"},
    "Sai Sudharsan":         {"batting_avg": 46.57, "strike_rate": 157.86, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "GT"},
    "Shubman Gill":          {"batting_avg": 44.14, "strike_rate": 159.27, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "GT"},
    "Virat Kohli":           {"batting_avg": 55.42, "strike_rate": 148.00, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "RCB"},
    "Yashasvi Jaiswal":      {"batting_avg": 43.00, "strike_rate": 159.71, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "RR"},
    "Rohit Sharma":          {"batting_avg": 30.20, "strike_rate": 139.00, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "MI"},
    "Shreyas Iyer":          {"batting_avg": 50.33, "strike_rate": 175.07, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "PBKS"},
    "Suryakumar Yadav":      {"batting_avg": 65.18, "strike_rate": 167.92, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "MI"},
    "Abhishek Sharma":       {"batting_avg": 33.77, "strike_rate": 193.39, "economy": 8.50, "wickets_per_match": 0.3,  "role": "All-Rounder",         "team": "SRH"},
    "Tilak Varma":           {"batting_avg": 34.50, "strike_rate": 158.00, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "MI"},
    "Rinku Singh":           {"batting_avg": 59.00, "strike_rate": 149.50, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "KKR"},
    "Ruturaj Gaikwad":       {"batting_avg": 35.00, "strike_rate": 140.00, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "CSK"},
    "Priyansh Arya":         {"batting_avg": 27.94, "strike_rate": 211.62, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "PBKS"},
    "Prabhsimran Singh":     {"batting_avg": 32.29, "strike_rate": 160.53, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "PBKS"},
    "Rajat Patidar":         {"batting_avg": 35.00, "strike_rate": 192.69, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "RCB"},
    "Dhruv Jurel":           {"batting_avg": 36.07, "strike_rate": 155.00, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Wicketkeeper-Batter", "team": "RR"},
    "Sanju Samson":          {"batting_avg": 34.13, "strike_rate": 151.00, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Wicketkeeper-Batter", "team": "CSK"},
    "Ishan Kishan":          {"batting_avg": 40.13, "strike_rate": 182.42, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Wicketkeeper-Batter", "team": "SRH"},
    "Devdutt Padikkal":      {"batting_avg": 32.86, "strike_rate": 148.00, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "RCB"},
    "Naman Dhir":            {"batting_avg": 25.20, "strike_rate": 182.60, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "MI"},
    "Jitesh Sharma":         {"batting_avg": 29.00, "strike_rate": 176.35, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Wicketkeeper-Batter", "team": "RCB"},
    "Shahrukh Khan":         {"batting_avg": 22.38, "strike_rate": 179.00, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "GT"},
    "Shashank Singh":        {"batting_avg": 26.14, "strike_rate": 188.57, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "PBKS"},
    "Angkrish Raghuvanshi":  {"batting_avg": 28.13, "strike_rate": 162.00, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "KKR"},
    "Ayush Mhatre":          {"batting_avg": 34.29, "strike_rate": 188.97, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "CSK"},

    # ── DOMESTIC INDIAN ALL-ROUNDERS ─────────────────────────────────────────
    "Ravindra Jadeja":       {"batting_avg": 66.50, "strike_rate": 152.00, "economy": 7.80, "wickets_per_match": 1.2,  "role": "All-Rounder",         "team": "RR"},
    "Hardik Pandya":         {"batting_avg": 28.00, "strike_rate": 148.00, "economy": 9.20, "wickets_per_match": 0.9,  "role": "All-Rounder",         "team": "MI"},
    "Axar Patel":            {"batting_avg": 24.00, "strike_rate": 153.00, "economy": 8.10, "wickets_per_match": 1.1,  "role": "All-Rounder",         "team": "DC"},
    "Washington Sundar":     {"batting_avg": 22.00, "strike_rate": 140.00, "economy": 7.90, "wickets_per_match": 1.0,  "role": "All-Rounder",         "team": "SRH"},
    "Shardul Thakur":        {"batting_avg": 19.00, "strike_rate": 152.00, "economy": 9.60, "wickets_per_match": 1.0,  "role": "All-Rounder",         "team": "KKR"},
    "Shivam Dube":           {"batting_avg": 32.00, "strike_rate": 162.00, "economy": 9.80, "wickets_per_match": 0.4,  "role": "All-Rounder",         "team": "CSK"},
    "Venkatesh Iyer":        {"batting_avg": 34.83, "strike_rate": 186.60, "economy": 9.50, "wickets_per_match": 0.3,  "role": "All-Rounder",         "team": "RCB"},
    "Krunal Pandya":         {"batting_avg": 23.00, "strike_rate": 140.00, "economy": 8.20, "wickets_per_match": 1.1,  "role": "All-Rounder",         "team": "LSG"},
    "Vipraj Nigam":          {"batting_avg": 20.29, "strike_rate": 179.74, "economy": 8.40, "wickets_per_match": 0.8,  "role": "All-Rounder",         "team": "DC"},
    "Ashutosh Sharma":       {"batting_avg": 24.43, "strike_rate": 181.91, "economy": 9.20, "wickets_per_match": 0.3,  "role": "All-Rounder",         "team": "DC"},

    # ── DOMESTIC INDIAN BOWLERS ──────────────────────────────────────────────
    "Jasprit Bumrah":        {"batting_avg": 8.00,  "strike_rate": 95.00,  "economy": 6.67, "wickets_per_match": 1.5,  "role": "Bowler",              "team": "MI"},
    "Arshdeep Singh":        {"batting_avg": 6.00,  "strike_rate": 80.00,  "economy": 8.88, "wickets_per_match": 1.3,  "role": "Bowler",              "team": "PBKS"},
    "Yuzvendra Chahal":      {"batting_avg": 5.00,  "strike_rate": 75.00,  "economy": 7.91, "wickets_per_match": 1.4,  "role": "Bowler",              "team": "PBKS"},
    "Kuldeep Yadav":         {"batting_avg": 6.00,  "strike_rate": 80.00,  "economy": 8.10, "wickets_per_match": 1.3,  "role": "Bowler",              "team": "DC"},
    "Varun Chakravarthy":    {"batting_avg": 5.00,  "strike_rate": 70.00,  "economy": 7.80, "wickets_per_match": 1.2,  "role": "Bowler",              "team": "KKR"},
    "Mohammed Shami":        {"batting_avg": 7.00,  "strike_rate": 90.00,  "economy": 8.50, "wickets_per_match": 1.2,  "role": "Bowler",              "team": "SRH"},
    "Bhuvneshwar Kumar":     {"batting_avg": 8.00,  "strike_rate": 88.00,  "economy": 7.90, "wickets_per_match": 1.1,  "role": "Bowler",              "team": "SRH"},
    "Prasidh Krishna":       {"batting_avg": 5.00,  "strike_rate": 72.00,  "economy": 9.10, "wickets_per_match": 1.6,  "role": "Bowler",              "team": "RR"},
    "Sai Kishore":           {"batting_avg": 6.00,  "strike_rate": 78.00,  "economy": 7.50, "wickets_per_match": 1.2,  "role": "Bowler",              "team": "GT"},
    "Vaibhav Arora":         {"batting_avg": 5.00,  "strike_rate": 70.00,  "economy": 9.20, "wickets_per_match": 1.1,  "role": "Bowler",              "team": "KKR"},
    "Deepak Chahar":         {"batting_avg": 10.00, "strike_rate": 95.00,  "economy": 8.20, "wickets_per_match": 1.1,  "role": "Bowler",              "team": "MI"},

    # ── INTERNATIONAL PLAYERS ────────────────────────────────────────────────
    "KL Rahul":              {"batting_avg": 45.62, "strike_rate": 174.41, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Wicketkeeper-Batter", "team": "DC"},
    "Jos Buttler":           {"batting_avg": 39.92, "strike_rate": 163.03, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Wicketkeeper-Batter", "team": "GT"},
    "Heinrich Klaasen":      {"batting_avg": 50.50, "strike_rate": 159.47, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Wicketkeeper-Batter", "team": "SRH"},
    "Nicholas Pooran":       {"batting_avg": 43.67, "strike_rate": 196.25, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Wicketkeeper-Batter", "team": "LSG"},
    "Rashid Khan":           {"batting_avg": 18.00, "strike_rate": 140.00, "economy": 7.09, "wickets_per_match": 1.5,  "role": "Bowler",              "team": "GT"},
    "Mitchell Marsh":        {"batting_avg": 48.23, "strike_rate": 163.71, "economy": 9.30, "wickets_per_match": 0.5,  "role": "All-Rounder",         "team": "LSG"},
    "Sunil Narine":          {"batting_avg": 27.00, "strike_rate": 168.00, "economy": 7.60, "wickets_per_match": 1.2,  "role": "All-Rounder",         "team": "KKR"},
    "Trent Boult":           {"batting_avg": 5.00,  "strike_rate": 75.00,  "economy": 8.30, "wickets_per_match": 1.3,  "role": "Bowler",              "team": "RR"},
    "Tim David":             {"batting_avg": 62.33, "strike_rate": 188.27, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "RCB"},
    "Ryan Rickelton":        {"batting_avg": 40.72, "strike_rate": 186.66, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Wicketkeeper-Batter", "team": "MI"},
    "Glenn Maxwell":         {"batting_avg": 25.00, "strike_rate": 155.00, "economy": 8.70, "wickets_per_match": 0.5,  "role": "All-Rounder",         "team": "RCB"},
    "Marcus Stoinis":        {"batting_avg": 22.86, "strike_rate": 186.04, "economy": 9.50, "wickets_per_match": 0.5,  "role": "All-Rounder",         "team": "LSG"},
    "Noor Ahmad":            {"batting_avg": 5.00,  "strike_rate": 70.00,  "economy": 8.20, "wickets_per_match": 1.5,  "role": "Bowler",              "team": "GT"},
    "Finn Allen":            {"batting_avg": 31.73, "strike_rate": 214.11, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "KKR"},
    "Josh Hazlewood":        {"batting_avg": 5.00,  "strike_rate": 68.00,  "economy": 8.10, "wickets_per_match": 1.4,  "role": "Bowler",              "team": "RCB"},
    "Dewald Brevis":         {"batting_avg": 30.00, "strike_rate": 180.00, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Batsman",             "team": "CSK"},
    "Urvil Patel":           {"batting_avg": 22.67, "strike_rate": 201.56, "economy": 9.50, "wickets_per_match": 0.0,  "role": "Wicketkeeper-Batter", "team": "CSK"},
}

PLAYERS = [(name, stats["role"], stats["team"]) for name, stats in REAL_PLAYER_STATS.items()]

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
TOSS = ["Bat First", "Chase"]

# Pitch suitability weights per role
PITCH_ROLE_WEIGHTS = {
    "Flat/Batting":     {"Batsman": 0.95, "Wicketkeeper-Batter": 0.90, "All-Rounder": 0.80, "Bowler": 0.45},
    "Hard/True Bounce": {"Batsman": 0.85, "Wicketkeeper-Batter": 0.80, "All-Rounder": 0.78, "Bowler": 0.82},
    "Green/Grassy":     {"Batsman": 0.50, "Wicketkeeper-Batter": 0.55, "All-Rounder": 0.72, "Bowler": 0.95},
    "Slow/Dry Spin":    {"Batsman": 0.65, "Wicketkeeper-Batter": 0.62, "All-Rounder": 0.80, "Bowler": 0.88},
    "Wet/Dew Heavy":    {"Batsman": 0.80, "Wicketkeeper-Batter": 0.82, "All-Rounder": 0.75, "Bowler": 0.70},
}

# Player-specific pitch modifiers based on real-world speciality
PLAYER_MODIFIERS = {
    # Batsmen
    "Vaibhav Sooryavanshi":  {"Flat/Batting": 0.20, "Hard/True Bounce": 0.18},
    "Virat Kohli":           {"Flat/Batting": 0.15, "Slow/Dry Spin": 0.12},
    "Suryakumar Yadav":      {"Flat/Batting": 0.18, "Hard/True Bounce": 0.15},
    "Sai Sudharsan":         {"Flat/Batting": 0.14, "Slow/Dry Spin": 0.10},
    "Shreyas Iyer":          {"Flat/Batting": 0.13, "Hard/True Bounce": 0.08},
    "Yashasvi Jaiswal":      {"Flat/Batting": 0.12, "Hard/True Bounce": 0.10},
    "Rohit Sharma":          {"Flat/Batting": 0.12, "Hard/True Bounce": 0.08},
    "Shubman Gill":          {"Flat/Batting": 0.13, "Hard/True Bounce": 0.09},
    "Priyansh Arya":         {"Flat/Batting": 0.16, "Hard/True Bounce": 0.12},
    "Finn Allen":            {"Flat/Batting": 0.18, "Hard/True Bounce": 0.14},
    "Tim David":             {"Flat/Batting": 0.17, "Wet/Dew Heavy": 0.12},
    "Nicholas Pooran":       {"Flat/Batting": 0.16, "Wet/Dew Heavy": 0.13},
    "Jos Buttler":           {"Flat/Batting": 0.14, "Hard/True Bounce": 0.12},
    "Heinrich Klaasen":      {"Flat/Batting": 0.13, "Hard/True Bounce": 0.10},
    "Rinku Singh":           {"Wet/Dew Heavy": 0.14, "Flat/Batting": 0.10},
    "Ryan Rickelton":        {"Flat/Batting": 0.14, "Hard/True Bounce": 0.11},
    "KL Rahul":              {"Flat/Batting": 0.12, "Hard/True Bounce": 0.10},
    "Ishan Kishan":          {"Flat/Batting": 0.13, "Wet/Dew Heavy": 0.10},
    # Bowlers
    "Jasprit Bumrah":        {"Green/Grassy": 0.22, "Hard/True Bounce": 0.20, "Wet/Dew Heavy": 0.18},
    "Arshdeep Singh":        {"Green/Grassy": 0.14, "Wet/Dew Heavy": 0.18},
    "Yuzvendra Chahal":      {"Slow/Dry Spin": 0.22, "Flat/Batting": 0.06},
    "Rashid Khan":           {"Slow/Dry Spin": 0.24, "Flat/Batting": 0.10},
    "Varun Chakravarthy":    {"Slow/Dry Spin": 0.20, "Flat/Batting": 0.06},
    "Kuldeep Yadav":         {"Slow/Dry Spin": 0.18, "Flat/Batting": 0.08},
    "Bhuvneshwar Kumar":     {"Green/Grassy": 0.20, "Wet/Dew Heavy": 0.14},
    "Mohammed Shami":        {"Green/Grassy": 0.18, "Hard/True Bounce": 0.15},
    "Trent Boult":           {"Green/Grassy": 0.20, "Wet/Dew Heavy": 0.16},
    "Prasidh Krishna":       {"Green/Grassy": 0.16, "Hard/True Bounce": 0.14},
    "Noor Ahmad":            {"Slow/Dry Spin": 0.18, "Flat/Batting": 0.08},
    "Josh Hazlewood":        {"Green/Grassy": 0.18, "Hard/True Bounce": 0.16},
    "Sai Kishore":           {"Slow/Dry Spin": 0.16, "Flat/Batting": 0.06},
    # All-Rounders
    "Ravindra Jadeja":       {"Slow/Dry Spin": 0.20, "Flat/Batting": 0.10},
    "Hardik Pandya":         {"Flat/Batting": 0.10, "Green/Grassy": 0.12},
    "Axar Patel":            {"Slow/Dry Spin": 0.18, "Flat/Batting": 0.08},
    "Sunil Narine":          {"Slow/Dry Spin": 0.18, "Flat/Batting": 0.14},
    "Abhishek Sharma":       {"Flat/Batting": 0.12, "Hard/True Bounce": 0.10},
    "Mitchell Marsh":        {"Flat/Batting": 0.12, "Hard/True Bounce": 0.10},
    "Glenn Maxwell":         {"Flat/Batting": 0.14, "Slow/Dry Spin": 0.08},
    "Washington Sundar":     {"Slow/Dry Spin": 0.14, "Flat/Batting": 0.06},
    "Venkatesh Iyer":        {"Flat/Batting": 0.14, "Hard/True Bounce": 0.10},
    "Shivam Dube":           {"Flat/Batting": 0.10, "Hard/True Bounce": 0.08},
    "Krunal Pandya":         {"Slow/Dry Spin": 0.12, "Flat/Batting": 0.06},
}


def _get_real_stats(player_name, pitch_type, rng):
    """Return real base stats with small per-match noise."""
    s = REAL_PLAYER_STATS[player_name]
    noise_avg  = rng.normal(0, s["batting_avg"] * 0.08)
    noise_sr   = rng.normal(0, s["strike_rate"] * 0.06)
    noise_econ = rng.normal(0, s["economy"] * 0.06)
    noise_wpm  = rng.normal(0, max(s["wickets_per_match"] * 0.15, 0.05))

    avg  = max(s["batting_avg"]  + noise_avg,  3.0)
    sr   = max(s["strike_rate"]  + noise_sr,  50.0)
    econ = max(s["economy"]      + noise_econ, 5.0)
    wpm  = max(s["wickets_per_match"] + noise_wpm, 0.0)
    return round(avg, 2), round(sr, 2), round(econ, 2), round(wpm, 2)


def _compute_suitability(player_name, role, pitch_type, batting_avg, strike_rate, economy, wickets_per_match, rng):
    base_weight = PITCH_ROLE_WEIGHTS[pitch_type][role]
    player_mod  = PLAYER_MODIFIERS.get(player_name, {}).get(pitch_type, 0.0)

    norm_avg  = min(batting_avg / 70, 1.0)
    norm_sr   = min(strike_rate / 220, 1.0)
    norm_econ = max(0, 1 - (economy - 5) / 7)
    norm_wpm  = min(wickets_per_match / 3, 1.0)

    if role == "Batsman":
        raw = 0.45 * norm_avg + 0.45 * norm_sr + 0.10 * norm_econ
    elif role == "Bowler":
        raw = 0.20 * norm_avg + 0.10 * norm_sr + 0.45 * norm_econ + 0.25 * norm_wpm
    elif role == "All-Rounder":
        raw = 0.30 * norm_avg + 0.25 * norm_sr + 0.25 * norm_econ + 0.20 * norm_wpm
    else:  # WK-Batter
        raw = 0.40 * norm_avg + 0.45 * norm_sr + 0.10 * norm_econ + 0.05 * norm_wpm

    score = (raw * base_weight + player_mod) * 100
    score += rng.normal(0, 2)
    return round(float(np.clip(score, 0, 100)), 2)


def _reason(player_name, role, pitch_type, suitability_score, batting_avg, strike_rate, economy):
    reasons = {
        "Flat/Batting": {
            "Batsman":             f"Flat track rewards timing — IPL avg {batting_avg:.0f} & SR {strike_rate:.0f} are outstanding here.",
            "Bowler":              f"Economy of {economy:.2f} helps contain runs on a batting-friendly surface.",
            "All-Rounder":         f"Bat-first pitches let all-rounders rack up runs; avg {batting_avg:.0f} & SR {strike_rate:.0f}.",
            "Wicketkeeper-Batter": f"WK-batters excel with quick hands — SR {strike_rate:.0f} is excellent on flat tracks.",
        },
        "Hard/True Bounce": {
            "Batsman":             f"True bounce suits front-foot attacking batters — SR {strike_rate:.0f}, avg {batting_avg:.0f}.",
            "Bowler":              f"Hit-the-deck bowlers extract extra carry — economy {economy:.2f} on hard surfaces.",
            "All-Rounder":         f"Extra pace off the pitch aids bowling; batting avg {batting_avg:.0f} still competitive.",
            "Wicketkeeper-Batter": f"Hard-pitch punchy footwork — avg {batting_avg:.0f} suits this format.",
        },
        "Green/Grassy": {
            "Batsman":             f"Strong technique holds up — manages avg {batting_avg:.0f} even on a seaming surface.",
            "Bowler":              f"Grass aids seam & swing — economy {economy:.2f} with wickets per match impressive.",
            "All-Rounder":         f"Dual value: seam bowling + solid batting avg {batting_avg:.0f} on a green top.",
            "Wicketkeeper-Batter": f"Patient WK-batter can exploit loose balls after top-order falls.",
        },
        "Slow/Dry Spin": {
            "Batsman":             f"Reads spin exceptionally well — IPL avg {batting_avg:.0f} on turning tracks.",
            "Bowler":              f"Spin-friendly surface; economy {economy:.2f} and high wicket yield on dry pitches.",
            "All-Rounder":         f"Spinning all-rounder — dual impact: batting avg {batting_avg:.0f} + turns the ball.",
            "Wicketkeeper-Batter": f"Steady on sticky tracks — SR {strike_rate:.0f} still effective.",
        },
        "Wet/Dew Heavy": {
            "Batsman":             f"Dew makes chasing easier — SR {strike_rate:.0f} maximised when ball skids onto bat.",
            "Bowler":              f"Yorker specialist economy {economy:.2f} is crucial as dew limits spin.",
            "All-Rounder":         f"Death-over contributor; batting avg {batting_avg:.0f} compensates when bowling is harder.",
            "Wicketkeeper-Batter": f"Dew helps timing — WK-batter with SR {strike_rate:.0f} thrives chasing.",
        },
    }
    return reasons.get(pitch_type, {}).get(role, f"Consistent IPL performer — suitability score {suitability_score:.1f}/100.")


def generate_ipl_dataset(n_records_per_combo: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = []
    for name, role, team in PLAYERS:
        for pitch in PITCH_TYPES:
            for venue in VENUES:
                for toss in TOSS:
                    for _ in range(n_records_per_combo):
                        avg, sr, econ, wpm = _get_real_stats(name, pitch, rng)
                        suit = _compute_suitability(name, role, pitch, avg, sr, econ, wpm, rng)
                        rows.append({
                            "player_name":       name,
                            "role":              role,
                            "team":              team,
                            "pitch_type":        pitch,
                            "venue":             venue,
                            "toss":              toss,
                            "batting_avg":       avg,
                            "strike_rate":       sr,
                            "economy":           econ,
                            "wickets_per_match": wpm,
                            "match_score":       round(float(rng.normal(162, 22)), 1),
                            "suitability_score": suit,
                        })
    return pd.DataFrame(rows)

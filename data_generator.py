import pandas as pd
import numpy as np

PLAYERS = [
    ("Virat Kohli", "Batsman", "RCB"),
    ("Rohit Sharma", "Batsman", "MI"),
    ("KL Rahul", "Wicketkeeper-Batter", "LSG"),
    ("Shubman Gill", "Batsman", "GT"),
    ("David Warner", "Batsman", "DC"),
    ("Faf du Plessis", "Batsman", "RCB"),
    ("Jos Buttler", "Wicketkeeper-Batter", "RR"),
    ("Suryakumar Yadav", "Batsman", "MI"),
    ("Hardik Pandya", "All-Rounder", "MI"),
    ("Ravindra Jadeja", "All-Rounder", "CSK"),
    ("Axar Patel", "All-Rounder", "DC"),
    ("Washington Sundar", "All-Rounder", "SRH"),
    ("Shardul Thakur", "All-Rounder", "KKR"),
    ("Shivam Dube", "All-Rounder", "CSK"),
    ("Glenn Maxwell", "All-Rounder", "RCB"),
    ("Marcus Stoinis", "All-Rounder", "LSG"),
    ("Jasprit Bumrah", "Bowler", "MI"),
    ("Mohammed Shami", "Bowler", "GT"),
    ("Bhuvneshwar Kumar", "Bowler", "SRH"),
    ("Trent Boult", "Bowler", "RR"),
    ("Arshdeep Singh", "Bowler", "PBKS"),
    ("Yuzvendra Chahal", "Bowler", "RR"),
    ("Rashid Khan", "Bowler", "GT"),
    ("Kuldeep Yadav", "Bowler", "DC"),
    ("Varun Chakravarthy", "Bowler", "KKR"),
    ("Sunil Narine", "All-Rounder", "KKR"),
    ("Rinku Singh", "Batsman", "KKR"),
    ("Ishan Kishan", "Wicketkeeper-Batter", "MI"),
    ("Ruturaj Gaikwad", "Batsman", "CSK"),
    ("Tilak Varma", "Batsman", "MI"),
]

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

# Player-specific modifiers (based on real-world knowledge)
PLAYER_MODIFIERS = {
    "Virat Kohli":        {"Flat/Batting": 0.15, "Slow/Dry Spin": 0.10},
    "Rohit Sharma":       {"Flat/Batting": 0.12, "Hard/True Bounce": 0.08},
    "Suryakumar Yadav":   {"Flat/Batting": 0.18, "Hard/True Bounce": 0.15},
    "Jasprit Bumrah":     {"Green/Grassy": 0.20, "Hard/True Bounce": 0.18, "Wet/Dew Heavy": 0.15},
    "Rashid Khan":        {"Slow/Dry Spin": 0.22, "Flat/Batting": 0.10},
    "Ravindra Jadeja":    {"Slow/Dry Spin": 0.18, "Flat/Batting": 0.08},
    "Yuzvendra Chahal":   {"Slow/Dry Spin": 0.20, "Flat/Batting": 0.05},
    "Mohammed Shami":     {"Green/Grassy": 0.18, "Hard/True Bounce": 0.15},
    "Bhuvneshwar Kumar":  {"Green/Grassy": 0.20, "Wet/Dew Heavy": 0.12},
    "Jos Buttler":        {"Flat/Batting": 0.14, "Hard/True Bounce": 0.12},
    "Glenn Maxwell":      {"Flat/Batting": 0.16, "Slow/Dry Spin": 0.08},
    "Trent Boult":        {"Green/Grassy": 0.18, "Wet/Dew Heavy": 0.14},
    "KL Rahul":           {"Flat/Batting": 0.12, "Hard/True Bounce": 0.10},
    "Kuldeep Yadav":      {"Slow/Dry Spin": 0.15, "Flat/Batting": 0.08},
    "Hardik Pandya":      {"Flat/Batting": 0.10, "Green/Grassy": 0.10},
    "Arshdeep Singh":     {"Green/Grassy": 0.12, "Wet/Dew Heavy": 0.16},
    "Sunil Narine":       {"Slow/Dry Spin": 0.18, "Flat/Batting": 0.12},
}

def _base_stats(player_name, role, pitch_type, rng):
    """Generate realistic base stats per player-pitch combo."""
    if role == "Batsman":
        avg = rng.normal(38, 8)
        sr  = rng.normal(138, 18)
        econ = rng.normal(9.5, 1.5)
        wpm  = rng.uniform(0, 0.2)
    elif role == "Bowler":
        avg  = rng.normal(18, 6)
        sr   = rng.normal(105, 20)
        econ = rng.normal(7.8, 1.2)
        wpm  = rng.normal(1.6, 0.5)
    elif role == "All-Rounder":
        avg  = rng.normal(28, 7)
        sr   = rng.normal(128, 16)
        econ = rng.normal(8.5, 1.2)
        wpm  = rng.normal(0.9, 0.4)
    else:  # WK
        avg  = rng.normal(32, 8)
        sr   = rng.normal(135, 18)
        econ = rng.normal(9.2, 1.5)
        wpm  = rng.uniform(0, 0.3)

    return max(avg, 5), max(sr, 60), max(econ, 5), max(wpm, 0)


def _compute_suitability(player_name, role, pitch_type, batting_avg, strike_rate, economy, wickets_per_match, rng):
    """Compute a suitability score 0–100 with pitch-role weighting + player modifiers."""
    base_weight = PITCH_ROLE_WEIGHTS[pitch_type][role]
    player_mod  = PLAYER_MODIFIERS.get(player_name, {}).get(pitch_type, 0.0)

    # Normalise each stat to 0-1 range
    norm_avg  = min(batting_avg / 60, 1.0)
    norm_sr   = min(strike_rate / 200, 1.0)
    norm_econ = max(0, 1 - (economy - 5) / 7)   # lower economy → higher score
    norm_wpm  = min(wickets_per_match / 3, 1.0)

    if role == "Batsman":
        raw = 0.45 * norm_avg + 0.45 * norm_sr + 0.10 * norm_econ
    elif role == "Bowler":
        raw = 0.20 * norm_avg + 0.10 * norm_sr + 0.45 * norm_econ + 0.25 * norm_wpm
    elif role == "All-Rounder":
        raw = 0.30 * norm_avg + 0.25 * norm_sr + 0.25 * norm_econ + 0.20 * norm_wpm
    else:
        raw = 0.40 * norm_avg + 0.45 * norm_sr + 0.10 * norm_econ + 0.05 * norm_wpm

    score = (raw * base_weight + player_mod) * 100
    score += rng.normal(0, 3)          # small noise
    return round(np.clip(score, 0, 100), 2)


def _reason(player_name, role, pitch_type, suitability_score, batting_avg, strike_rate, economy):
    reasons = {
        "Flat/Batting": {
            "Batsman":              f"Flat track rewards timing — avg {batting_avg:.0f} and SR {strike_rate:.0f} are ideal here.",
            "Bowler":               f"Economy of {economy:.2f} helps contain runs on a belting pitch.",
            "All-Rounder":          f"Flexible role; bat-first pitches let all-rounders contribute heavily.",
            "Wicketkeeper-Batter":  f"WK-batters shine with quick scoring — SR {strike_rate:.0f} is excellent.",
        },
        "Hard/True Bounce": {
            "Batsman":              f"True bounce suits front-foot attacking batters — SR {strike_rate:.0f} with avg {batting_avg:.0f}.",
            "Bowler":               f"Hit-the-deck bowlers extract extra bounce — {economy:.2f} econ on hard surfaces.",
            "All-Rounder":          f"Extra pace off the surface aids bowling while batting remains viable.",
            "Wicketkeeper-Batter":  f"Hard pitch favors punchy footwork — avg {batting_avg:.0f} suits this format.",
        },
        "Green/Grassy": {
            "Batsman":              f"Strong technique allows avg {batting_avg:.0f} even on seaming surface.",
            "Bowler":               f"Grass aids seam & swing — economy {economy:.2f} and wickets per match impressive.",
            "All-Rounder":          f"Double value: seam bowling + solid batting avg {batting_avg:.0f}.",
            "Wicketkeeper-Batter":  f"Patient WK-batter can exploit loose balls after top-order falls.",
        },
        "Slow/Dry Spin": {
            "Batsman":              f"Reads spin well — avg {batting_avg:.0f} on turning surfaces is above average.",
            "Bowler":               f"Spin-friendly surface; economy {economy:.2f} and high wicket yield.",
            "All-Rounder":          f"Spinning all-rounder adds dual impact — both batting and turning the ball.",
            "Wicketkeeper-Batter":  f"Steady scoring on sticky track — SR {strike_rate:.0f} is still effective.",
        },
        "Wet/Dew Heavy": {
            "Batsman":              f"Chasing on dew track is easier — SR {strike_rate:.0f} maximised when ball skids on.",
            "Bowler":               f"Yorker specialist with economy {economy:.2f} is key as dew makes spin difficult.",
            "All-Rounder":          f"Reliable death-over contributor; batting contribution compensates for bowling limits.",
            "Wicketkeeper-Batter":  f"Dew helps timing — WK-batter with SR {strike_rate:.0f} thrives chasing.",
        },
    }
    return reasons.get(pitch_type, {}).get(role, f"Strong all-around performer — score {suitability_score:.1f}/100.")


def generate_ipl_dataset(n_records_per_combo: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = []
    for name, role, team in PLAYERS:
        for pitch in PITCH_TYPES:
            for venue in VENUES:
                for toss in TOSS:
                    for _ in range(n_records_per_combo):
                        avg, sr, econ, wpm = _base_stats(name, role, pitch, rng)
                        suit = _compute_suitability(name, role, pitch, avg, sr, econ, wpm, rng)
                        rows.append({
                            "player_name":        name,
                            "role":               role,
                            "team":               team,
                            "pitch_type":         pitch,
                            "venue":              venue,
                            "toss":               toss,
                            "batting_avg":        round(avg, 2),
                            "strike_rate":        round(sr, 2),
                            "economy":            round(econ, 2),
                            "wickets_per_match":  round(wpm, 2),
                            "match_score":        round(rng.normal(155, 25), 1),
                            "suitability_score":  suit,
                        })
    return pd.DataFrame(rows)

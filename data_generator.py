import pandas as pd
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Dataset size guarantee:
#   81 players × 5 pitch types × 8 venues × 2 toss options × 4 records = 25,920 rows
#   This comfortably exceeds the MIN_DATASET_ROWS = 700 requirement.
# ────────────────────────────────────────────────────────────────────────────────
MIN_DATASET_ROWS = 700

REAL_PLAYER_STATS = {

    # ══ CHENNAI SUPER KINGS (CSK) ═══════════════════════════════════════════
    # Uncapped Indians
    "Ayush Mhatre":        {"batting_avg":34.29,"strike_rate":188.97,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"CSK","data_source":"real_ipl"},
    "Anshul Kamboj":       {"batting_avg":8.00, "strike_rate":85.00, "economy":9.30,"wickets_per_match":1.0,"role":"Bowler",             "team":"CSK","data_source":"estimated_domestic"},
    "Shreyas Gopal":       {"batting_avg":7.00, "strike_rate":80.00, "economy":8.40,"wickets_per_match":1.1,"role":"Bowler",             "team":"CSK","data_source":"estimated_domestic"},
    "Mukesh Choudhary":    {"batting_avg":5.00, "strike_rate":65.00, "economy":9.00,"wickets_per_match":1.0,"role":"Bowler",             "team":"CSK","data_source":"real_ipl"},
    "Kartik Sharma":       {"batting_avg":22.00,"strike_rate":165.00,"economy":8.80,"wickets_per_match":0.6,"role":"All-Rounder",        "team":"CSK","data_source":"estimated_domestic"},
    "Sarfaraz Khan":       {"batting_avg":26.00,"strike_rate":162.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"CSK","data_source":"estimated_domestic"},
    "Ramakrishna Ghosh":   {"batting_avg":18.00,"strike_rate":150.00,"economy":8.50,"wickets_per_match":0.5,"role":"All-Rounder",        "team":"CSK","data_source":"role_based_fallback"},
    "Gurjapneet Singh":    {"batting_avg":5.00, "strike_rate":65.00, "economy":9.20,"wickets_per_match":0.9,"role":"Bowler",             "team":"CSK","data_source":"role_based_fallback"},

    # ══ MUMBAI INDIANS (MI) ════════════════════════════════════════════════
    # Uncapped Indians
    "Tilak Varma":         {"batting_avg":34.50,"strike_rate":158.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"MI","data_source":"real_ipl"},
    "Naman Dhir":          {"batting_avg":25.20,"strike_rate":182.60,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"MI","data_source":"real_ipl"},
    "Robin Minz":          {"batting_avg":22.00,"strike_rate":155.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"MI","data_source":"estimated_domestic"},
    "Ashwani Kumar":       {"batting_avg":5.00, "strike_rate":65.00, "economy":9.30,"wickets_per_match":0.9,"role":"Bowler",             "team":"MI","data_source":"role_based_fallback"},
    "Raj Angad Bawa":      {"batting_avg":18.00,"strike_rate":148.00,"economy":9.00,"wickets_per_match":0.5,"role":"All-Rounder",        "team":"MI","data_source":"estimated_domestic"},
    "Raghu Sharma":        {"batting_avg":5.00, "strike_rate":65.00, "economy":8.80,"wickets_per_match":1.0,"role":"Bowler",             "team":"MI","data_source":"role_based_fallback"},

    # ══ KOLKATA KNIGHT RIDERS (KKR) ════════════════════════════════════════════════
    # Uncapped Indians
    "Rinku Singh":         {"batting_avg":59.00,"strike_rate":149.50,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"KKR","data_source":"real_ipl"},
    "Angkrish Raghuvanshi":{"batting_avg":28.13,"strike_rate":162.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",           "team":"KKR","data_source":"real_ipl"},
    "Ramandeep Singh":     {"batting_avg":22.00,"strike_rate":176.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"KKR","data_source":"real_ipl"},
    "Ankul Roy":           {"batting_avg":12.00,"strike_rate":128.00,"economy":8.40,"wickets_per_match":0.8,"role":"All-Rounder",        "team":"KKR","data_source":"estimated_domestic"},
    "Akashdeep":           {"batting_avg":5.00, "strike_rate":65.00, "economy":9.40,"wickets_per_match":1.0,"role":"Bowler",             "team":"KKR","data_source":"estimated_domestic"},

    # ══ ROYAL CHALLENGERS BENGALURU (RCB) ══════════════════════════════════════════════
    # Uncapped Indians
    "Rajat Patidar":       {"batting_avg":35.00,"strike_rate":192.69,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"RCB","data_source":"real_ipl"},
    "Yash Dayal":          {"batting_avg":5.00, "strike_rate":65.00, "economy":9.40,"wickets_per_match":1.0,"role":"Bowler",             "team":"RCB","data_source":"real_ipl"},
    "Swapnil Singh":       {"batting_avg":14.00,"strike_rate":130.00,"economy":8.30,"wickets_per_match":0.9,"role":"All-Rounder",        "team":"RCB","data_source":"estimated_domestic"},
    "Rasikh Salam":        {"batting_avg":5.00, "strike_rate":65.00, "economy":9.10,"wickets_per_match":1.0,"role":"Bowler",             "team":"RCB","data_source":"estimated_domestic"},
    "Suyash Sharma":       {"batting_avg":5.00, "strike_rate":65.00, "economy":8.60,"wickets_per_match":1.0,"role":"Bowler",             "team":"RCB","data_source":"estimated_domestic"},

    # ══ PUNJAB KINGS (PBKS) ════════════════════════════════════════════════
    # Uncapped Indians
    "Prabhsimran Singh":   {"batting_avg":32.29,"strike_rate":160.53,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"real_ipl"},
    "Priyansh Arya":       {"batting_avg":27.94,"strike_rate":211.62,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"real_ipl"},
    "Shashank Singh":      {"batting_avg":26.14,"strike_rate":188.57,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"real_ipl"},
    "Nehal Wadhera":       {"batting_avg":24.00,"strike_rate":162.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"real_ipl"},
    "Harpreet Brar":       {"batting_avg":12.00,"strike_rate":130.00,"economy":8.50,"wickets_per_match":0.9,"role":"All-Rounder",        "team":"PBKS","data_source":"real_ipl"},
    "Musheer Khan":        {"batting_avg":20.00,"strike_rate":145.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"estimated_domestic"},
    "Vyshak Vijaykumar":   {"batting_avg":6.00, "strike_rate":70.00, "economy":9.30,"wickets_per_match":1.0,"role":"Bowler",             "team":"PBKS","data_source":"estimated_domestic"},
    "Vishnu Vinod":        {"batting_avg":22.00,"strike_rate":148.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"PBKS","data_source":"estimated_domestic"},

    # ══ GUJARAT TITANS (GT) ════════════════════════════════════════════════
    # Uncapped Indians
    "Sai Sudharsan":       {"batting_avg":46.57,"strike_rate":157.86,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"GT","data_source":"real_ipl"},
    "Sai Kishore":         {"batting_avg":6.00, "strike_rate":78.00, "economy":7.50,"wickets_per_match":1.2,"role":"Bowler",             "team":"GT","data_source":"real_ipl"},
    "Shahrukh Khan":       {"batting_avg":22.38,"strike_rate":179.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"GT","data_source":"real_ipl"},
    "Rahul Tewatia":       {"batting_avg":26.00,"strike_rate":155.00,"economy":8.80,"wickets_per_match":0.5,"role":"All-Rounder",        "team":"GT","data_source":"real_ipl"},
    "Nishant Sindhu":      {"batting_avg":16.00,"strike_rate":148.00,"economy":8.60,"wickets_per_match":0.7,"role":"All-Rounder",        "team":"GT","data_source":"estimated_domestic"},
    "Gurnoor Singh Brar":  {"batting_avg":6.00, "strike_rate":68.00, "economy":9.20,"wickets_per_match":1.0,"role":"Bowler",             "team":"GT","data_source":"estimated_domestic"},
    "Manav Suthar":        {"batting_avg":8.00, "strike_rate":80.00, "economy":8.20,"wickets_per_match":1.0,"role":"Bowler",             "team":"GT","data_source":"estimated_domestic"},
    "Kumar Kushagra":      {"batting_avg":20.00,"strike_rate":152.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"GT","data_source":"estimated_domestic"},
    "Anuj Rawat":          {"batting_avg":18.00,"strike_rate":145.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"GT","data_source":"estimated_domestic"},

    # ══ RAJASTHAN ROYALS (RR) ════════════════════════════════════════════════
    # Uncapped Indians
    "Vaibhav Sooryavanshi":{"batting_avg":45.33,"strike_rate":237.30,"economy":9.80,"wickets_per_match":0.0,"role":"Batsman",            "team":"RR","data_source":"real_ipl"},
    "Dhruv Jurel":         {"batting_avg":36.07,"strike_rate":155.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"RR","data_source":"real_ipl"},
    "Riyan Parag":         {"batting_avg":30.00,"strike_rate":158.00,"economy":8.60,"wickets_per_match":0.4,"role":"All-Rounder",        "team":"RR","data_source":"real_ipl"},
    "Kuldeep Sen":         {"batting_avg":5.00, "strike_rate":65.00, "economy":9.20,"wickets_per_match":1.1,"role":"Bowler",             "team":"RR","data_source":"estimated_domestic"},
    "Shubham Dubey":       {"batting_avg":22.00,"strike_rate":155.00,"economy":8.80,"wickets_per_match":0.5,"role":"All-Rounder",        "team":"RR","data_source":"estimated_domestic"},
    "Yudhvir Singh Charak":{"batting_avg":8.00, "strike_rate":85.00, "economy":9.10,"wickets_per_match":1.1,"role":"Bowler",             "team":"RR","data_source":"estimated_domestic"},
    "Tushar Deshpande":    {"batting_avg":6.00, "strike_rate":68.00, "economy":9.40,"wickets_per_match":1.1,"role":"Bowler",             "team":"RR","data_source":"real_ipl"},

    # ══ SUNRISERS HYDERABAD (SRH) ════════════════════════════════════════════════
    # Uncapped Indians
    "Abhishek Sharma":     {"batting_avg":33.77,"strike_rate":193.39,"economy":8.50,"wickets_per_match":0.3,"role":"All-Rounder",        "team":"SRH","data_source":"real_ipl"},
    "Nitish Kumar Reddy":  {"batting_avg":26.00,"strike_rate":158.00,"economy":9.20,"wickets_per_match":0.6,"role":"All-Rounder",        "team":"SRH","data_source":"real_ipl"},
    "Harsh Dubey":         {"batting_avg":6.00, "strike_rate":70.00, "economy":8.30,"wickets_per_match":1.1,"role":"Bowler",             "team":"SRH","data_source":"estimated_domestic"},
    "Aniket Verma":        {"batting_avg":20.00,"strike_rate":158.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"SRH","data_source":"estimated_domestic"},
    "R Smaran":            {"batting_avg":22.00,"strike_rate":150.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"SRH","data_source":"estimated_domestic"},
    "Rahul Tripathi":      {"batting_avg":26.00,"strike_rate":152.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"SRH","data_source":"real_ipl"},

    # ══ LUCKNOW SUPER GIANTS (LSG) ═══════════════════════════════════════════════
    # Uncapped Indians
    "Abdul Samad":         {"batting_avg":24.00,"strike_rate":170.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"LSG","data_source":"real_ipl"},
    "Ayush Badoni":        {"batting_avg":26.00,"strike_rate":155.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"LSG","data_source":"real_ipl"},
    "Mohsin Khan":         {"batting_avg":5.00, "strike_rate":65.00, "economy":8.70,"wickets_per_match":1.1,"role":"Bowler",             "team":"LSG","data_source":"real_ipl"},
    "Himmat Singh":        {"batting_avg":20.00,"strike_rate":148.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"LSG","data_source":"estimated_domestic"},
    "Arshin Kulkarni":     {"batting_avg":16.00,"strike_rate":145.00,"economy":8.70,"wickets_per_match":0.7,"role":"All-Rounder",        "team":"LSG","data_source":"estimated_domestic"},
    "Digvesh Rathi":       {"batting_avg":6.00, "strike_rate":68.00, "economy":8.50,"wickets_per_match":1.0,"role":"Bowler",             "team":"LSG","data_source":"estimated_domestic"},

    # ══ DELHI CAPITALS (DC) ════════════════════════════════════════════════
    # Uncapped Indians
    "Ashutosh Sharma":     {"batting_avg":24.43,"strike_rate":181.91,"economy":9.20,"wickets_per_match":0.3,"role":"All-Rounder",        "team":"DC","data_source":"real_ipl"},
    "Vipraj Nigam":        {"batting_avg":20.29,"strike_rate":179.74,"economy":8.40,"wickets_per_match":0.8,"role":"All-Rounder",        "team":"DC","data_source":"real_ipl"},
    "Sameer Rizvi":        {"batting_avg":22.00,"strike_rate":165.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"DC","data_source":"real_ipl"},
    "Abishek Porel":       {"batting_avg":20.00,"strike_rate":152.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"DC","data_source":"estimated_domestic"},
    "Mukesh Kumar":        {"batting_avg":5.00, "strike_rate":65.00, "economy":9.10,"wickets_per_match":1.1,"role":"Bowler",             "team":"DC","data_source":"real_ipl"},
    "Yash Dhull":          {"batting_avg":22.00,"strike_rate":148.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"DC","data_source":"estimated_domestic"},

    # ══ IPL 2025 AUCTION UNSOLD — UNCAPPED INDIANS ═════════════════════════════════════════════
    # Batsmen / WK-Batters
    "Prithvi Shaw":        {"batting_avg":23.95,"strike_rate":147.47,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"UNSOLD","data_source":"real_ipl"},
    "Anmolpreet Singh":    {"batting_avg":20.00,"strike_rate":138.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"UNSOLD","data_source":"estimated_domestic"},
    "Upendra Yadav":       {"batting_avg":18.00,"strike_rate":145.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"UNSOLD","data_source":"estimated_domestic"},
    "Luvnith Sisodia":     {"batting_avg":16.00,"strike_rate":140.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"UNSOLD","data_source":"estimated_domestic"},
    "Lalit Yadav":         {"batting_avg":18.00,"strike_rate":140.00,"economy":8.80,"wickets_per_match":0.5,"role":"All-Rounder",        "team":"UNSOLD","data_source":"real_ipl"},
    "Mayank Agarwal":      {"batting_avg":26.00,"strike_rate":140.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"UNSOLD","data_source":"real_ipl"},
    # Bowlers
    "Kartik Tyagi":        {"batting_avg":5.00, "strike_rate":65.00, "economy":9.30,"wickets_per_match":1.0,"role":"Bowler",             "team":"UNSOLD","data_source":"real_ipl"},
    "Chetan Sakariya":     {"batting_avg":5.00, "strike_rate":65.00, "economy":9.00,"wickets_per_match":1.1,"role":"Bowler",             "team":"UNSOLD","data_source":"real_ipl"},
    "Navdeep Saini":       {"batting_avg":5.00, "strike_rate":65.00, "economy":9.50,"wickets_per_match":1.0,"role":"Bowler",             "team":"UNSOLD","data_source":"real_ipl"},
    "Vidwath Kaverappa":   {"batting_avg":5.00, "strike_rate":65.00, "economy":9.20,"wickets_per_match":1.1,"role":"Bowler",             "team":"UNSOLD","data_source":"estimated_domestic"},
    "Rajan Kumar":         {"batting_avg":5.00, "strike_rate":65.00, "economy":9.10,"wickets_per_match":1.0,"role":"Bowler",             "team":"UNSOLD","data_source":"estimated_domestic"},
    "Piyush Chawla":       {"batting_avg":6.00, "strike_rate":72.00, "economy":8.00,"wickets_per_match":1.1,"role":"Bowler",             "team":"UNSOLD","data_source":"real_ipl"},
    # All-Rounders
    "Utkarsh Singh":       {"batting_avg":16.00,"strike_rate":145.00,"economy":9.20,"wickets_per_match":0.6,"role":"All-Rounder",        "team":"UNSOLD","data_source":"estimated_domestic"},
    "Mayank Dagar":        {"batting_avg":14.00,"strike_rate":130.00,"economy":8.50,"wickets_per_match":0.8,"role":"All-Rounder",        "team":"UNSOLD","data_source":"estimated_domestic"},
    "Prince Chaudhary":    {"batting_avg":16.00,"strike_rate":142.00,"economy":9.00,"wickets_per_match":0.6,"role":"All-Rounder",        "team":"UNSOLD","data_source":"estimated_domestic"},
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

# ── Venue + Pitch combo score modifiers ───────────────────────────────────────────────
VENUE_PITCH_MODIFIERS = {
    "Wankhede Stadium, Mumbai": {
        "Flat/Batting":     {"Batsman": 8,  "Wicketkeeper-Batter": 7,  "All-Rounder": 4,  "Bowler": -4},
        "Hard/True Bounce": {"Batsman": 5,  "Wicketkeeper-Batter": 4,  "All-Rounder": 5,  "Bowler":  6},
        "Green/Grassy":     {"Batsman": -3, "Wicketkeeper-Batter": -2, "All-Rounder": 3,  "Bowler":  9},
        "Slow/Dry Spin":    {"Batsman": 2,  "Wicketkeeper-Batter": 2,  "All-Rounder": 4,  "Bowler":  3},
        "Wet/Dew Heavy":    {"Batsman": 9,  "Wicketkeeper-Batter": 8,  "All-Rounder": 5,  "Bowler": -5},
    },
    "M Chinnaswamy Stadium, Bengaluru": {
        "Flat/Batting":     {"Batsman": 10, "Wicketkeeper-Batter": 9,  "All-Rounder": 5,  "Bowler": -6},
        "Hard/True Bounce": {"Batsman": 6,  "Wicketkeeper-Batter": 5,  "All-Rounder": 4,  "Bowler":  3},
        "Green/Grassy":     {"Batsman": -2, "Wicketkeeper-Batter": -1, "All-Rounder": 2,  "Bowler":  7},
        "Slow/Dry Spin":    {"Batsman": 3,  "Wicketkeeper-Batter": 3,  "All-Rounder": 5,  "Bowler":  4},
        "Wet/Dew Heavy":    {"Batsman": 8,  "Wicketkeeper-Batter": 7,  "All-Rounder": 4,  "Bowler": -4},
    },
    "Eden Gardens, Kolkata": {
        "Flat/Batting":     {"Batsman": 5,  "Wicketkeeper-Batter": 5,  "All-Rounder": 4,  "Bowler": -2},
        "Hard/True Bounce": {"Batsman": 3,  "Wicketkeeper-Batter": 3,  "All-Rounder": 5,  "Bowler":  7},
        "Green/Grassy":     {"Batsman": -4, "Wicketkeeper-Batter": -3, "All-Rounder": 3,  "Bowler": 11},
        "Slow/Dry Spin":    {"Batsman": 4,  "Wicketkeeper-Batter": 3,  "All-Rounder": 7,  "Bowler":  8},
        "Wet/Dew Heavy":    {"Batsman": 6,  "Wicketkeeper-Batter": 6,  "All-Rounder": 4,  "Bowler": -3},
    },
    "Narendra Modi Stadium, Ahmedabad": {
        "Flat/Batting":     {"Batsman": 4,  "Wicketkeeper-Batter": 4,  "All-Rounder": 3,  "Bowler": -1},
        "Hard/True Bounce": {"Batsman": 3,  "Wicketkeeper-Batter": 3,  "All-Rounder": 4,  "Bowler":  5},
        "Green/Grassy":     {"Batsman": -2, "Wicketkeeper-Batter": -1, "All-Rounder": 2,  "Bowler":  6},
        "Slow/Dry Spin":    {"Batsman": 5,  "Wicketkeeper-Batter": 4,  "All-Rounder": 8,  "Bowler": 10},
        "Wet/Dew Heavy":    {"Batsman": 5,  "Wicketkeeper-Batter": 5,  "All-Rounder": 3,  "Bowler": -2},
    },
    "Arun Jaitley Stadium, Delhi": {
        "Flat/Batting":     {"Batsman": 5,  "Wicketkeeper-Batter": 5,  "All-Rounder": 4,  "Bowler": -3},
        "Hard/True Bounce": {"Batsman": 7,  "Wicketkeeper-Batter": 6,  "All-Rounder": 6,  "Bowler":  8},
        "Green/Grassy":     {"Batsman": -3, "Wicketkeeper-Batter": -2, "All-Rounder": 3,  "Bowler": 10},
        "Slow/Dry Spin":    {"Batsman": 3,  "Wicketkeeper-Batter": 2,  "All-Rounder": 5,  "Bowler":  6},
        "Wet/Dew Heavy":    {"Batsman": 6,  "Wicketkeeper-Batter": 6,  "All-Rounder": 4,  "Bowler": -2},
    },
    "MA Chidambaram Stadium, Chennai": {
        "Flat/Batting":     {"Batsman": 3,  "Wicketkeeper-Batter": 3,  "All-Rounder": 4,  "Bowler":  0},
        "Hard/True Bounce": {"Batsman": 1,  "Wicketkeeper-Batter": 1,  "All-Rounder": 3,  "Bowler":  4},
        "Green/Grassy":     {"Batsman": -2, "Wicketkeeper-Batter": -1, "All-Rounder": 2,  "Bowler":  7},
        "Slow/Dry Spin":    {"Batsman": 6,  "Wicketkeeper-Batter": 5,  "All-Rounder": 9,  "Bowler": 13},
        "Wet/Dew Heavy":    {"Batsman": 4,  "Wicketkeeper-Batter": 4,  "All-Rounder": 3,  "Bowler":  1},
    },
    "Rajiv Gandhi Stadium, Hyderabad": {
        "Flat/Batting":     {"Batsman": 7,  "Wicketkeeper-Batter": 7,  "All-Rounder": 5,  "Bowler": -4},
        "Hard/True Bounce": {"Batsman": 4,  "Wicketkeeper-Batter": 4,  "All-Rounder": 4,  "Bowler":  5},
        "Green/Grassy":     {"Batsman": -2, "Wicketkeeper-Batter": -1, "All-Rounder": 2,  "Bowler":  8},
        "Slow/Dry Spin":    {"Batsman": 4,  "Wicketkeeper-Batter": 4,  "All-Rounder": 6,  "Bowler":  7},
        "Wet/Dew Heavy":    {"Batsman": 7,  "Wicketkeeper-Batter": 7,  "All-Rounder": 4,  "Bowler": -3},
    },
    "Sawai Mansingh Stadium, Jaipur": {
        "Flat/Batting":     {"Batsman": 4,  "Wicketkeeper-Batter": 4,  "All-Rounder": 3,  "Bowler": -2},
        "Hard/True Bounce": {"Batsman": 3,  "Wicketkeeper-Batter": 3,  "All-Rounder": 4,  "Bowler":  5},
        "Green/Grassy":     {"Batsman": -3, "Wicketkeeper-Batter": -2, "All-Rounder": 3,  "Bowler":  8},
        "Slow/Dry Spin":    {"Batsman": 5,  "Wicketkeeper-Batter": 5,  "All-Rounder": 7,  "Bowler": 11},
        "Wet/Dew Heavy":    {"Batsman": 5,  "Wicketkeeper-Batter": 5,  "All-Rounder": 3,  "Bowler": -1},
    },
}

PITCH_ROLE_WEIGHTS = {
    "Flat/Batting":     {"Batsman": 0.95, "Wicketkeeper-Batter": 0.90, "All-Rounder": 0.80, "Bowler": 0.45},
    "Hard/True Bounce": {"Batsman": 0.85, "Wicketkeeper-Batter": 0.80, "All-Rounder": 0.78, "Bowler": 0.82},
    "Green/Grassy":     {"Batsman": 0.50, "Wicketkeeper-Batter": 0.55, "All-Rounder": 0.72, "Bowler": 0.95},
    "Slow/Dry Spin":    {"Batsman": 0.65, "Wicketkeeper-Batter": 0.62, "All-Rounder": 0.80, "Bowler": 0.88},
    "Wet/Dew Heavy":    {"Batsman": 0.80, "Wicketkeeper-Batter": 0.82, "All-Rounder": 0.75, "Bowler": 0.70},
}

# ── PLAYER_MODIFIERS (uncapped Indian players only) ──────────────────────────────────
PLAYER_MODIFIERS = {
    # ─ Batsmen (signed)
    "Vaibhav Sooryavanshi": {"Flat/Batting": 0.20, "Hard/True Bounce": 0.18},
    "Sai Sudharsan":        {"Flat/Batting": 0.14, "Slow/Dry Spin": 0.10},
    "Priyansh Arya":        {"Flat/Batting": 0.16, "Hard/True Bounce": 0.12},
    "Rinku Singh":          {"Wet/Dew Heavy": 0.14, "Flat/Batting": 0.10},
    "Rajat Patidar":        {"Flat/Batting": 0.14, "Hard/True Bounce": 0.10},
    "Tilak Varma":          {"Flat/Batting": 0.12, "Slow/Dry Spin": 0.09},
    "Prabhsimran Singh":    {"Flat/Batting": 0.11, "Hard/True Bounce": 0.09},
    "Shashank Singh":       {"Flat/Batting": 0.12, "Wet/Dew Heavy": 0.10},
    "Dhruv Jurel":          {"Flat/Batting": 0.11, "Hard/True Bounce": 0.08},
    "Abhishek Sharma":      {"Flat/Batting": 0.12, "Hard/True Bounce": 0.10},
    "Ayush Mhatre":         {"Flat/Batting": 0.14, "Hard/True Bounce": 0.11},
    "Angkrish Raghuvanshi": {"Flat/Batting": 0.10, "Hard/True Bounce": 0.08},
    "Shahrukh Khan":        {"Flat/Batting": 0.10, "Wet/Dew Heavy": 0.10},
    "Riyan Parag":          {"Slow/Dry Spin": 0.10, "Flat/Batting": 0.08},
    "Nitish Kumar Reddy":   {"Flat/Batting": 0.10, "Hard/True Bounce": 0.08},
    # ─ Batsmen (unsold)
    "Prithvi Shaw":         {"Flat/Batting": 0.12, "Hard/True Bounce": 0.10},
    "Mayank Agarwal":       {"Flat/Batting": 0.10, "Hard/True Bounce": 0.08},
    "Lalit Yadav":          {"Flat/Batting": 0.08, "Slow/Dry Spin": 0.06},
    # ─ Bowlers (signed)
    "Sai Kishore":          {"Slow/Dry Spin": 0.16, "Flat/Batting": 0.06},
    "Yash Dayal":           {"Green/Grassy": 0.10, "Wet/Dew Heavy": 0.08},
    "Anshul Kamboj":        {"Green/Grassy": 0.10, "Hard/True Bounce": 0.08},
    "Harsh Dubey":          {"Slow/Dry Spin": 0.10, "Green/Grassy": 0.08},
    "Vyshak Vijaykumar":    {"Green/Grassy": 0.10, "Hard/True Bounce": 0.08},
    "Gurnoor Singh Brar":   {"Green/Grassy": 0.10, "Hard/True Bounce": 0.08},
    # ─ Bowlers (unsold)
    "Kartik Tyagi":         {"Green/Grassy": 0.10, "Hard/True Bounce": 0.10},
    "Chetan Sakariya":      {"Green/Grassy": 0.10, "Wet/Dew Heavy": 0.08},
    "Navdeep Saini":        {"Hard/True Bounce": 0.10, "Green/Grassy": 0.08},
    "Piyush Chawla":        {"Slow/Dry Spin": 0.14, "Flat/Batting": 0.06},
    "Vidwath Kaverappa":    {"Green/Grassy": 0.10, "Hard/True Bounce": 0.08},
    # ─ All-rounders (signed)
    "Rahul Tewatia":        {"Flat/Batting": 0.10, "Wet/Dew Heavy": 0.08},
    "Harpreet Brar":        {"Slow/Dry Spin": 0.10, "Flat/Batting": 0.06},
    "Nishant Sindhu":       {"Flat/Batting": 0.08, "Slow/Dry Spin": 0.06},
    "Ashutosh Sharma":      {"Flat/Batting": 0.10, "Hard/True Bounce": 0.08},
    "Vipraj Nigam":         {"Flat/Batting": 0.08, "Slow/Dry Spin": 0.06},
    "Arshin Kulkarni":      {"Slow/Dry Spin": 0.08, "Flat/Batting": 0.06},
    # ─ All-rounders (unsold)
    "Utkarsh Singh":        {"Flat/Batting": 0.08, "Hard/True Bounce": 0.06},
    "Mayank Dagar":         {"Slow/Dry Spin": 0.10, "Flat/Batting": 0.06},
    "Prince Chaudhary":     {"Flat/Batting": 0.08, "Slow/Dry Spin": 0.06},
}


def _get_real_stats(player_name, pitch_type, rng):
    s = REAL_PLAYER_STATS[player_name]
    noise_avg  = rng.normal(0, s["batting_avg"]  * 0.08)
    noise_sr   = rng.normal(0, s["strike_rate"]  * 0.06)
    noise_econ = rng.normal(0, s["economy"]      * 0.06)
    noise_wpm  = rng.normal(0, max(s["wickets_per_match"] * 0.15, 0.05))
    avg  = max(s["batting_avg"]  + noise_avg,  3.0)
    sr   = max(s["strike_rate"]  + noise_sr,  50.0)
    econ = max(s["economy"]      + noise_econ, 5.0)
    wpm  = max(s["wickets_per_match"] + noise_wpm, 0.0)
    return round(avg, 2), round(sr, 2), round(econ, 2), round(wpm, 2)


def _compute_suitability(player_name, role, pitch_type, venue, batting_avg, strike_rate, economy, wickets_per_match, rng):
    base_weight  = PITCH_ROLE_WEIGHTS[pitch_type][role]
    player_mod   = PLAYER_MODIFIERS.get(player_name, {}).get(pitch_type, 0.0)
    venue_mod    = VENUE_PITCH_MODIFIERS.get(venue, {}).get(pitch_type, {}).get(role, 0)
    norm_avg  = min(batting_avg  / 70,  1.0)
    norm_sr   = min(strike_rate  / 220, 1.0)
    norm_econ = max(0.0, 1.0 - (economy - 5.0) / 7.0)
    norm_wpm  = min(wickets_per_match / 3.0, 1.0)
    if role == "Batsman":
        raw = 0.45 * norm_avg + 0.45 * norm_sr + 0.10 * norm_econ
    elif role == "Bowler":
        raw = 0.20 * norm_avg + 0.10 * norm_sr + 0.45 * norm_econ + 0.25 * norm_wpm
    elif role == "All-Rounder":
        raw = 0.30 * norm_avg + 0.25 * norm_sr + 0.25 * norm_econ + 0.20 * norm_wpm
    else:  # Wicketkeeper-Batter
        raw = 0.40 * norm_avg + 0.45 * norm_sr + 0.10 * norm_econ + 0.05 * norm_wpm
    score = (raw * base_weight + player_mod) * 100 + venue_mod
    score += rng.normal(0, 2)
    return round(float(np.clip(score, 0, 100)), 2)


def _reason(player_name, role, pitch_type, suitability_score, batting_avg, strike_rate, economy):
    ds  = REAL_PLAYER_STATS.get(player_name, {}).get("data_source", "real_ipl")
    tag = "" if ds == "real_ipl" else " (est.)"
    reasons = {
        "Flat/Batting": {
            "Batsman":             f"Flat track rewards timing — IPL avg {batting_avg:.0f}{tag} & SR {strike_rate:.0f}.",
            "Bowler":              f"Economy {economy:.2f}{tag} helps contain runs on a flat surface.",
            "All-Rounder":         f"Bat-first pitch unlocks both skills; avg {batting_avg:.0f}{tag}, SR {strike_rate:.0f}.",
            "Wicketkeeper-Batter": f"WK-batter with SR {strike_rate:.0f}{tag} excels on flat batting tracks.",
        },
        "Hard/True Bounce": {
            "Batsman":             f"True bounce suits front-foot batters — SR {strike_rate:.0f}{tag}, avg {batting_avg:.0f}.",
            "Bowler":              f"Hit-the-deck pace bowler; economy {economy:.2f}{tag} on hard surfaces.",
            "All-Rounder":         f"Extra bounce aids bowling; batting avg {batting_avg:.0f}{tag} still competitive.",
            "Wicketkeeper-Batter": f"Hard-pitch punchy batter — avg {batting_avg:.0f}{tag} suits this format.",
        },
        "Green/Grassy": {
            "Batsman":             f"Solid technique holds up — avg {batting_avg:.0f}{tag} on a seaming surface.",
            "Bowler":              f"Seam & swing support; economy {economy:.2f}{tag} + wickets impressive on green top.",
            "All-Rounder":         f"Seam bowling + batting avg {batting_avg:.0f}{tag} = dual impact on green top.",
            "Wicketkeeper-Batter": f"Patient batter exploits loose deliveries after top-order falls.",
        },
        "Slow/Dry Spin": {
            "Batsman":             f"Reads spin well — avg {batting_avg:.0f}{tag} on turning tracks.",
            "Bowler":              f"Spin surface; economy {economy:.2f}{tag} + high wicket yield on dry pitches.",
            "All-Rounder":         f"Spinning all-rounder: avg {batting_avg:.0f}{tag} + turns the ball.",
            "Wicketkeeper-Batter": f"Steady on sticky tracks — SR {strike_rate:.0f}{tag} still effective.",
        },
        "Wet/Dew Heavy": {
            "Batsman":             f"Dew aids chasing — SR {strike_rate:.0f}{tag} maximised when ball skids on.",
            "Bowler":              f"Yorker specialist; economy {economy:.2f}{tag} crucial as dew limits spin.",
            "All-Rounder":         f"Death-over impact; avg {batting_avg:.0f}{tag} compensates when bowling is harder.",
            "Wicketkeeper-Batter": f"Dew helps timing — SR {strike_rate:.0f}{tag} thrives when chasing.",
        },
    }
    return reasons.get(pitch_type, {}).get(
        role, f"Consistent IPL performer — suitability score {suitability_score:.1f}/100{tag}."
    )


def generate_ipl_dataset(n_records_per_combo: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = []
    for name, role, team in PLAYERS:
        for pitch in PITCH_TYPES:
            for venue in VENUES:
                for toss in TOSS:
                    for _ in range(n_records_per_combo):
                        avg, sr, econ, wpm = _get_real_stats(name, pitch, rng)
                        suit = _compute_suitability(name, role, pitch, venue, avg, sr, econ, wpm, rng)
                        rows.append({
                            "player_name":       name,
                            "role":              role,
                            "team":              team,
                            "data_source":       REAL_PLAYER_STATS[name]["data_source"],
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
    df = pd.DataFrame(rows)
    assert len(df) >= MIN_DATASET_ROWS, (
        f"Dataset has only {len(df)} rows — must be >= {MIN_DATASET_ROWS}. "
        f"Increase n_records_per_combo or add more players."
    )
    return df

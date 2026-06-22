import pandas as pd
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Dataset size guarantee:
#   700 players × 5 pitch types × 8 venues × 2 toss options × 4 records = 224,000 rows
#   This comfortably exceeds the MIN_DATASET_ROWS = 700 requirement.
# ────────────────────────────────────────────────────────────────────────────────
MIN_DATASET_ROWS = 700

REAL_PLAYER_STATS = {

    # ══ CHENNAI SUPER KINGS (CSK) ═══════════════════════════════════════════
    "Ayush Mhatre":        {"batting_avg":34.29,"strike_rate":188.97,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"CSK","data_source":"real_ipl"},
    "Anshul Kamboj":       {"batting_avg":8.00, "strike_rate":85.00, "economy":9.30,"wickets_per_match":1.0,"role":"Bowler",             "team":"CSK","data_source":"estimated_domestic"},
    "Shreyas Gopal":       {"batting_avg":7.00, "strike_rate":80.00, "economy":8.40,"wickets_per_match":1.1,"role":"Bowler",             "team":"CSK","data_source":"estimated_domestic"},
    "Mukesh Choudhary":    {"batting_avg":5.00, "strike_rate":65.00, "economy":9.00,"wickets_per_match":1.0,"role":"Bowler",             "team":"CSK","data_source":"real_ipl"},
    "Kartik Sharma":       {"batting_avg":22.00,"strike_rate":165.00,"economy":8.80,"wickets_per_match":0.6,"role":"All-Rounder",        "team":"CSK","data_source":"estimated_domestic"},
    "Sarfaraz Khan":       {"batting_avg":26.00,"strike_rate":162.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"CSK","data_source":"estimated_domestic"},
    "Ramakrishna Ghosh":   {"batting_avg":18.00,"strike_rate":150.00,"economy":8.50,"wickets_per_match":0.5,"role":"All-Rounder",        "team":"CSK","data_source":"role_based_fallback"},
    "Gurjapneet Singh":    {"batting_avg":5.00, "strike_rate":65.00, "economy":9.20,"wickets_per_match":0.9,"role":"Bowler",             "team":"CSK","data_source":"role_based_fallback"},

    # ══ MUMBAI INDIANS (MI) ════════════════════════════════════════════════
    "Tilak Varma":         {"batting_avg":34.50,"strike_rate":158.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"MI","data_source":"real_ipl"},
    "Naman Dhir":          {"batting_avg":25.20,"strike_rate":182.60,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"MI","data_source":"real_ipl"},
    "Robin Minz":          {"batting_avg":22.00,"strike_rate":155.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"MI","data_source":"estimated_domestic"},
    "Ashwani Kumar":       {"batting_avg":5.00, "strike_rate":65.00, "economy":9.30,"wickets_per_match":0.9,"role":"Bowler",             "team":"MI","data_source":"role_based_fallback"},
    "Raj Angad Bawa":      {"batting_avg":18.00,"strike_rate":148.00,"economy":9.00,"wickets_per_match":0.5,"role":"All-Rounder",        "team":"MI","data_source":"estimated_domestic"},
    "Raghu Sharma":        {"batting_avg":5.00, "strike_rate":65.00, "economy":8.80,"wickets_per_match":1.0,"role":"Bowler",             "team":"MI","data_source":"role_based_fallback"},

    # ══ KOLKATA KNIGHT RIDERS (KKR) ════════════════════════════════════════
    "Rinku Singh":         {"batting_avg":59.00,"strike_rate":149.50,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"KKR","data_source":"real_ipl"},
    "Angkrish Raghuvanshi":{"batting_avg":28.13,"strike_rate":162.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"KKR","data_source":"real_ipl"},
    "Ramandeep Singh":     {"batting_avg":22.00,"strike_rate":176.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"KKR","data_source":"real_ipl"},
    "Ankul Roy":           {"batting_avg":12.00,"strike_rate":128.00,"economy":8.40,"wickets_per_match":0.8,"role":"All-Rounder",        "team":"KKR","data_source":"estimated_domestic"},
    "Akashdeep":           {"batting_avg":5.00, "strike_rate":65.00, "economy":9.40,"wickets_per_match":1.0,"role":"Bowler",             "team":"KKR","data_source":"estimated_domestic"},

    # ══ ROYAL CHALLENGERS BENGALURU (RCB) ══════════════════════════════════
    "Rajat Patidar":       {"batting_avg":35.00,"strike_rate":192.69,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"RCB","data_source":"real_ipl"},
    "Yash Dayal":          {"batting_avg":5.00, "strike_rate":65.00, "economy":9.40,"wickets_per_match":1.0,"role":"Bowler",             "team":"RCB","data_source":"real_ipl"},
    "Swapnil Singh":       {"batting_avg":14.00,"strike_rate":130.00,"economy":8.30,"wickets_per_match":0.9,"role":"All-Rounder",        "team":"RCB","data_source":"estimated_domestic"},
    "Rasikh Salam":        {"batting_avg":5.00, "strike_rate":65.00, "economy":9.10,"wickets_per_match":1.0,"role":"Bowler",             "team":"RCB","data_source":"estimated_domestic"},
    "Suyash Sharma":       {"batting_avg":5.00, "strike_rate":65.00, "economy":8.60,"wickets_per_match":1.0,"role":"Bowler",             "team":"RCB","data_source":"estimated_domestic"},

    # ══ PUNJAB KINGS (PBKS) ════════════════════════════════════════════════
    "Prabhsimran Singh":   {"batting_avg":32.29,"strike_rate":160.53,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"real_ipl"},
    "Priyansh Arya":       {"batting_avg":27.94,"strike_rate":211.62,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"real_ipl"},
    "Shashank Singh":      {"batting_avg":26.14,"strike_rate":188.57,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"real_ipl"},
    "Nehal Wadhera":       {"batting_avg":24.00,"strike_rate":162.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"real_ipl"},
    "Harpreet Brar":       {"batting_avg":12.00,"strike_rate":130.00,"economy":8.50,"wickets_per_match":0.9,"role":"All-Rounder",        "team":"PBKS","data_source":"real_ipl"},
    "Musheer Khan":        {"batting_avg":20.00,"strike_rate":145.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"estimated_domestic"},
    "Vyshak Vijaykumar":   {"batting_avg":6.00, "strike_rate":70.00, "economy":9.30,"wickets_per_match":1.0,"role":"Bowler",             "team":"PBKS","data_source":"estimated_domestic"},
    "Vishnu Vinod":        {"batting_avg":22.00,"strike_rate":148.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"PBKS","data_source":"estimated_domestic"},

    # ══ GUJARAT TITANS (GT) ════════════════════════════════════════════════
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
    "Vaibhav Sooryavanshi":{"batting_avg":45.33,"strike_rate":237.30,"economy":9.80,"wickets_per_match":0.0,"role":"Batsman",            "team":"RR","data_source":"real_ipl"},
    "Dhruv Jurel":         {"batting_avg":36.07,"strike_rate":155.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"RR","data_source":"real_ipl"},
    "Riyan Parag":         {"batting_avg":30.00,"strike_rate":158.00,"economy":8.60,"wickets_per_match":0.4,"role":"All-Rounder",        "team":"RR","data_source":"real_ipl"},
    "Kuldeep Sen":         {"batting_avg":5.00, "strike_rate":65.00, "economy":9.20,"wickets_per_match":1.1,"role":"Bowler",             "team":"RR","data_source":"estimated_domestic"},
    "Shubham Dubey":       {"batting_avg":22.00,"strike_rate":155.00,"economy":8.80,"wickets_per_match":0.5,"role":"All-Rounder",        "team":"RR","data_source":"estimated_domestic"},
    "Yudhvir Singh Charak":{"batting_avg":8.00, "strike_rate":85.00, "economy":9.10,"wickets_per_match":1.1,"role":"Bowler",             "team":"RR","data_source":"estimated_domestic"},
    "Tushar Deshpande":    {"batting_avg":6.00, "strike_rate":68.00, "economy":9.40,"wickets_per_match":1.1,"role":"Bowler",             "team":"RR","data_source":"real_ipl"},

    # ══ SUNRISERS HYDERABAD (SRH) ════════════════════════════════════════════
    "Abhishek Sharma":     {"batting_avg":33.77,"strike_rate":193.39,"economy":8.50,"wickets_per_match":0.3,"role":"All-Rounder",        "team":"SRH","data_source":"real_ipl"},
    "Nitish Kumar Reddy":  {"batting_avg":26.00,"strike_rate":158.00,"economy":9.20,"wickets_per_match":0.6,"role":"All-Rounder",        "team":"SRH","data_source":"real_ipl"},
    "Harsh Dubey":         {"batting_avg":6.00, "strike_rate":70.00, "economy":8.30,"wickets_per_match":1.1,"role":"Bowler",             "team":"SRH","data_source":"estimated_domestic"},
    "Aniket Verma":        {"batting_avg":20.00,"strike_rate":158.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"SRH","data_source":"estimated_domestic"},
    "R Smaran":            {"batting_avg":22.00,"strike_rate":150.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"SRH","data_source":"estimated_domestic"},
    "Rahul Tripathi":      {"batting_avg":26.00,"strike_rate":152.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"SRH","data_source":"real_ipl"},

    # ══ LUCKNOW SUPER GIANTS (LSG) ═══════════════════════════════════════════
    "Abdul Samad":         {"batting_avg":24.00,"strike_rate":170.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"LSG","data_source":"real_ipl"},
    "Ayush Badoni":        {"batting_avg":26.00,"strike_rate":155.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"LSG","data_source":"real_ipl"},
    "Mohsin Khan":         {"batting_avg":5.00, "strike_rate":65.00, "economy":8.70,"wickets_per_match":1.1,"role":"Bowler",             "team":"LSG","data_source":"real_ipl"},
    "Himmat Singh":        {"batting_avg":20.00,"strike_rate":148.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"LSG","data_source":"estimated_domestic"},
    "Arshin Kulkarni":     {"batting_avg":16.00,"strike_rate":145.00,"economy":8.70,"wickets_per_match":0.7,"role":"All-Rounder",        "team":"LSG","data_source":"estimated_domestic"},
    "Digvesh Rathi":       {"batting_avg":6.00, "strike_rate":68.00, "economy":8.50,"wickets_per_match":1.0,"role":"Bowler",             "team":"LSG","data_source":"estimated_domestic"},

    # ══ DELHI CAPITALS (DC) ════════════════════════════════════════════════
    "Ashutosh Sharma":     {"batting_avg":24.43,"strike_rate":181.91,"economy":9.20,"wickets_per_match":0.3,"role":"All-Rounder",        "team":"DC","data_source":"real_ipl"},
    "Vipraj Nigam":        {"batting_avg":20.29,"strike_rate":179.74,"economy":8.40,"wickets_per_match":0.8,"role":"All-Rounder",        "team":"DC","data_source":"real_ipl"},
    "Sameer Rizvi":        {"batting_avg":22.00,"strike_rate":165.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"DC","data_source":"real_ipl"},
    "Abishek Porel":       {"batting_avg":20.00,"strike_rate":152.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"DC","data_source":"estimated_domestic"},
    "Mukesh Kumar":        {"batting_avg":5.00, "strike_rate":65.00, "economy":9.10,"wickets_per_match":1.1,"role":"Bowler",             "team":"DC","data_source":"real_ipl"},
    "Yash Dhull":          {"batting_avg":22.00,"strike_rate":148.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"DC","data_source":"estimated_domestic"},

    # ══ IPL 2025 AUCTION UNSOLD ══════════════════════════════════════════════
    "Prithvi Shaw":        {"batting_avg":23.95,"strike_rate":147.47,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"UNSOLD","data_source":"real_ipl"},
    "Anmolpreet Singh":    {"batting_avg":20.00,"strike_rate":138.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"UNSOLD","data_source":"estimated_domestic"},
    "Upendra Yadav":       {"batting_avg":18.00,"strike_rate":145.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"UNSOLD","data_source":"estimated_domestic"},
    "Luvnith Sisodia":     {"batting_avg":16.00,"strike_rate":140.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"UNSOLD","data_source":"estimated_domestic"},
    "Lalit Yadav":         {"batting_avg":18.00,"strike_rate":140.00,"economy":8.80,"wickets_per_match":0.5,"role":"All-Rounder",        "team":"UNSOLD","data_source":"real_ipl"},
    "Mayank Agarwal":      {"batting_avg":26.00,"strike_rate":140.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"UNSOLD","data_source":"real_ipl"},
    "Kartik Tyagi":        {"batting_avg":5.00, "strike_rate":65.00, "economy":9.30,"wickets_per_match":1.0,"role":"Bowler",             "team":"UNSOLD","data_source":"real_ipl"},
    "Chetan Sakariya":     {"batting_avg":5.00, "strike_rate":65.00, "economy":9.00,"wickets_per_match":1.1,"role":"Bowler",             "team":"UNSOLD","data_source":"real_ipl"},
    "Navdeep Saini":       {"batting_avg":5.00, "strike_rate":65.00, "economy":9.50,"wickets_per_match":1.0,"role":"Bowler",             "team":"UNSOLD","data_source":"real_ipl"},
    "Vidwath Kaverappa":   {"batting_avg":5.00, "strike_rate":65.00, "economy":9.20,"wickets_per_match":1.1,"role":"Bowler",             "team":"UNSOLD","data_source":"estimated_domestic"},
    "Rajan Kumar":         {"batting_avg":5.00, "strike_rate":65.00, "economy":9.10,"wickets_per_match":1.0,"role":"Bowler",             "team":"UNSOLD","data_source":"estimated_domestic"},
    "Piyush Chawla":       {"batting_avg":6.00, "strike_rate":72.00, "economy":8.00,"wickets_per_match":1.1,"role":"Bowler",             "team":"UNSOLD","data_source":"real_ipl"},
    "Utkarsh Singh":       {"batting_avg":16.00,"strike_rate":145.00,"economy":9.20,"wickets_per_match":0.6,"role":"All-Rounder",        "team":"UNSOLD","data_source":"estimated_domestic"},
    "Mayank Dagar":        {"batting_avg":14.00,"strike_rate":130.00,"economy":8.50,"wickets_per_match":0.8,"role":"All-Rounder",        "team":"UNSOLD","data_source":"estimated_domestic"},
    "Prince Chaudhary":    {"batting_avg":16.00,"strike_rate":142.00,"economy":9.00,"wickets_per_match":0.6,"role":"All-Rounder",        "team":"UNSOLD","data_source":"estimated_domestic"},

    # ══ SYNTHETIC DOMESTIC UNCAPPED INDIANS (622 players) ════════════════════

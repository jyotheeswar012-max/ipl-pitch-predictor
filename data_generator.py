import pandas as pd
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
REAL_PLAYER_STATS = {

    # ══ CHENNAI SUPER KINGS (CSK) ═══════════════════════════════════════════
    "Ruturaj Gaikwad":     {"batting_avg":35.00,"strike_rate":140.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"CSK","data_source":"real_ipl"},
    "Ayush Mhatre":        {"batting_avg":34.29,"strike_rate":188.97,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"CSK","data_source":"real_ipl"},
    "MS Dhoni":            {"batting_avg":24.00,"strike_rate":178.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"CSK","data_source":"real_ipl"},
    "Shivam Dube":         {"batting_avg":32.00,"strike_rate":162.00,"economy":9.80,"wickets_per_match":0.4,"role":"All-Rounder",        "team":"CSK","data_source":"real_ipl"},
    "Sanju Samson":        {"batting_avg":34.13,"strike_rate":151.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"CSK","data_source":"real_ipl"},
    "Khaleel Ahmed":       {"batting_avg":5.00, "strike_rate":68.00, "economy":8.90,"wickets_per_match":1.1,"role":"Bowler",             "team":"CSK","data_source":"real_ipl"},
    "Anshul Kamboj":       {"batting_avg":8.00, "strike_rate":85.00, "economy":9.30,"wickets_per_match":1.0,"role":"Bowler",             "team":"CSK","data_source":"estimated_domestic"},
    "Shreyas Gopal":       {"batting_avg":7.00, "strike_rate":80.00, "economy":8.40,"wickets_per_match":1.1,"role":"Bowler",             "team":"CSK","data_source":"estimated_domestic"},
    "Mukesh Choudhary":    {"batting_avg":5.00, "strike_rate":65.00, "economy":9.00,"wickets_per_match":1.0,"role":"Bowler",             "team":"CSK","data_source":"real_ipl"},
    "Kartik Sharma":       {"batting_avg":22.00,"strike_rate":165.00,"economy":8.80,"wickets_per_match":0.6,"role":"All-Rounder",        "team":"CSK","data_source":"estimated_domestic"},
    "Rahul Chahar":        {"batting_avg":6.00, "strike_rate":75.00, "economy":8.20,"wickets_per_match":1.2,"role":"Bowler",             "team":"CSK","data_source":"real_ipl"},
    "Sarfaraz Khan":       {"batting_avg":26.00,"strike_rate":162.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"CSK","data_source":"estimated_domestic"},
    "Ramakrishna Ghosh":   {"batting_avg":18.00,"strike_rate":150.00,"economy":8.50,"wickets_per_match":0.5,"role":"All-Rounder",        "team":"CSK","data_source":"role_based_fallback"},
    "Gurjapneet Singh":    {"batting_avg":5.00, "strike_rate":65.00, "economy":9.20,"wickets_per_match":0.9,"role":"Bowler",             "team":"CSK","data_source":"role_based_fallback"},

    # ══ MUMBAI INDIANS (MI) ════════════════════════════════════════════════
    "Rohit Sharma":        {"batting_avg":30.20,"strike_rate":139.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"MI","data_source":"real_ipl"},
    "Suryakumar Yadav":    {"batting_avg":65.18,"strike_rate":167.92,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"MI","data_source":"real_ipl"},
    "Tilak Varma":         {"batting_avg":34.50,"strike_rate":158.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"MI","data_source":"real_ipl"},
    "Hardik Pandya":       {"batting_avg":28.00,"strike_rate":148.00,"economy":9.20,"wickets_per_match":0.9,"role":"All-Rounder",        "team":"MI","data_source":"real_ipl"},
    "Jasprit Bumrah":      {"batting_avg":8.00, "strike_rate":95.00, "economy":6.67,"wickets_per_match":1.5,"role":"Bowler",             "team":"MI","data_source":"real_ipl"},
    "Deepak Chahar":       {"batting_avg":10.00,"strike_rate":95.00, "economy":8.20,"wickets_per_match":1.1,"role":"Bowler",             "team":"MI","data_source":"real_ipl"},
    "Naman Dhir":          {"batting_avg":25.20,"strike_rate":182.60,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"MI","data_source":"real_ipl"},
    "Shardul Thakur":      {"batting_avg":19.00,"strike_rate":152.00,"economy":9.60,"wickets_per_match":1.0,"role":"All-Rounder",        "team":"MI","data_source":"real_ipl"},
    "Robin Minz":          {"batting_avg":22.00,"strike_rate":155.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"MI","data_source":"estimated_domestic"},
    "Ashwani Kumar":       {"batting_avg":5.00, "strike_rate":65.00, "economy":9.30,"wickets_per_match":0.9,"role":"Bowler",             "team":"MI","data_source":"role_based_fallback"},
    "Raj Angad Bawa":      {"batting_avg":18.00,"strike_rate":148.00,"economy":9.00,"wickets_per_match":0.5,"role":"All-Rounder",        "team":"MI","data_source":"estimated_domestic"},
    "Raghu Sharma":        {"batting_avg":5.00, "strike_rate":65.00, "economy":8.80,"wickets_per_match":1.0,"role":"Bowler",             "team":"MI","data_source":"role_based_fallback"},

    # ══ KOLKATA KNIGHT RIDERS (KKR) ════════════════════════════════════════════════
    "Rinku Singh":         {"batting_avg":59.00,"strike_rate":149.50,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"KKR","data_source":"real_ipl"},
    "Angkrish Raghuvanshi":{"batting_avg":28.13,"strike_rate":162.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",           "team":"KKR","data_source":"real_ipl"},
    "Varun Chakravarthy":  {"batting_avg":5.00, "strike_rate":70.00, "economy":7.80,"wickets_per_match":1.2,"role":"Bowler",             "team":"KKR","data_source":"real_ipl"},
    "Harshit Rana":        {"batting_avg":8.00, "strike_rate":90.00, "economy":9.20,"wickets_per_match":1.1,"role":"Bowler",             "team":"KKR","data_source":"real_ipl"},
    "Vaibhav Arora":       {"batting_avg":5.00, "strike_rate":70.00, "economy":9.20,"wickets_per_match":1.1,"role":"Bowler",             "team":"KKR","data_source":"real_ipl"},
    "Ajinkya Rahane":      {"batting_avg":26.00,"strike_rate":130.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"KKR","data_source":"real_ipl"},
    "Ramandeep Singh":     {"batting_avg":22.00,"strike_rate":176.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"KKR","data_source":"real_ipl"},
    "Umran Malik":         {"batting_avg":4.00, "strike_rate":60.00, "economy":9.80,"wickets_per_match":1.0,"role":"Bowler",             "team":"KKR","data_source":"real_ipl"},
    "Ankul Roy":           {"batting_avg":12.00,"strike_rate":128.00,"economy":8.40,"wickets_per_match":0.8,"role":"All-Rounder",        "team":"KKR","data_source":"estimated_domestic"},
    "Rahul Tripathi":      {"batting_avg":24.00,"strike_rate":145.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"KKR","data_source":"real_ipl"},
    "Akashdeep":           {"batting_avg":5.00, "strike_rate":65.00, "economy":9.40,"wickets_per_match":1.0,"role":"Bowler",             "team":"KKR","data_source":"estimated_domestic"},

    # ══ ROYAL CHALLENGERS BENGALURU (RCB) ══════════════════════════════════════════════
    "Rajat Patidar":       {"batting_avg":35.00,"strike_rate":192.69,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"RCB","data_source":"real_ipl"},
    "Virat Kohli":         {"batting_avg":55.42,"strike_rate":148.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"RCB","data_source":"real_ipl"},
    "Devdutt Padikkal":    {"batting_avg":32.86,"strike_rate":148.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"RCB","data_source":"real_ipl"},
    "Jitesh Sharma":       {"batting_avg":29.00,"strike_rate":176.35,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"RCB","data_source":"real_ipl"},
    "Krunal Pandya":       {"batting_avg":23.00,"strike_rate":140.00,"economy":8.20,"wickets_per_match":1.1,"role":"All-Rounder",        "team":"RCB","data_source":"real_ipl"},
    "Bhuvneshwar Kumar":   {"batting_avg":8.00, "strike_rate":88.00, "economy":7.90,"wickets_per_match":1.1,"role":"Bowler",             "team":"RCB","data_source":"real_ipl"},
    "Yash Dayal":          {"batting_avg":5.00, "strike_rate":65.00, "economy":9.40,"wickets_per_match":1.0,"role":"Bowler",             "team":"RCB","data_source":"real_ipl"},
    "Venkatesh Iyer":      {"batting_avg":34.83,"strike_rate":186.60,"economy":9.50,"wickets_per_match":0.3,"role":"All-Rounder",        "team":"RCB","data_source":"real_ipl"},
    "Nuwan Thushara":      {"batting_avg":5.00, "strike_rate":65.00, "economy":8.70,"wickets_per_match":1.2,"role":"Bowler",             "team":"RCB","data_source":"estimated_domestic"},
    "Swapnil Singh":       {"batting_avg":14.00,"strike_rate":130.00,"economy":8.30,"wickets_per_match":0.9,"role":"All-Rounder",        "team":"RCB","data_source":"estimated_domestic"},
    "Rasikh Salam":        {"batting_avg":5.00, "strike_rate":65.00, "economy":9.10,"wickets_per_match":1.0,"role":"Bowler",             "team":"RCB","data_source":"estimated_domestic"},
    "Suyash Sharma":       {"batting_avg":5.00, "strike_rate":65.00, "economy":8.60,"wickets_per_match":1.0,"role":"Bowler",             "team":"RCB","data_source":"estimated_domestic"},

    # ══ PUNJAB KINGS (PBKS) ════════════════════════════════════════════════
    "Shreyas Iyer":        {"batting_avg":50.33,"strike_rate":175.07,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"real_ipl"},
    "Prabhsimran Singh":   {"batting_avg":32.29,"strike_rate":160.53,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"real_ipl"},
    "Priyansh Arya":       {"batting_avg":27.94,"strike_rate":211.62,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"real_ipl"},
    "Shashank Singh":      {"batting_avg":26.14,"strike_rate":188.57,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"real_ipl"},
    "Yuzvendra Chahal":    {"batting_avg":5.00, "strike_rate":75.00, "economy":7.91,"wickets_per_match":1.4,"role":"Bowler",             "team":"PBKS","data_source":"real_ipl"},
    "Arshdeep Singh":      {"batting_avg":6.00, "strike_rate":80.00, "economy":8.88,"wickets_per_match":1.3,"role":"Bowler",             "team":"PBKS","data_source":"real_ipl"},
    "Nehal Wadhera":       {"batting_avg":24.00,"strike_rate":162.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"real_ipl"},
    "Harpreet Brar":       {"batting_avg":12.00,"strike_rate":130.00,"economy":8.50,"wickets_per_match":0.9,"role":"All-Rounder",        "team":"PBKS","data_source":"real_ipl"},
    "Azmatullah Omarzai":  {"batting_avg":22.00,"strike_rate":158.00,"economy":9.10,"wickets_per_match":0.7,"role":"All-Rounder",        "team":"PBKS","data_source":"real_ipl"},
    "Musheer Khan":        {"batting_avg":20.00,"strike_rate":145.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"PBKS","data_source":"estimated_domestic"},
    "Vyshak Vijaykumar":   {"batting_avg":6.00, "strike_rate":70.00, "economy":9.30,"wickets_per_match":1.0,"role":"Bowler",             "team":"PBKS","data_source":"estimated_domestic"},
    "Vishnu Vinod":        {"batting_avg":22.00,"strike_rate":148.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"PBKS","data_source":"estimated_domestic"},

    # ══ GUJARAT TITANS (GT) ════════════════════════════════════════════════
    "Shubman Gill":        {"batting_avg":44.14,"strike_rate":159.27,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"GT","data_source":"real_ipl"},
    "Sai Sudharsan":       {"batting_avg":46.57,"strike_rate":157.86,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"GT","data_source":"real_ipl"},
    "Washington Sundar":   {"batting_avg":22.00,"strike_rate":140.00,"economy":7.90,"wickets_per_match":1.0,"role":"All-Rounder",        "team":"GT","data_source":"real_ipl"},
    "Mohammed Siraj":      {"batting_avg":6.00, "strike_rate":72.00, "economy":9.10,"wickets_per_match":1.2,"role":"Bowler",             "team":"GT","data_source":"real_ipl"},
    "Prasidh Krishna":     {"batting_avg":5.00, "strike_rate":72.00, "economy":9.10,"wickets_per_match":1.6,"role":"Bowler",             "team":"GT","data_source":"real_ipl"},
    "Sai Kishore":         {"batting_avg":6.00, "strike_rate":78.00, "economy":7.50,"wickets_per_match":1.2,"role":"Bowler",             "team":"GT","data_source":"real_ipl"},
    "Shahrukh Khan":       {"batting_avg":22.38,"strike_rate":179.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"GT","data_source":"real_ipl"},
    "Rahul Tewatia":       {"batting_avg":26.00,"strike_rate":155.00,"economy":8.80,"wickets_per_match":0.5,"role":"All-Rounder",        "team":"GT","data_source":"real_ipl"},
    "Nishant Sindhu":      {"batting_avg":16.00,"strike_rate":148.00,"economy":8.60,"wickets_per_match":0.7,"role":"All-Rounder",        "team":"GT","data_source":"estimated_domestic"},
    "Ishant Sharma":       {"batting_avg":5.00, "strike_rate":65.00, "economy":9.00,"wickets_per_match":0.9,"role":"Bowler",             "team":"GT","data_source":"real_ipl"},
    "Gurnoor Singh Brar":  {"batting_avg":6.00, "strike_rate":68.00, "economy":9.20,"wickets_per_match":1.0,"role":"Bowler",             "team":"GT","data_source":"estimated_domestic"},
    "Manav Suthar":        {"batting_avg":8.00, "strike_rate":80.00, "economy":8.20,"wickets_per_match":1.0,"role":"Bowler",             "team":"GT","data_source":"estimated_domestic"},
    "Jayant Yadav":        {"batting_avg":14.00,"strike_rate":130.00,"economy":8.30,"wickets_per_match":0.9,"role":"All-Rounder",        "team":"GT","data_source":"real_ipl"},
    "Kumar Kushagra":      {"batting_avg":20.00,"strike_rate":152.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"GT","data_source":"estimated_domestic"},
    "Anuj Rawat":          {"batting_avg":18.00,"strike_rate":145.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"GT","data_source":"estimated_domestic"},

    # ══ RAJASTHAN ROYALS (RR) ════════════════════════════════════════════════
    "Vaibhav Sooryavanshi":{"batting_avg":45.33,"strike_rate":237.30,"economy":9.80,"wickets_per_match":0.0,"role":"Batsman",            "team":"RR","data_source":"real_ipl"},
    "Yashasvi Jaiswal":    {"batting_avg":43.00,"strike_rate":159.71,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"RR","data_source":"real_ipl"},
    "Dhruv Jurel":         {"batting_avg":36.07,"strike_rate":155.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"RR","data_source":"real_ipl"},
    "Ravindra Jadeja":     {"batting_avg":66.50,"strike_rate":152.00,"economy":7.80,"wickets_per_match":1.2,"role":"All-Rounder",        "team":"RR","data_source":"real_ipl"},
    "Riyan Parag":         {"batting_avg":30.00,"strike_rate":158.00,"economy":8.60,"wickets_per_match":0.4,"role":"All-Rounder",        "team":"RR","data_source":"real_ipl"},
    "Ravi Bishnoi":        {"batting_avg":6.00, "strike_rate":72.00, "economy":8.00,"wickets_per_match":1.3,"role":"Bowler",             "team":"RR","data_source":"real_ipl"},
    "Shubham Dubey":       {"batting_avg":22.00,"strike_rate":155.00,"economy":8.80,"wickets_per_match":0.5,"role":"All-Rounder",        "team":"RR","data_source":"estimated_domestic"},
    "Yudhvir Singh Charak":{"batting_avg":8.00, "strike_rate":85.00, "economy":9.10,"wickets_per_match":1.1,"role":"Bowler",             "team":"RR","data_source":"estimated_domestic"},
    "Tushar Deshpande":    {"batting_avg":6.00, "strike_rate":68.00, "economy":9.40,"wickets_per_match":1.1,"role":"Bowler",             "team":"RR","data_source":"real_ipl"},

    # ══ SUNRISERS HYDERABAD (SRH) ════════════════════════════════════════════════
    "Abhishek Sharma":     {"batting_avg":33.77,"strike_rate":193.39,"economy":8.50,"wickets_per_match":0.3,"role":"All-Rounder",        "team":"SRH","data_source":"real_ipl"},
    "Ishan Kishan":        {"batting_avg":40.13,"strike_rate":182.42,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"SRH","data_source":"real_ipl"},
    "Nitish Kumar Reddy":  {"batting_avg":26.00,"strike_rate":158.00,"economy":9.20,"wickets_per_match":0.6,"role":"All-Rounder",        "team":"SRH","data_source":"real_ipl"},
    "Harshal Patel":       {"batting_avg":8.00, "strike_rate":88.00, "economy":8.70,"wickets_per_match":1.2,"role":"Bowler",             "team":"SRH","data_source":"real_ipl"},
    "Jaydev Unadkat":      {"batting_avg":8.00, "strike_rate":80.00, "economy":8.60,"wickets_per_match":1.0,"role":"Bowler",             "team":"SRH","data_source":"real_ipl"},
    "Harsh Dubey":         {"batting_avg":6.00, "strike_rate":70.00, "economy":8.30,"wickets_per_match":1.1,"role":"Bowler",             "team":"SRH","data_source":"estimated_domestic"},
    "Aniket Verma":        {"batting_avg":20.00,"strike_rate":158.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"SRH","data_source":"estimated_domestic"},
    "R Smaran":            {"batting_avg":22.00,"strike_rate":150.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"SRH","data_source":"estimated_domestic"},

    # ══ LUCKNOW SUPER GIANTS (LSG) ═══════════════════════════════════════════════
    "Rishabh Pant":        {"batting_avg":35.00,"strike_rate":162.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"LSG","data_source":"real_ipl"},
    "Mohammed Shami":      {"batting_avg":7.00, "strike_rate":90.00, "economy":8.50,"wickets_per_match":1.2,"role":"Bowler",             "team":"LSG","data_source":"real_ipl"},
    "Mayank Yadav":        {"batting_avg":5.00, "strike_rate":65.00, "economy":8.30,"wickets_per_match":1.3,"role":"Bowler",             "team":"LSG","data_source":"real_ipl"},
    "Avesh Khan":          {"batting_avg":6.00, "strike_rate":70.00, "economy":9.30,"wickets_per_match":1.1,"role":"Bowler",             "team":"LSG","data_source":"real_ipl"},
    "Mohsin Khan":         {"batting_avg":5.00, "strike_rate":65.00, "economy":8.70,"wickets_per_match":1.1,"role":"Bowler",             "team":"LSG","data_source":"real_ipl"},
    "Abdul Samad":         {"batting_avg":24.00,"strike_rate":170.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"LSG","data_source":"real_ipl"},
    "Ayush Badoni":        {"batting_avg":26.00,"strike_rate":155.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"LSG","data_source":"real_ipl"},
    "Himmat Singh":        {"batting_avg":20.00,"strike_rate":148.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"LSG","data_source":"estimated_domestic"},
    "Arshin Kulkarni":     {"batting_avg":16.00,"strike_rate":145.00,"economy":8.70,"wickets_per_match":0.7,"role":"All-Rounder",        "team":"LSG","data_source":"estimated_domestic"},
    "Digvesh Rathi":       {"batting_avg":6.00, "strike_rate":68.00, "economy":8.50,"wickets_per_match":1.0,"role":"Bowler",             "team":"LSG","data_source":"estimated_domestic"},

    # ══ DELHI CAPITALS (DC) ════════════════════════════════════════════════
    "Axar Patel":          {"batting_avg":24.00,"strike_rate":153.00,"economy":8.10,"wickets_per_match":1.1,"role":"All-Rounder",        "team":"DC","data_source":"real_ipl"},
    "KL Rahul":            {"batting_avg":45.62,"strike_rate":174.41,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"DC","data_source":"real_ipl"},
    "Kuldeep Yadav":       {"batting_avg":6.00, "strike_rate":80.00, "economy":8.10,"wickets_per_match":1.3,"role":"Bowler",             "team":"DC","data_source":"real_ipl"},
    "T Natarajan":         {"batting_avg":5.00, "strike_rate":65.00, "economy":8.80,"wickets_per_match":1.1,"role":"Bowler",             "team":"DC","data_source":"real_ipl"},
    "Karun Nair":          {"batting_avg":22.00,"strike_rate":145.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"DC","data_source":"real_ipl"},
    "Ashutosh Sharma":     {"batting_avg":24.43,"strike_rate":181.91,"economy":9.20,"wickets_per_match":0.3,"role":"All-Rounder",        "team":"DC","data_source":"real_ipl"},
    "Vipraj Nigam":        {"batting_avg":20.29,"strike_rate":179.74,"economy":8.40,"wickets_per_match":0.8,"role":"All-Rounder",        "team":"DC","data_source":"real_ipl"},
    "Sameer Rizvi":        {"batting_avg":22.00,"strike_rate":165.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"DC","data_source":"real_ipl"},
    "Abishek Porel":       {"batting_avg":20.00,"strike_rate":152.00,"economy":9.50,"wickets_per_match":0.0,"role":"Wicketkeeper-Batter","team":"DC","data_source":"estimated_domestic"},
    "Nitish Rana":         {"batting_avg":24.00,"strike_rate":145.00,"economy":8.50,"wickets_per_match":0.3,"role":"All-Rounder",        "team":"DC","data_source":"real_ipl"},
    "Mukesh Kumar":        {"batting_avg":5.00, "strike_rate":65.00, "economy":9.10,"wickets_per_match":1.1,"role":"Bowler",             "team":"DC","data_source":"real_ipl"},
    "Prithvi Shaw":        {"batting_avg":26.00,"strike_rate":160.00,"economy":9.50,"wickets_per_match":0.0,"role":"Batsman",            "team":"DC","data_source":"real_ipl"},
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

# ── Indian players only ───────────────────────────────────────────────────────────────
PLAYER_MODIFIERS = {
    "Vaibhav Sooryavanshi": {"Flat/Batting": 0.20, "Hard/True Bounce": 0.18},
    "Virat Kohli":          {"Flat/Batting": 0.15, "Slow/Dry Spin": 0.12},
    "Suryakumar Yadav":     {"Flat/Batting": 0.18, "Hard/True Bounce": 0.15},
    "Sai Sudharsan":        {"Flat/Batting": 0.14, "Slow/Dry Spin": 0.10},
    "Shreyas Iyer":         {"Flat/Batting": 0.13, "Hard/True Bounce": 0.08},
    "Yashasvi Jaiswal":     {"Flat/Batting": 0.12, "Hard/True Bounce": 0.10},
    "Rohit Sharma":         {"Flat/Batting": 0.12, "Hard/True Bounce": 0.08},
    "Shubman Gill":         {"Flat/Batting": 0.13, "Hard/True Bounce": 0.09},
    "Priyansh Arya":        {"Flat/Batting": 0.16, "Hard/True Bounce": 0.12},
    "Rinku Singh":          {"Wet/Dew Heavy": 0.14, "Flat/Batting": 0.10},
    "KL Rahul":             {"Flat/Batting": 0.12, "Hard/True Bounce": 0.10},
    "Ishan Kishan":         {"Flat/Batting": 0.13, "Wet/Dew Heavy": 0.10},
    "Rishabh Pant":         {"Flat/Batting": 0.14, "Hard/True Bounce": 0.10},
    "Abhishek Sharma":      {"Flat/Batting": 0.12, "Hard/True Bounce": 0.10},
    "Rajat Patidar":        {"Flat/Batting": 0.14, "Hard/True Bounce": 0.10},
    "MS Dhoni":             {"Wet/Dew Heavy": 0.16, "Flat/Batting": 0.08},
    "Jasprit Bumrah":       {"Green/Grassy": 0.22, "Hard/True Bounce": 0.20, "Wet/Dew Heavy": 0.18},
    "Arshdeep Singh":       {"Green/Grassy": 0.14, "Wet/Dew Heavy": 0.18},
    "Yuzvendra Chahal":     {"Slow/Dry Spin": 0.22, "Flat/Batting": 0.06},
    "Varun Chakravarthy":   {"Slow/Dry Spin": 0.20, "Flat/Batting": 0.06},
    "Kuldeep Yadav":        {"Slow/Dry Spin": 0.18, "Flat/Batting": 0.08},
    "Bhuvneshwar Kumar":    {"Green/Grassy": 0.20, "Wet/Dew Heavy": 0.14},
    "Mohammed Shami":       {"Green/Grassy": 0.18, "Hard/True Bounce": 0.15},
    "Prasidh Krishna":      {"Green/Grassy": 0.16, "Hard/True Bounce": 0.14},
    "Sai Kishore":          {"Slow/Dry Spin": 0.16, "Flat/Batting": 0.06},
    "Ravi Bishnoi":         {"Slow/Dry Spin": 0.20, "Flat/Batting": 0.06},
    "Mayank Yadav":         {"Green/Grassy": 0.14, "Hard/True Bounce": 0.12},
    "Harshit Rana":         {"Green/Grassy": 0.12, "Hard/True Bounce": 0.10},
    "Harshal Patel":        {"Wet/Dew Heavy": 0.14, "Slow/Dry Spin": 0.08},
    "Mohammed Siraj":       {"Green/Grassy": 0.14, "Hard/True Bounce": 0.12},
    "Ravindra Jadeja":      {"Slow/Dry Spin": 0.20, "Flat/Batting": 0.10},
    "Hardik Pandya":        {"Flat/Batting": 0.10, "Green/Grassy": 0.12},
    "Axar Patel":           {"Slow/Dry Spin": 0.18, "Flat/Batting": 0.08},
    "Washington Sundar":    {"Slow/Dry Spin": 0.14, "Flat/Batting": 0.06},
    "Venkatesh Iyer":       {"Flat/Batting": 0.14, "Hard/True Bounce": 0.10},
    "Shivam Dube":          {"Flat/Batting": 0.10, "Hard/True Bounce": 0.08},
    "Krunal Pandya":        {"Slow/Dry Spin": 0.12, "Flat/Batting": 0.06},
    "Riyan Parag":          {"Slow/Dry Spin": 0.10, "Flat/Batting": 0.08},
    "Nitish Kumar Reddy":   {"Flat/Batting": 0.10, "Hard/True Bounce": 0.08},
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
    return pd.DataFrame(rows)

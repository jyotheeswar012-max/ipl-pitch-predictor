import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder


def train_model(df: pd.DataFrame):
    """Train DecisionTree regressor on suitability_score."""
    le_pitch = LabelEncoder()
    le_venue = LabelEncoder()
    le_toss  = LabelEncoder()
    le_role  = LabelEncoder()

    df = df.copy()
    df['pitch_encoded'] = le_pitch.fit_transform(df['pitch_type'])
    df['venue_encoded'] = le_venue.fit_transform(df['venue'])
    df['toss_encoded']  = le_toss.fit_transform(df['toss'])
    df['role_encoded']  = le_role.fit_transform(df['role'])

    feature_cols = [
        'batting_avg', 'strike_rate', 'economy', 'wickets_per_match',
        'pitch_encoded', 'venue_encoded', 'toss_encoded', 'role_encoded'
    ]

    X = df[feature_cols]
    y = df['suitability_score']

    model = DecisionTreeRegressor(
        max_depth=12, random_state=42
    )
    model.fit(X, y)

    label_encoders = {
        'pitch': le_pitch,
        'venue': le_venue,
        'toss':  le_toss,
        'role':  le_role,
    }
    return model, label_encoders, feature_cols


# FIX 3: Toss modifier — Bat First favours bowlers early,
# Chase favours batters/WKs (settled pitch + dew effect)
TOSS_ROLE_MODIFIER = {
    "Bat First": {"Batsman": 2, "Wicketkeeper-Batter": 1, "All-Rounder": 1, "Bowler": 3},
    "Chase":     {"Batsman": 3, "Wicketkeeper-Batter": 3, "All-Rounder": 2, "Bowler": 0},
}


def predict_best_players(
    df: pd.DataFrame,
    model,
    label_encoders: dict,
    feature_cols: list,
    pitch_type: str,
    venue: str,
    toss: str,
    top_n: int = 10,
    role_filter: list = None,
):
    from data_generator import _reason, REAL_PLAYER_STATS, VENUE_PITCH_MODIFIERS

    # FIX 2: filter by selected pitch_type so stats are pitch-relevant
    pitch_df = df[df['pitch_type'] == pitch_type]
    agg = pitch_df.groupby(['player_name', 'role', 'team'], as_index=False).agg(
        batting_avg=('batting_avg', 'mean'),
        strike_rate=('strike_rate', 'mean'),
        economy=('economy', 'mean'),
        wickets_per_match=('wickets_per_match', 'mean'),
    )

    agg['data_source'] = agg['player_name'].apply(
        lambda n: REAL_PLAYER_STATS.get(n, {}).get('data_source', 'real_ipl')
    )

    if role_filter:
        agg = agg[agg['role'].isin(role_filter)]

    if agg.empty:
        return agg

    le_p = label_encoders['pitch']
    le_v = label_encoders['venue']
    le_t = label_encoders['toss']
    le_r = label_encoders['role']

    try:
        p_enc = le_p.transform([pitch_type])[0]
    except ValueError:
        p_enc = 0
    try:
        v_enc = le_v.transform([venue])[0]
    except ValueError:
        v_enc = 0
    try:
        t_enc = le_t.transform([toss])[0]
    except ValueError:
        t_enc = 0

    rows = []
    for _, row in agg.iterrows():
        try:
            r_enc = le_r.transform([row['role']])[0]
        except ValueError:
            r_enc = 0

        features = pd.DataFrame([[
            row['batting_avg'], row['strike_rate'], row['economy'],
            row['wickets_per_match'], p_enc, v_enc, t_enc, r_enc
        ]], columns=feature_cols)

        pred_score = float(model.predict(features)[0])

        # FIX 1: player_mod already learned by model during training — do NOT add again
        # FIX 3: apply toss modifier so Bat First / Chase actually changes scores
        venue_mod = VENUE_PITCH_MODIFIERS.get(venue, {}).get(pitch_type, {}).get(row['role'], 0)
        toss_mod  = TOSS_ROLE_MODIFIER.get(toss, {}).get(row['role'], 0)
        final_score = round(float(np.clip(pred_score + venue_mod + toss_mod, 0, 100)), 2)

        reason = _reason(
            row['player_name'], row['role'], pitch_type,
            final_score, row['batting_avg'], row['strike_rate'], row['economy']
        )

        rows.append({
            'player_name':       row['player_name'],
            'role':              row['role'],
            'team':              row['team'],
            'data_source':       row['data_source'],
            'batting_avg':       round(row['batting_avg'], 2),
            'strike_rate':       round(row['strike_rate'], 2),
            'economy':           round(row['economy'], 2),
            'wickets_per_match': round(row['wickets_per_match'], 2),
            'suitability_score': final_score,
            'reason':            reason,
        })

    result = (
        pd.DataFrame(rows)
        .sort_values('suitability_score', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return result

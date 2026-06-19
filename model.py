import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def train_model(df: pd.DataFrame):
    """Train RandomForest on suitability_score."""
    le_pitch  = LabelEncoder()
    le_venue  = LabelEncoder()
    le_toss   = LabelEncoder()
    le_role   = LabelEncoder()

    df = df.copy()
    df['pitch_encoded'] = le_pitch.fit_transform(df['pitch_type'])
    df['venue_encoded'] = le_venue.fit_transform(df['venue'])
    df['toss_encoded']  = le_toss.fit_transform(df['toss'])
    df['role_encoded']  = le_role.fit_transform(df['role'])

    feature_cols = ['batting_avg', 'strike_rate', 'economy', 'wickets_per_match',
                    'pitch_encoded', 'venue_encoded', 'toss_encoded', 'role_encoded']

    X = df[feature_cols]
    y = df['suitability_score']

    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X, y)

    label_encoders = {
        'pitch':  le_pitch,
        'venue':  le_venue,
        'toss':   le_toss,
        'role':   le_role,
    }
    return model, label_encoders, feature_cols


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
    """Predict suitability for all players given the conditions."""
    from data_generator import PLAYERS, PITCH_ROLE_WEIGHTS, PLAYER_MODIFIERS, _reason

    # Aggregate player-level stats from historical data
    agg = df.groupby(['player_name', 'role', 'team']).agg(
        batting_avg=('batting_avg', 'mean'),
        strike_rate=('strike_rate', 'mean'),
        economy=('economy', 'mean'),
        wickets_per_match=('wickets_per_match', 'mean'),
    ).reset_index()

    # Apply role filter
    if role_filter:
        agg = agg[agg['role'].isin(role_filter)]

    if agg.empty:
        return agg

    # Encode input conditions
    le_p = label_encoders['pitch']
    le_v = label_encoders['venue']
    le_t = label_encoders['toss']
    le_r = label_encoders['role']

    try:
        p_enc = le_p.transform([pitch_type])[0]
        v_enc = le_v.transform([venue])[0]
        t_enc = le_t.transform([toss])[0]
    except ValueError:
        p_enc, v_enc, t_enc = 0, 0, 0

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

        pred_score = model.predict(features)[0]

        # Add player-specific modifier
        player_mod = PLAYER_MODIFIERS.get(row['player_name'], {}).get(pitch_type, 0.0)
        final_score = round(np.clip(pred_score + player_mod * 100, 0, 100), 2)

        reason = _reason(
            row['player_name'], row['role'], pitch_type,
            final_score, row['batting_avg'], row['strike_rate'], row['economy']
        )

        rows.append({
            **row.to_dict(),
            'suitability_score': final_score,
            'reason': reason,
        })

    result = pd.DataFrame(rows).sort_values('suitability_score', ascending=False).head(top_n).reset_index(drop=True)
    return result

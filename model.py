import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


TOSS_ROLE_MODIFIER = {
    "Bat First": {"Batsman": 2, "Wicketkeeper-Batter": 1, "All-Rounder": 1, "Bowler": 3},
    "Chase":     {"Batsman": 3, "Wicketkeeper-Batter": 3, "All-Rounder": 2, "Bowler": 0},
}


def train_model(df: pd.DataFrame):
    """Train a VotingRegressor ensemble of RandomForest + GradientBoosting."""
    df = df.copy()

    le_pitch = LabelEncoder()
    le_venue = LabelEncoder()
    le_toss  = LabelEncoder()
    le_role  = LabelEncoder()
    le_team  = LabelEncoder()

    df["pitch_enc"] = le_pitch.fit_transform(df["pitch_type"])
    df["venue_enc"] = le_venue.fit_transform(df["venue"])
    df["toss_enc"]  = le_toss.fit_transform(df["toss"])
    df["role_enc"]  = le_role.fit_transform(df["role"])
    df["team_enc"]  = le_team.fit_transform(df["team"])

    # engineered features
    df["avg_sr_product"] = df["batting_avg"] * df["strike_rate"] / 1000
    df["bowling_impact"] = df["wickets_per_match"] / (df["economy"] + 0.01)
    df["is_synth"]       = (df["data_source"] == "synthetic_domestic").astype(int)

    feature_cols = [
        "batting_avg", "strike_rate", "economy", "wickets_per_match",
        "pitch_enc", "venue_enc", "toss_enc", "role_enc", "team_enc",
        "avg_sr_product", "bowling_impact", "is_synth",
    ]

    X = df[feature_cols]
    y = df["suitability_score"]

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=14, min_samples_leaf=4,
        max_features="sqrt", n_jobs=-1, random_state=42,
    )
    gb = GradientBoostingRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.08,
        subsample=0.8, min_samples_leaf=4, random_state=42,
    )

    ensemble = VotingRegressor(estimators=[("rf", rf), ("gb", gb)])
    ensemble.fit(X, y)

    y_pred = ensemble.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2  = r2_score(y, y_pred)
    print(f"[Model] Train MAE={mae:.3f}  R\u00b2={r2:.4f}  rows={len(df)}")

    label_encoders = {
        "pitch": le_pitch, "venue": le_venue,
        "toss":  le_toss,  "role":  le_role, "team": le_team,
    }
    return ensemble, label_encoders, feature_cols


def _safe_enc(le, val):
    try:
        return le.transform([val])[0]
    except ValueError:
        return 0


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

    pitch_df = df[df["pitch_type"] == pitch_type]
    agg = pitch_df.groupby(["player_name", "role", "team", "data_source"], as_index=False).agg(
        batting_avg=("batting_avg", "mean"),
        strike_rate=("strike_rate", "mean"),
        economy=("economy", "mean"),
        wickets_per_match=("wickets_per_match", "mean"),
    )

    if role_filter:
        agg = agg[agg["role"].isin(role_filter)]
    if agg.empty:
        return agg

    le_p  = label_encoders["pitch"]
    le_v  = label_encoders["venue"]
    le_t  = label_encoders["toss"]
    le_r  = label_encoders["role"]
    le_tm = label_encoders["team"]

    p_enc = _safe_enc(le_p, pitch_type)
    v_enc = _safe_enc(le_v, venue)
    t_enc = _safe_enc(le_t, toss)

    rows = []
    for _, row in agg.iterrows():
        r_enc  = _safe_enc(le_r,  row["role"])
        tm_enc = _safe_enc(le_tm, row["team"])
        avg, sr, econ, wpm = (
            row["batting_avg"], row["strike_rate"],
            row["economy"],     row["wickets_per_match"],
        )
        is_synth = 1 if row["data_source"] == "synthetic_domestic" else 0

        feat = pd.DataFrame([[
            avg, sr, econ, wpm,
            p_enc, v_enc, t_enc, r_enc, tm_enc,
            avg * sr / 1000,
            wpm / (econ + 0.01),
            is_synth,
        ]], columns=feature_cols)

        pred = float(model.predict(feat)[0])
        venue_mod = VENUE_PITCH_MODIFIERS.get(venue, {}).get(pitch_type, {}).get(row["role"], 0)
        toss_mod  = TOSS_ROLE_MODIFIER.get(toss, {}).get(row["role"], 0)
        final     = round(float(np.clip(pred + venue_mod + toss_mod, 0, 100)), 2)

        reason = _reason(
            row["player_name"], row["role"], pitch_type,
            final, avg, sr, econ,
        )
        rows.append({
            "player_name":       row["player_name"],
            "role":              row["role"],
            "team":              row["team"],
            "data_source":       row["data_source"],
            "batting_avg":       round(avg,  2),
            "strike_rate":       round(sr,   2),
            "economy":           round(econ, 2),
            "wickets_per_match": round(wpm,  2),
            "suitability_score": final,
            "reason":            reason,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("suitability_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

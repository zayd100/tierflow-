"""
generate_synthetic.py
─────────────────────
Generates a realistic synthetic contact_attempts dataset for TierFlow.
Use this to test the pipeline before you have real data.

The generator bakes in plausible patterns:
  - Higher connect rates on Tue/Wed/Thu, 9-11am and 2-4pm lead local time
  - Tier 1 leads answer more often than Tier 3
  - SaaS and Finance leads respond better during business hours
  - Healthcare leads have slightly better evening windows
  - Meeting bookings cluster around already-answered calls with high scores

Run:
    python generate_synthetic.py
    python generate_synthetic.py --rows 5000 --seed 99
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

INDUSTRIES = ["SaaS", "Healthcare", "Finance", "Retail", "Logistics"]
TIMEZONES  = [
    "Asia/Karachi", "America/New_York", "America/Chicago",
    "America/Los_Angeles", "Europe/London", "Europe/Berlin",
]
CHANNELS   = ["call", "email", "call", "call", "sms"]   # weighted toward call
ROLES      = ["Warmer", "Warmer", "Closer"]              # 2:1 warmers

N_REPS  = 12
N_LEADS = 400


def connect_probability(dow: int, hour: int, industry: str, tier: int) -> float:
    """
    Rough probability of a 'connect' (answered or meeting_booked) outcome,
    based on baked-in patterns.
    """
    # Base by hour (lead local time)
    hour_base = {
        range(0,  8): 0.03,
        range(8,  9): 0.18,
        range(9, 12): 0.45,
        range(12,13): 0.22,
        range(13,15): 0.35,
        range(15,17): 0.42,
        range(17,19): 0.20,
        range(19,24): 0.07,
    }
    base = 0.1
    for hr_range, prob in hour_base.items():
        if hour in hr_range:
            base = prob
            break

    # Day multiplier
    day_mult = {0: 1.0, 1: 1.2, 2: 1.25, 3: 1.15, 4: 0.85, 5: 0.4, 6: 0.25}
    base *= day_mult.get(dow, 1.0)

    # Industry tweak
    ind_mult = {
        "SaaS":       1.1,
        "Finance":    1.05,
        "Healthcare": 0.95,
        "Retail":     0.85,
        "Logistics":  0.9,
    }
    base *= ind_mult.get(industry, 1.0)

    # Tier: tier 1 connects easier
    tier_mult = {1: 1.3, 2: 1.0, 3: 0.65}
    base *= tier_mult.get(tier, 1.0)

    return min(max(base, 0.01), 0.98)


def meeting_given_connect(score: int, tier: int) -> float:
    """Probability a connect turns into a meeting booked."""
    return min(0.05 + (score / 100) * 0.55 + (4 - tier) * 0.08, 0.85)


def generate(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    rep_ids   = [f"R{i:02d}" for i in range(1, N_REPS + 1)]
    rep_roles = {r: random.choice(ROLES) for r in rep_ids}
    lead_ids  = [f"L{i:03d}" for i in range(1, N_LEADS + 1)]
    lead_meta = {
        lid: {
            "industry":  rng.choice(INDUSTRIES),
            "tier":      int(rng.choice([1, 2, 3], p=[0.25, 0.40, 0.35])),
            "score":     int(rng.integers(20, 99)),
            "timezone":  rng.choice(TIMEZONES),
        }
        for lid in lead_ids
    }

    rows = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end   = pd.Timestamp("2024-06-30", tz="UTC")

    for i in range(1, n_rows + 1):
        lead  = rng.choice(lead_ids)
        meta  = lead_meta[lead]
        rep   = rng.choice(rep_ids)

        # Random UTC timestamp
        delta_s = rng.integers(0, int((end - start).total_seconds()))
        ts_utc  = start + pd.Timedelta(seconds=int(delta_s))

        # Lead local time
        tz       = pytz.timezone(meta["timezone"])
        local_dt = ts_utc.astimezone(tz)
        dow      = local_dt.weekday()
        hour     = local_dt.hour

        # Determine outcome
        p_connect = connect_probability(dow, hour, meta["industry"], meta["tier"])
        connected = rng.random() < p_connect

        if connected:
            p_meeting = meeting_given_connect(meta["score"], meta["tier"])
            if rng.random() < p_meeting:
                outcome  = "meeting_booked"
                duration = int(rng.integers(300, 700))
            else:
                outcome  = "answered"
                duration = int(rng.integers(60, 420))
        else:
            outcome  = rng.choice(["no_reply", "voicemail", "no_reply", "bounced"],
                                  p=[0.50, 0.35, 0.10, 0.05])
            duration = 0

        rows.append({
            "attempt_id":       i,
            "timestamp_utc":    ts_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "lead_id":          lead,
            "rep_id":           rep,
            "rep_role":         rep_roles[rep],
            "contact_channel":  rng.choice(CHANNELS),
            "industry":         meta["industry"],
            "lead_tier":        meta["tier"],
            "lead_score":       meta["score"],
            "outcome":          outcome,
            "duration_seconds": duration,
            "lead_timezone":    meta["timezone"],
            "notes":            "",
        })

    df = pd.DataFrame(rows)
    out = DATA_DIR / "contact_attempts_synthetic.csv"
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} rows → {out}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default=2000, type=int)
    p.add_argument("--seed", default=42,   type=int)
    args = p.parse_args()
    generate(args.rows, args.seed)

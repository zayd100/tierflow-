"""
TierFlow — Contact Time Heatmap Pipeline
=========================================
Loads contact attempt data, localises timestamps to lead timezone,
computes connect_rate and meeting_rate per (day, hour) cell,
applies a minimum-sample confidence mask, then renders heatmaps
segmented by industry and lead tier.

Run:
    python heatmap_pipeline.py
    python heatmap_pipeline.py --metric meeting_rate
    python heatmap_pipeline.py --industry SaaS --tier 1
    python heatmap_pipeline.py --min-samples 5 --metric connect_rate
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pytz

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
OUTPUT_DIR  = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DAY_LABELS  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HOUR_LABELS = [f"{h:02d}:00" for h in range(24)]

POSITIVE_OUTCOMES = {
    "connect_rate":  {"answered", "meeting_booked"},
    "meeting_rate":  {"meeting_booked"},
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: Path) -> pd.DataFrame:
    """Load CSV and run basic schema validation."""
    required = {
        "attempt_id", "timestamp_utc", "lead_id", "rep_id", "rep_role",
        "contact_channel", "industry", "lead_tier", "lead_score",
        "outcome", "duration_seconds", "lead_timezone",
    }
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["lead_tier"]     = df["lead_tier"].astype(int)
    df["lead_score"]    = pd.to_numeric(df["lead_score"], errors="coerce")
    df["outcome"]       = df["outcome"].str.strip().str.lower()
    df["industry"]      = df["industry"].str.strip()

    print(f"[load]  {len(df)} rows | "
          f"{df['industry'].nunique()} industries | "
          f"{df['lead_tier'].nunique()} tiers | "
          f"{df['outcome'].nunique()} outcome types")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. TIMEZONE LOCALISATION
# ══════════════════════════════════════════════════════════════════════════════

def localise_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive lead_local_hour (0–23) and lead_local_dow (0=Mon … 6=Sun)
    from timestamp_utc + lead_timezone.

    Rows with unrecognised timezones get NaT and are dropped with a warning.
    """
    local_hours, local_dows = [], []

    for _, row in df.iterrows():
        try:
            tz   = pytz.timezone(row["lead_timezone"])
            local_dt = row["timestamp_utc"].astimezone(tz)
            local_hours.append(local_dt.hour)
            local_dows.append(local_dt.weekday())   # 0 = Monday
        except Exception:
            local_hours.append(np.nan)
            local_dows.append(np.nan)

    df = df.copy()
    df["lead_local_hour"] = local_hours
    df["lead_local_dow"]  = local_dows

    bad = df["lead_local_hour"].isna().sum()
    if bad:
        print(f"[tz]    WARNING: {bad} rows had unrecognised timezones — dropped")
    df = df.dropna(subset=["lead_local_hour", "lead_local_dow"])
    df["lead_local_hour"] = df["lead_local_hour"].astype(int)
    df["lead_local_dow"]  = df["lead_local_dow"].astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def add_outcome_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag columns for each metric type."""
    df = df.copy()
    for metric, good_outcomes in POSITIVE_OUTCOMES.items():
        df[metric] = df["outcome"].isin(good_outcomes).astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. PIVOT TABLE (the heatmap grid)
# ══════════════════════════════════════════════════════════════════════════════

def build_pivot(
    df: pd.DataFrame,
    metric: str,
    min_samples: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        rate_pivot  — (7 days × 24 hours) mean rate, NaN where < min_samples
        count_pivot — (7 days × 24 hours) attempt counts (for annotation)
    """
    grouped = df.groupby(["lead_local_dow", "lead_local_hour"])

    rate_raw  = grouped[metric].mean().unstack(fill_value=np.nan)
    count_raw = grouped[metric].count().unstack(fill_value=0)

    # Reindex to full 7×24 grid
    rate_pivot  = rate_raw.reindex(index=range(7), columns=range(24))
    count_pivot = count_raw.reindex(index=range(7), columns=range(24), fill_value=0)

    # Mask cells below minimum sample threshold
    rate_pivot[count_pivot < min_samples] = np.nan

    return rate_pivot, count_pivot


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(
    rate_pivot: pd.DataFrame,
    count_pivot: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    annotate_counts: bool = True,
) -> None:
    """Render a styled 7×24 heatmap and save to disk."""

    fig, ax = plt.subplots(figsize=(18, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Custom colourmap: dark grey (low/no data) → amber → coral
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "tierflow",
        ["#1e2130", "#f0a500", "#e84040"],
        N=256,
    )
    cmap.set_bad(color="#1a1d27")   # NaN / masked cells

    rate_matrix  = rate_pivot.values
    count_matrix = count_pivot.reindex(
        index=rate_pivot.index, columns=rate_pivot.columns, fill_value=0
    ).values

    masked = np.ma.masked_invalid(rate_matrix)

    im = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Gridlines
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color="#2a2d3a", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Axis labels
    ax.set_xticks(range(24))
    ax.set_xticklabels(HOUR_LABELS, rotation=45, ha="right",
                       fontsize=8, color="#9ca3af")
    ax.set_yticks(range(7))
    ax.set_yticklabels(DAY_LABELS, fontsize=9, color="#9ca3af")

    # Cell annotations
    if annotate_counts:
        for r in range(7):
            for c in range(24):
                rate_val  = rate_matrix[r, c]
                count_val = int(count_matrix[r, c])
                if np.isnan(rate_val) or count_val == 0:
                    continue
                label = f"{rate_val:.0%}\n({count_val})"
                color = "white" if rate_val > 0.55 else "#9ca3af"
                ax.text(c, r, label, ha="center", va="center",
                        fontsize=6.5, color=color, weight="500")

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.ax.yaxis.set_tick_params(color="#9ca3af")
    cbar.outline.set_edgecolor("#2a2d3a")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#9ca3af", fontsize=8)
    cbar.set_label(metric.replace("_", " ").title(), color="#9ca3af", fontsize=9)

    # Title & subtitle
    metric_label = metric.replace("_", " ").title()
    ax.set_title(
        f"{title}  —  {metric_label}  (lead's local time)",
        color="white", fontsize=12, pad=14, weight="500",
    )
    ax.set_xlabel("Lead's local hour", color="#9ca3af", fontsize=9)
    ax.set_ylabel("Day of week", color="#9ca3af", fontsize=9)

    note = "Grey cells = fewer than min_samples attempts (statistically unreliable)"
    fig.text(0.01, -0.04, note, color="#6b7280", fontsize=7.5, ha="left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot]  Saved → {output_path.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY STATS  (top 5 windows)
# ══════════════════════════════════════════════════════════════════════════════

def top_windows(
    rate_pivot: pd.DataFrame,
    count_pivot: pd.DataFrame,
    metric: str,
    n: int = 5,
) -> pd.DataFrame:
    """Return the top-n (day, hour) cells sorted by rate descending."""
    rows = []
    for dow in range(7):
        for hour in range(24):
            rate  = rate_pivot.iloc[dow, hour] if hour in rate_pivot.columns else np.nan
            count = count_pivot.iloc[dow, hour] if hour in count_pivot.columns else 0
            if not np.isnan(rate):
                rows.append({
                    "day":    DAY_LABELS[dow],
                    "hour":   f"{hour:02d}:00",
                    metric:   round(rate, 3),
                    "n":      int(count),
                })
    df = pd.DataFrame(rows).sort_values(metric, ascending=False).head(n)
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def run(
    csv_path: Path,
    metric: str        = "connect_rate",
    industry_filter: str | None = None,
    tier_filter: int | None     = None,
    min_samples: int   = 10,
    annotate: bool     = True,
) -> None:

    # ── Load ──
    df = load_data(csv_path)
    df = localise_timestamps(df)
    df = add_outcome_flags(df)

    # ── Optional filters ──
    label_parts = []
    if industry_filter:
        df = df[df["industry"].str.lower() == industry_filter.lower()]
        label_parts.append(industry_filter)
        print(f"[filter] industry = {industry_filter}  ({len(df)} rows remaining)")
    if tier_filter is not None:
        df = df[df["lead_tier"] == tier_filter]
        label_parts.append(f"Tier {tier_filter}")
        print(f"[filter] tier = {tier_filter}  ({len(df)} rows remaining)")

    if df.empty:
        print("[warn]  No data after filters. Exiting.")
        return

    segment_label = " · ".join(label_parts) if label_parts else "All segments"

    # ── Overall heatmap ──
    rate_pivot, count_pivot = build_pivot(df, metric, min_samples)
    slug = f"heatmap_{metric}"
    if label_parts:
        slug += "_" + "_".join(label_parts).replace(" ", "").lower()

    plot_heatmap(
        rate_pivot, count_pivot,
        metric      = metric,
        title       = segment_label,
        output_path = OUTPUT_DIR / f"{slug}.png",
        annotate_counts = annotate,
    )

    # ── Per-industry breakdown (only when no industry filter applied) ──
    if not industry_filter:
        for ind in sorted(df["industry"].unique()):
            sub  = df[df["industry"] == ind]
            rp, cp = build_pivot(sub, metric, min_samples)
            tier_suffix = f"_tier{tier_filter}" if tier_filter else ""
            plot_heatmap(
                rp, cp,
                metric      = metric,
                title       = f"{ind}{(' · Tier ' + str(tier_filter)) if tier_filter else ''}",
                output_path = OUTPUT_DIR / f"heatmap_{metric}_{ind.lower()}{tier_suffix}.png",
                annotate_counts = annotate,
            )

    # ── Top windows summary ──
    print(f"\n── Top 5 windows ({metric}, {segment_label}) ──")
    top = top_windows(rate_pivot, count_pivot, metric)
    if top.empty:
        print("  Not enough data above min_samples threshold yet.")
    else:
        print(top.to_string(index=False))
    print()

    # ── Export summary CSV ──
    summary_path = OUTPUT_DIR / f"top_windows_{metric}.csv"
    top.to_csv(summary_path, index=False)
    print(f"[export] Summary → {summary_path.relative_to(ROOT)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="TierFlow contact-time heatmap pipeline")
    p.add_argument("--csv",         default=str(DATA_DIR / "contact_attempts_seed.csv"))
    p.add_argument("--metric",      default="connect_rate",
                   choices=list(POSITIVE_OUTCOMES.keys()))
    p.add_argument("--industry",    default=None,
                   help="Filter to a single industry (e.g. SaaS)")
    p.add_argument("--tier",        default=None, type=int,
                   help="Filter to a single lead tier (1, 2, or 3)")
    p.add_argument("--min-samples", default=10, type=int,
                   help="Minimum attempts per cell before it appears on the heatmap")
    p.add_argument("--no-annotate", action="store_true",
                   help="Suppress rate/count annotations inside cells")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        csv_path        = Path(args.csv),
        metric          = args.metric,
        industry_filter = args.industry,
        tier_filter     = args.tier,
        min_samples     = args.min_samples,
        annotate        = not args.no_annotate,
    )
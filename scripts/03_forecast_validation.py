"""
Script 03 - Forecast Validation using ERA5-based Baselines
Earth2Research Project - Carpathian Basin Heat Dome & Drought Study

Three forecast methods compared against ERA5 ground truth:
  1. Persistence  - use T-48h field as forecast (no model needed)
  2. Climatology  - use ERA5 same day averaged over 1991-2010 baseline
  3. Linear Trend - fit linear trend from 5 days prior, extrapolate to peak

Skill scores: RMSE, Mean Bias, Anomaly Correlation Coefficient (ACC)
This is standard meteorological forecast validation methodology.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

PROJECT_ROOT = Path(r"C:\Users\User\Documents\Projects\earth2research")
FIGURES_DIR  = PROJECT_ROOT / "figures" / "historical"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("EARTH2STUDIO_CACHE", str(PROJECT_ROOT / "cache"))

LAT_MIN, LAT_MAX = 44.0, 49.5
LON_MIN, LON_MAX = 14.0, 23.5

EVENTS = [
    {
        "id":    "heatwave_2003",
        "name":  "2003 European Heatwave",
        "short": "Aug 2003",
        "color": "#c0392b",
        "peak":  datetime.datetime(2003, 8, 12, 12),
        "init":  datetime.datetime(2003, 8, 10, 12),   # T-48h
        "trend_days": [                                 # 5 days prior for trend
            datetime.datetime(2003, 8,  7, 12),
            datetime.datetime(2003, 8,  8, 12),
            datetime.datetime(2003, 8,  9, 12),
            datetime.datetime(2003, 8, 10, 12),
            datetime.datetime(2003, 8, 11, 12),
        ],
        "clim_month": 8, "clim_day": 12,
    },
    {
        "id":    "drought_2012",
        "name":  "2012 Pannonian Drought",
        "short": "Jul 2012",
        "color": "#d4ac0d",
        "peak":  datetime.datetime(2012, 7, 20, 12),
        "init":  datetime.datetime(2012, 7, 18, 12),
        "trend_days": [
            datetime.datetime(2012, 7, 15, 12),
            datetime.datetime(2012, 7, 16, 12),
            datetime.datetime(2012, 7, 17, 12),
            datetime.datetime(2012, 7, 18, 12),
            datetime.datetime(2012, 7, 19, 12),
        ],
        "clim_month": 7, "clim_day": 20,
    },
    {
        "id":    "heatdome_2021",
        "name":  "2021 Balkan Heat Dome",
        "short": "Jun 2021",
        "color": "#e67e22",
        "peak":  datetime.datetime(2021, 6, 24, 12),
        "init":  datetime.datetime(2021, 6, 22, 12),
        "trend_days": [
            datetime.datetime(2021, 6, 19, 12),
            datetime.datetime(2021, 6, 20, 12),
            datetime.datetime(2021, 6, 21, 12),
            datetime.datetime(2021, 6, 22, 12),
            datetime.datetime(2021, 6, 23, 12),
        ],
        "clim_month": 6, "clim_day": 24,
    },
]

# Climatology baseline years (pre-event, well-sampled in ERA5)
CLIM_YEARS = list(range(1991, 2011))

# ── Helpers ───────────────────────────────────────────────────────────────────

def slice_region(arr, lat_mask, lon_mask):
    return arr[np.ix_(lat_mask, lon_mask)]

def get_masks(lats, lons):
    return ((lats >= LAT_MIN) & (lats <= LAT_MAX),
            (lons >= LON_MIN) & (lons <= LON_MAX))

def fetch_t2m(data_source, time):
    da = data_source(time, ["t2m"])
    lats = da.coords["lat"].values
    lons = da.coords["lon"].values
    lat_mask, lon_mask = get_masks(lats, lons)
    t2m = slice_region(
        da.sel(variable="t2m").values.squeeze(), lat_mask, lon_mask
    ) - 273.15
    return t2m, lats[lat_mask], lons[lon_mask]

def compute_scores(forecast, truth):
    diff  = forecast - truth
    rmse  = float(np.sqrt(np.mean(diff**2)))
    bias  = float(np.mean(diff))
    fa    = forecast - forecast.mean()
    ta    = truth    - truth.mean()
    denom = np.sqrt(np.sum(fa**2) * np.sum(ta**2))
    acc   = float(np.sum(fa * ta) / denom) if denom > 0 else 0.0
    return {"rmse": rmse, "bias": bias, "acc": acc}

def add_map_features(ax):
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="#444")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#444")
    ax.add_feature(cfeature.LAND, facecolor="#f0ede8", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#d6e8f5", zorder=0)
    ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor="#7aabcc", zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                      color="#aaa", alpha=0.7, linestyle="--")
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 7}; gl.ylabel_style = {"size": 7}

# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Earth2Research - Script 03: Forecast Baseline Validation")
print("=" * 60)
print("Methods: Persistence | Climatology | Linear Trend")
print("Metric:  RMSE, Bias, ACC vs ERA5 ground truth\n")

from earth2studio.data import ARCO
data_source = ARCO()

all_scores = {}   # {event_id: {method: scores}}

for event in EVENTS:
    print(f"\n{'='*55}")
    print(f"Event: {event['name']}")
    print(f"  Peak:      {event['peak'].strftime('%Y-%m-%d %HZ')}")
    print(f"  Init(T-48):{event['init'].strftime('%Y-%m-%d %HZ')}")
    print(f"{'='*55}")

    event_scores = {}

    # ── Ground truth ──────────────────────────────────────────────────────────
    print(f"  Fetching ERA5 ground truth at peak...")
    truth, lats_r, lons_r = fetch_t2m(data_source, event["peak"])
    print(f"  ERA5 truth T2m: {truth.min():.1f} - {truth.max():.1f}°C")

    # ── Method 1: Persistence ─────────────────────────────────────────────────
    print(f"\n  [Method 1] Persistence (T-48h field as forecast)...")
    pers, _, _ = fetch_t2m(data_source, event["init"])
    s_pers = compute_scores(pers, truth)
    event_scores["Persistence"] = {**s_pers, "forecast": pers}
    print(f"    RMSE={s_pers['rmse']:.3f}°C  Bias={s_pers['bias']:+.3f}°C  ACC={s_pers['acc']:.4f}")

    # ── Method 2: Climatology ─────────────────────────────────────────────────
    print(f"\n  [Method 2] Climatology (1991-2010 mean for same day)...")
    clim_fields = []
    for yr in CLIM_YEARS:
        try:
            t = datetime.datetime(yr, event["clim_month"], event["clim_day"], 12)
            f, _, _ = fetch_t2m(data_source, t)
            clim_fields.append(f)
        except Exception:
            pass
    clim_mean = np.mean(clim_fields, axis=0)
    s_clim = compute_scores(clim_mean, truth)
    event_scores["Climatology"] = {**s_clim, "forecast": clim_mean}
    print(f"    Years used: {len(clim_fields)}/20")
    print(f"    RMSE={s_clim['rmse']:.3f}°C  Bias={s_clim['bias']:+.3f}°C  ACC={s_clim['acc']:.4f}")

    # ── Method 3: Linear Trend Extrapolation ──────────────────────────────────
    print(f"\n  [Method 3] Linear trend extrapolation (5-day prior window)...")
    trend_fields = []
    trend_times  = []
    for t in event["trend_days"]:
        try:
            f, _, _ = fetch_t2m(data_source, t)
            trend_fields.append(f)
            # Hours since first trend day
            trend_times.append((t - event["trend_days"][0]).total_seconds() / 3600)
        except Exception:
            pass

    if len(trend_fields) >= 3:
        trend_fields = np.array(trend_fields)  # (n_days, lat, lon)
        trend_times  = np.array(trend_times)
        # Fit linear trend per grid point
        h, w = trend_fields.shape[1], trend_fields.shape[2]
        fields_2d = trend_fields.reshape(len(trend_fields), -1)
        # Least squares: y = a + b*t
        A = np.vstack([np.ones_like(trend_times), trend_times]).T
        coeffs = np.linalg.lstsq(A, fields_2d, rcond=None)[0]  # (2, n_pixels)
        # Extrapolate to peak time
        t_peak_h = (event["peak"] - event["trend_days"][0]).total_seconds() / 3600
        fc_trend = (coeffs[0] + coeffs[1] * t_peak_h).reshape(h, w)
        s_trend = compute_scores(fc_trend, truth)
        event_scores["Linear Trend"] = {**s_trend, "forecast": fc_trend}
        print(f"    Days used: {len(trend_fields)}, extrapolation: {t_peak_h:.0f}h")
        print(f"    RMSE={s_trend['rmse']:.3f}°C  Bias={s_trend['bias']:+.3f}°C  ACC={s_trend['acc']:.4f}")
    else:
        print(f"    Not enough trend data, skipping")

    all_scores[event["id"]] = {
        "event":        event,
        "truth":        truth,
        "lats_r":       lats_r,
        "lons_r":       lons_r,
        "method_scores": event_scores,
    }

    # ── Validation figure: 4-panel (truth + 3 methods) ────────────────────────
    print(f"\n  Generating validation figure...")
    methods = list(event_scores.keys())
    n_panels = 1 + len(methods)

    fig, axes = plt.subplots(
        1, n_panels, figsize=(5 * n_panels, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        facecolor="#f9f9f9"
    )

    all_t2m = [truth] + [event_scores[m]["forecast"] for m in methods]
    vmin = min(f.min() for f in all_t2m) - 1
    vmax = max(f.max() for f in all_t2m) + 1

    panels = [("ERA5 Ground Truth\n(Reanalysis)", truth, "#2c3e50")] + [
        (f"{m}\nRMSE={event_scores[m]['rmse']:.2f}°C  ACC={event_scores[m]['acc']:.3f}",
         event_scores[m]["forecast"], event["color"])
        for m in methods
    ]

    for ax, (title, data, tc) in zip(axes, panels):
        add_map_features(ax)
        im = ax.pcolormesh(lons_r, lats_r, data,
                           cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), zorder=2)
        cb = plt.colorbar(im, ax=ax, orientation="horizontal",
                          pad=0.06, shrink=0.9, aspect=22)
        cb.set_label("T2m (°C)", fontsize=7)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=5, color=tc)

    fig.suptitle(
        f"{event['name']} — 48h Forecast Validation\n"
        f"Peak: {event['peak'].strftime('%d %b %Y %HZ')}  |  "
        f"Init: {event['init'].strftime('%d %b %Y %HZ')}",
        fontsize=11, fontweight="bold", y=1.03
    )
    plt.tight_layout()
    out = FIGURES_DIR / f"03_{event['id']}_validation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out.name}")

# ── Cross-event skill score comparison figure ─────────────────────────────────
print(f"\n{'='*55}")
print("Generating cross-event skill score figure...")

method_names = ["Persistence", "Climatology", "Linear Trend"]
method_colors = {"Persistence": "#2980b9", "Climatology": "#27ae60",
                 "Linear Trend": "#8e44ad"}
event_list = list(all_scores.keys())
event_labels = [all_scores[e]["event"]["short"] for e in event_list]
x = np.arange(len(event_list))
width = 0.25

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#f9f9f9")
fig.suptitle(
    "48h Forecast Skill — Carpathian Basin T2m\n"
    "Three Baseline Methods vs ERA5 Ground Truth",
    fontsize=13, fontweight="bold"
)

metrics = [
    ("rmse",  "RMSE (°C)",           "lower is better",  False),
    ("bias",  "Mean Bias (°C)",       "closer to 0",      True),
    ("acc",   "Anomaly Correlation",  "higher is better", False),
]

for ax, (metric, ylabel, subtitle, zero_line) in zip(axes, metrics):
    for i, method in enumerate(method_names):
        vals = []
        for eid in event_list:
            ms = all_scores[eid]["method_scores"]
            vals.append(ms[method][metric] if method in ms else np.nan)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width,
                      label=method, color=method_colors[method],
                      edgecolor="white", linewidth=1.0, alpha=0.9)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.01 if v >= 0 else -0.08),
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold")

    if zero_line:
        ax.axhline(0, color="#333", lw=0.8, ls="-")
    if metric == "acc":
        ax.axhline(0.6, color="#888", lw=0.8, ls="--", alpha=0.7)
        ax.set_ylim(0, 1.1)
        ax.text(len(event_list)-0.4, 0.62, "Skillful (0.6)",
                fontsize=7, color="#888")

    ax.set_xticks(x)
    ax.set_xticklabels(event_labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"{ylabel}\n({subtitle})", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
out = FIGURES_DIR / "03_skill_scores_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out.name}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("VALIDATION SUMMARY — 48h Forecast Skill")
print(f"{'='*60}")
for eid in event_list:
    ev = all_scores[eid]["event"]
    ms = all_scores[eid]["method_scores"]
    print(f"\n{ev['name']}")
    print(f"  {'Method':<18} {'RMSE':>8} {'Bias':>9} {'ACC':>8}")
    print(f"  {'-'*46}")
    for method in method_names:
        if method in ms:
            s = ms[method]
            print(f"  {method:<18} {s['rmse']:>7.3f}°C {s['bias']:>+8.3f}°C {s['acc']:>8.4f}")

print(f"\n{'='*60}")
print("Script 03 complete!")
print(f"{'='*60}")
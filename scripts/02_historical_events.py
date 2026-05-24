"""
Script 02 - Historical Extreme Event Analysis
Earth2Research Project - Carpathian Basin Heat Dome & Drought Study

Analyzes 3 historical extreme events:
  1. 2003 European Heatwave (Aug 2003)
  2. 2012 Pannonian Drought (Jul-Aug 2012)
  3. 2021 Balkan/Hungarian Heat Dome (Jun 2021)

For each event we fetch ERA5 data across multiple timesteps,
compute heat dome and drought indicators, and generate figures.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

PROJECT_ROOT = Path(r"C:\Users\User\Documents\Projects\earth2research")
FIGURES_DIR  = PROJECT_ROOT / "figures" / "historical"
DATA_DIR     = PROJECT_ROOT / "data" / "era5"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("EARTH2STUDIO_CACHE", str(PROJECT_ROOT / "cache"))

# Carpathian Basin bounding box
LAT_MIN, LAT_MAX = 44.0, 49.5
LON_MIN, LON_MAX = 14.0, 23.5

# ── Event definitions ─────────────────────────────────────────────────────────
# Each event: name, peak date/time, analysis window (daily 12Z snapshots)
EVENTS = [
    {
        "id":    "heatwave_2003",
        "name":  "2003 European Heatwave",
        "short": "Aug 2003",
        "peak":  datetime.datetime(2003, 8, 12, 12),
        "window": [
            datetime.datetime(2003, 8,  4, 12),
            datetime.datetime(2003, 8,  6, 12),
            datetime.datetime(2003, 8,  8, 12),
            datetime.datetime(2003, 8, 10, 12),
            datetime.datetime(2003, 8, 12, 12),
            datetime.datetime(2003, 8, 14, 12),
            datetime.datetime(2003, 8, 16, 12),
        ],
        "color": "#c0392b",
        "type":  "heatwave",
    },
    {
        "id":    "drought_2012",
        "name":  "2012 Pannonian Drought",
        "short": "Jul-Aug 2012",
        "peak":  datetime.datetime(2012, 7, 20, 12),
        "window": [
            datetime.datetime(2012, 7,  6, 12),
            datetime.datetime(2012, 7, 10, 12),
            datetime.datetime(2012, 7, 14, 12),
            datetime.datetime(2012, 7, 18, 12),
            datetime.datetime(2012, 7, 20, 12),
            datetime.datetime(2012, 7, 24, 12),
            datetime.datetime(2012, 7, 28, 12),
        ],
        "color": "#d4ac0d",
        "type":  "drought",
    },
    {
        "id":    "heatdome_2021",
        "name":  "2021 Balkan Heat Dome",
        "short": "Jun 2021",
        "peak":  datetime.datetime(2021, 6, 24, 12),
        "window": [
            datetime.datetime(2021, 6, 18, 12),
            datetime.datetime(2021, 6, 20, 12),
            datetime.datetime(2021, 6, 22, 12),
            datetime.datetime(2021, 6, 24, 12),
            datetime.datetime(2021, 6, 26, 12),
            datetime.datetime(2021, 6, 28, 12),
            datetime.datetime(2021, 6, 30, 12),
        ],
        "color": "#e67e22",
        "type":  "heatwave",
    },
]

# ── Helper functions ──────────────────────────────────────────────────────────

def fetch_era5(data_source, time, variables):
    """Fetch ERA5 data and return as dict of 2D numpy arrays."""
    da = data_source(time, variables)
    lats = da.coords["lat"].values
    lons = da.coords["lon"].values
    lat_mask = (lats >= LAT_MIN) & (lats <= LAT_MAX)
    lon_mask = (lons >= LON_MIN) & (lons <= LON_MAX)
    result = {
        "lats": lats[lat_mask],
        "lons": lons[lon_mask],
        "lat_mask": lat_mask,
        "lon_mask": lon_mask,
    }
    for var in variables:
        arr = da.sel(variable=var).values.squeeze()
        result[var] = arr[np.ix_(lat_mask, lon_mask)]
    return result


def heat_dome_index(z500):
    """
    Heat dome index: mean Z500 anomaly relative to spatial mean.
    Higher Z500 = stronger high-pressure dome = heat dome.
    Returns scalar in metres.
    """
    return float(z500.mean())


def heat_stress_index(t2m_c):
    """
    Simple heat stress: area fraction where T2m > 35C.
    Returns fraction 0-1.
    """
    return float((t2m_c > 35.0).mean())


def drought_proxy(tp_mm):
    """
    Drought proxy: mean precipitation (mm/6h) over region.
    Lower = drier conditions.
    """
    return float(tp_mm.mean())


def add_map_features(ax, extent=None):
    if extent is None:
        extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="#444")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#444")
    ax.add_feature(cfeature.LAND, facecolor="#f0ede8", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#d6e8f5", zorder=0)
    ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor="#7aabcc", zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                      color="#aaa", alpha=0.7, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}


# ── Main analysis ─────────────────────────────────────────────────────────────

print("=" * 60)
print("Earth2Research - Script 02: Historical Event Analysis")
print("=" * 60)

from earth2studio.data import ARCO
data_source = ARCO()

# Variables to fetch
VARS = ["t2m", "z500", "tp"]   # 2m temp, 500hPa geopotential, total precip

all_results = {}   # store time-series stats per event

for event in EVENTS:
    print(f"\n{'='*50}")
    print(f"Processing: {event['name']}")
    print(f"{'='*50}")

    time_series = {
        "times":      [],
        "t2m_max":    [],
        "t2m_mean":   [],
        "z500_mean":  [],
        "heat_stress":[],
        "precip_mean":[],
    }

    peak_data = None   # store full 2D arrays at peak time

    for t in event["window"]:
        print(f"  Fetching {t.strftime('%Y-%m-%d %HZ')} ...", end=" ", flush=True)
        try:
            data = fetch_era5(data_source, t, VARS)

            t2m_c  = data["t2m"] - 273.15
            z500_m = data["z500"] / 9.80665
            tp_mm  = data["tp"] * 1000   # m → mm

            time_series["times"].append(t)
            time_series["t2m_max"].append(float(t2m_c.max()))
            time_series["t2m_mean"].append(float(t2m_c.mean()))
            time_series["z500_mean"].append(heat_dome_index(z500_m))
            time_series["heat_stress"].append(heat_stress_index(t2m_c))
            time_series["precip_mean"].append(drought_proxy(tp_mm))

            if t == event["peak"]:
                peak_data = {
                    "t2m_c":  t2m_c,
                    "z500_m": z500_m,
                    "tp_mm":  tp_mm,
                    "lats":   data["lats"],
                    "lons":   data["lons"],
                }
                print(f"PEAK | T2m max={t2m_c.max():.1f}C | Z500 mean={z500_m.mean():.0f}m", flush=True)
            else:
                print(f"T2m max={t2m_c.max():.1f}C", flush=True)

        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            continue

    all_results[event["id"]] = {
        "event":       event,
        "time_series": time_series,
        "peak_data":   peak_data,
    }

    # ── Figure A: Peak day maps ───────────────────────────────────────────────
    if peak_data is not None:
        print(f"\n  Generating peak day map...")
        fig, axes = plt.subplots(
            1, 2, figsize=(13, 5.5),
            subplot_kw={"projection": ccrs.PlateCarree()},
            facecolor="#f9f9f9"
        )
        fig.suptitle(
            f"{event['name']} — Peak Day ({event['peak'].strftime('%d %b %Y, %HZ')})\n"
            f"Carpathian Basin | ARCO-ERA5 Reanalysis",
            fontsize=12, fontweight="bold", y=1.02
        )

        lats_r = peak_data["lats"]
        lons_r = peak_data["lons"]
        t2m_c  = peak_data["t2m_c"]
        z500_m = peak_data["z500_m"]

        # T2m panel
        ax1 = axes[0]
        add_map_features(ax1)
        vmin = max(10, t2m_c.min() - 1)
        vmax = min(50, t2m_c.max() + 1)
        im1 = ax1.pcolormesh(lons_r, lats_r, t2m_c,
                              cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
                              transform=ccrs.PlateCarree(), zorder=2)
        cb1 = plt.colorbar(im1, ax=ax1, orientation="horizontal",
                            pad=0.06, shrink=0.85, aspect=28)
        cb1.set_label("2m Temperature (°C)", fontsize=9)
        cb1.ax.tick_params(labelsize=8)
        ax1.set_title("2m Temperature", fontsize=10, fontweight="bold", pad=6)

        max_idx = np.unravel_index(t2m_c.argmax(), t2m_c.shape)
        ax1.plot(lons_r[max_idx[1]], lats_r[max_idx[0]],
                 "w*", ms=13, transform=ccrs.PlateCarree(), zorder=5)
        ax1.annotate(f"Max: {t2m_c.max():.1f}°C",
                     xy=(lons_r[max_idx[1]], lats_r[max_idx[0]]),
                     xytext=(lons_r[max_idx[1]] + 0.3, lats_r[max_idx[0]] - 0.45),
                     fontsize=8, color="white", fontweight="bold",
                     transform=ccrs.PlateCarree(),
                     bbox=dict(boxstyle="round,pad=0.2", fc=event["color"], alpha=0.9))

        # Z500 panel
        ax2 = axes[1]
        add_map_features(ax2)
        im2 = ax2.pcolormesh(lons_r, lats_r, z500_m,
                              cmap="YlOrRd",
                              transform=ccrs.PlateCarree(), zorder=2)
        cs = ax2.contour(lons_r, lats_r, z500_m, levels=8,
                          colors="white", linewidths=0.7,
                          transform=ccrs.PlateCarree(), zorder=3)
        ax2.clabel(cs, inline=True, fontsize=7, fmt="%d m")
        cb2 = plt.colorbar(im2, ax=ax2, orientation="horizontal",
                            pad=0.06, shrink=0.85, aspect=28)
        cb2.set_label("500hPa Geopotential Height (m)", fontsize=9)
        cb2.ax.tick_params(labelsize=8)
        ax2.set_title("Z500 — Heat Dome Indicator", fontsize=10, fontweight="bold", pad=6)

        plt.tight_layout()
        out = FIGURES_DIR / f"02_{event['id']}_peak_map.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved: {out.name}")

    # ── Figure B: Time-series panel ───────────────────────────────────────────
    if len(time_series["times"]) > 1:
        print(f"  Generating time-series plot...")
        ts = time_series
        times = ts["times"]
        labels = [t.strftime("%d %b\n%HZ") for t in times]
        x = np.arange(len(times))

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), facecolor="#f9f9f9",
                                  sharex=True)
        fig.suptitle(f"{event['name']} — Event Time Series\nCarpathian Basin Regional Mean",
                     fontsize=12, fontweight="bold")

        ec = event["color"]

        # T2m max
        axes[0].plot(x, ts["t2m_max"], "o-", color=ec, lw=2, ms=6, label="T2m max")
        axes[0].fill_between(x, ts["t2m_mean"], ts["t2m_max"], alpha=0.2, color=ec)
        axes[0].plot(x, ts["t2m_mean"], "--", color=ec, lw=1.2, alpha=0.7, label="T2m mean")
        axes[0].axhline(35, color="#888", lw=0.8, ls=":", label="35°C threshold")
        axes[0].set_ylabel("Temperature (°C)", fontsize=9)
        axes[0].legend(fontsize=8, loc="upper left")
        axes[0].set_title("2m Temperature", fontsize=9, pad=4)
        axes[0].grid(True, alpha=0.3)

        # Z500
        axes[1].plot(x, ts["z500_mean"], "s-", color="#8e44ad", lw=2, ms=6)
        axes[1].set_ylabel("Z500 (m)", fontsize=9)
        axes[1].set_title("500hPa Geopotential Height (Heat Dome Intensity)", fontsize=9, pad=4)
        axes[1].grid(True, alpha=0.3)

        # Precipitation
        axes[2].bar(x, ts["precip_mean"], color="#2980b9", alpha=0.7, width=0.5)
        axes[2].set_ylabel("Precip (mm/6h)", fontsize=9)
        axes[2].set_title("Regional Mean Precipitation (Drought Indicator)", fontsize=9, pad=4)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, fontsize=8)
        axes[2].grid(True, alpha=0.3, axis="y")

        # Mark peak
        peak_idx = times.index(event["peak"]) if event["peak"] in times else None
        if peak_idx is not None:
            for ax in axes:
                ax.axvline(peak_idx, color=ec, lw=1.5, ls="--", alpha=0.6)
            axes[0].annotate("Peak", xy=(peak_idx, ts["t2m_max"][peak_idx]),
                             xytext=(peak_idx + 0.15, ts["t2m_max"][peak_idx] + 0.3),
                             fontsize=8, color=ec, fontweight="bold")

        plt.tight_layout()
        out = FIGURES_DIR / f"02_{event['id']}_timeseries.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved: {out.name}")


# ── Figure C: 3-event comparison summary ─────────────────────────────────────
print(f"\n{'='*50}")
print("Generating 3-event comparison figure...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5),
                          subplot_kw={"projection": ccrs.PlateCarree()},
                          facecolor="#f9f9f9")
fig.suptitle(
    "Carpathian Basin Extreme Events — Peak Day 2m Temperature Comparison\n"
    "ARCO-ERA5 Reanalysis | Earth2Studio",
    fontsize=13, fontweight="bold", y=1.02
)

# Find global t2m range across all events for consistent colorbar
all_t2m = []
for eid, res in all_results.items():
    if res["peak_data"] is not None:
        all_t2m.append(res["peak_data"]["t2m_c"])

if all_t2m:
    global_vmin = max(10, min(a.min() for a in all_t2m) - 1)
    global_vmax = min(50, max(a.max() for a in all_t2m) + 1)

    for i, (event, (eid, res)) in enumerate(zip(EVENTS, all_results.items())):
        ax = axes[i]
        pd = res["peak_data"]
        if pd is None:
            ax.set_title(f"{event['short']}\n(no data)", fontsize=10)
            continue

        add_map_features(ax)
        im = ax.pcolormesh(pd["lons"], pd["lats"], pd["t2m_c"],
                            cmap="RdYlBu_r",
                            vmin=global_vmin, vmax=global_vmax,
                            transform=ccrs.PlateCarree(), zorder=2)

        max_idx = np.unravel_index(pd["t2m_c"].argmax(), pd["t2m_c"].shape)
        ax.plot(pd["lons"][max_idx[1]], pd["lats"][max_idx[0]],
                "w*", ms=12, transform=ccrs.PlateCarree(), zorder=5)

        ax.set_title(
            f"{event['name']}\n{event['peak'].strftime('%d %b %Y')} | "
            f"Max: {pd['t2m_c'].max():.1f}°C",
            fontsize=9, fontweight="bold", pad=6
        )

        if i == 2:
            cb = plt.colorbar(im, ax=axes, orientation="horizontal",
                               pad=0.06, shrink=0.6, aspect=40, fraction=0.02)
            cb.set_label("2m Temperature (°C)", fontsize=10)
            cb.ax.tick_params(labelsize=9)

plt.tight_layout()
out = FIGURES_DIR / "02_three_event_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out.name}")

# ── Print summary statistics ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")
print(f"{'Event':<30} {'Peak T2m':>10} {'Mean Z500':>10} {'Max Precip':>12}")
print("-" * 65)
for eid, res in all_results.items():
    ev = res["event"]
    ts = res["time_series"]
    if ts["t2m_max"]:
        peak_t = max(ts["t2m_max"])
        mean_z = np.mean(ts["z500_mean"])
        max_p  = max(ts["precip_mean"]) if ts["precip_mean"] else 0
        print(f"{ev['name']:<30} {peak_t:>9.1f}C {mean_z:>10.0f}m {max_p:>11.3f}mm")

print(f"\n{'='*60}")
print("Script 02 complete!")
print(f"Figures saved to: {FIGURES_DIR}")
print(f"{'='*60}")
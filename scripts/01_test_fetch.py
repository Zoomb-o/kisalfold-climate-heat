"""
Script 01 - ERA5 Data Fetch Test
Earth2Research Project - Carpathian Basin Heat Dome & Drought Study
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

# Suppress OpenMP duplicate library warning (harmless conda issue)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

PROJECT_ROOT = Path(r"C:\Users\User\Documents\Projects\earth2research")
FIGURES_DIR  = PROJECT_ROOT / "figures" / "historical"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("EARTH2STUDIO_CACHE", str(PROJECT_ROOT / "cache"))

# Carpathian Basin bounding box
LAT_MIN, LAT_MAX = 44.0, 49.5
LON_MIN, LON_MAX = 14.0, 23.5

print("=" * 60)
print("Earth2Research - Test 01: ERA5 Data Fetch")
print("=" * 60)
print(f"\nTarget region: Carpathian Basin")
print(f"  Lat: {LAT_MIN}N - {LAT_MAX}N")
print(f"  Lon: {LON_MIN}E - {LON_MAX}E")
print(f"\nFetching ERA5 data for 2003-08-12 12Z (heatwave peak, midday)...")

try:
    import datetime
    from earth2studio.data import ARCO

    data_source = ARCO()
    # 12Z = midday UTC = afternoon temperatures in central Europe
    time = datetime.datetime(2003, 8, 12, 12, 0, 0)
    variables = ["t2m", "z500"]

    print(f"Fetching: {variables} at {time}\n")
    da = data_source(time, variables)

    print("Data fetched!")
    print(f"  Shape: {da.shape}")

    lats = da.coords["lat"].values
    lons = da.coords["lon"].values

    print(f"  Lat range in data: {lats.min():.1f} to {lats.max():.1f}")
    print(f"  Lon range in data: {lons.min():.1f} to {lons.max():.1f}")

    # ARCO uses 0-360 longitudes, our region is 14-23.5E which is fine (no wrap)
    lat_mask = (lats >= LAT_MIN) & (lats <= LAT_MAX)
    lon_mask = (lons >= LON_MIN) & (lons <= LON_MAX)

    print(f"\n  lat_mask matches: {lat_mask.sum()} points")
    print(f"  lon_mask matches: {lon_mask.sum()} points")

    # Extract using .sel() for safety
    t2m_global  = da.sel(variable="t2m").values.squeeze()
    z500_global = da.sel(variable="z500").values.squeeze()

    print(f"  t2m_global shape: {t2m_global.shape}")
    print(f"  Full global T2m range: {t2m_global.min()-273.15:.1f}C - {t2m_global.max()-273.15:.1f}C")

    # Check a known hot spot: France/Iberia during 2003 heatwave
    # approx lat 43-48N, lon 0-10E
    lat_mask_fr = (lats >= 43.0) & (lats <= 48.0)
    lon_mask_fr = (lons >= 0.0)  & (lons <= 10.0)
    t2m_france = t2m_global[np.ix_(lat_mask_fr, lon_mask_fr)] - 273.15
    print(f"  France region T2m check: {t2m_france.max():.1f}C (should be ~35-40C for 2003)")

    t2m_region  = t2m_global[np.ix_(lat_mask, lon_mask)] - 273.15
    z500_region = z500_global[np.ix_(lat_mask, lon_mask)] / 9.80665

    lats_region = lats[lat_mask]
    lons_region = lons[lon_mask]

    print(f"\nCarpathian Basin:")
    print(f"  Grid: {t2m_region.shape[0]} lat x {t2m_region.shape[1]} lon")
    print(f"  T2m:  {t2m_region.min():.1f}C - {t2m_region.max():.1f}C")
    print(f"  Z500: {z500_region.min():.0f}m - {z500_region.max():.0f}m")

    # Auto-scale colorbar to actual data
    t2m_vmin = max(10, t2m_region.min() - 2)
    t2m_vmax = min(45, t2m_region.max() + 2)

    print("\nGenerating plot...")

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
        facecolor="#f8f8f8"
    )
    fig.suptitle(
        "Carpathian Basin - 2003 European Heatwave (12 Aug 2003, 12Z midday)\n"
        "Source: ARCO-ERA5 reanalysis via Earth2Studio",
        fontsize=13, fontweight="bold", y=1.01
    )

    def add_map_features(ax):
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="#444")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#444")
        ax.add_feature(cfeature.LAND, facecolor="#f0ede8", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#d6e8f5", zorder=0)
        ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor="#7aabcc", zorder=1)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="#aaa",
                          alpha=0.7, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}

    # Panel 1 - 2m Temperature
    ax1 = axes[0]
    add_map_features(ax1)
    im1 = ax1.pcolormesh(
        lons_region, lats_region, t2m_region,
        cmap="RdYlBu_r", vmin=t2m_vmin, vmax=t2m_vmax,
        transform=ccrs.PlateCarree(), zorder=2
    )
    cb1 = plt.colorbar(im1, ax=ax1, orientation="horizontal",
                       pad=0.06, shrink=0.85, aspect=30)
    cb1.set_label("2m Temperature (C)", fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    ax1.set_title("2m Temperature (T2m)", fontsize=11, fontweight="bold", pad=8)

    max_idx = np.unravel_index(t2m_region.argmax(), t2m_region.shape)
    ax1.plot(lons_region[max_idx[1]], lats_region[max_idx[0]],
             "w*", markersize=14, transform=ccrs.PlateCarree(), zorder=5)
    ax1.annotate(
        f"Max: {t2m_region.max():.1f}C",
        xy=(lons_region[max_idx[1]], lats_region[max_idx[0]]),
        xytext=(lons_region[max_idx[1]] + 0.3, lats_region[max_idx[0]] - 0.4),
        fontsize=8, color="white", fontweight="bold",
        transform=ccrs.PlateCarree(),
        bbox=dict(boxstyle="round,pad=0.2", fc="#c0392b", alpha=0.85)
    )

    # Panel 2 - Z500
    ax2 = axes[1]
    add_map_features(ax2)
    im2 = ax2.pcolormesh(
        lons_region, lats_region, z500_region,
        cmap="RdPu", transform=ccrs.PlateCarree(), zorder=2
    )
    cs = ax2.contour(
        lons_region, lats_region, z500_region,
        levels=8, colors="white", linewidths=0.6,
        transform=ccrs.PlateCarree(), zorder=3
    )
    ax2.clabel(cs, inline=True, fontsize=7, fmt="%d m")
    cb2 = plt.colorbar(im2, ax=ax2, orientation="horizontal",
                       pad=0.06, shrink=0.85, aspect=30)
    cb2.set_label("500hPa Geopotential Height (m)", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    ax2.set_title("500hPa Geopotential Height (Z500)\nHeat dome indicator",
                  fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout()
    out_path = FIGURES_DIR / "01_test_carpathian_heatwave_2003.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"\nFigure saved: {out_path}")

except Exception as e:
    print(f"\nError: {e}")
    raise

print("\n" + "=" * 60)
print("Test 01 complete!")
print("=" * 60)
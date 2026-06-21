"""
inputs/reservoir.py
-------------------
Derives the height-area-volume (HAV) curve of a reservoir from a DEM.

Workflow
--------
1. Download a Copernicus GLO-30 DSM tile (30 m, free, no sign-in) for an
   initial bounding box around the dam.
2. Condition the DEM (fill depressions, resolve flats) using pysheds and
   delineate the upstream catchment (D8 routing).
3. Check whether the catchment mask touches the DEM boundary. If so, expand
   the bounding box and repeat from step 1 until the catchment fits entirely
   within the DEM extent (dynamic coverage fitting).
4. Sweep water surface elevation (WSE) from the base of the dam to the crest
   in user-defined steps, computing inundated area and cumulative volume
   within the catchment mask at each step (bathtub method).
5. Export results as CSV and GeoJSON.

DEM source
----------
Copernicus GLO-30 (2023_1), hosted as unsigned Cloud-Optimised GeoTIFFs on
AWS S3:
    s3://copernicus-dem-30m/<tile>/<tile>.tif  (no credentials required)

Accessed via the public HTTPS endpoint:
    https://copernicus-dem-30m.s3.amazonaws.com/<tile>/<tile>.tif

Tile naming: Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM
(1-degree tiles, named by SW corner)

Notes
-----
- All units are SI: elevations in m (masl), area in m^2, volume in m^3.
- The DEM is a DSM; vegetation/canopy bias may cause minor volume
  overestimation in forested areas.
- Dam crest elevation = DEM elevation at dam location + dam height.
- The bathtub fill is bounded by the upstream catchment mask; it does not
  spread downstream of the dam.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rasterio
import rasterio.merge
import rasterio.transform
import rasterio.warp
from pysheds.grid import Grid
from rasterio.crs import CRS
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GLO30_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "{tile}/{tile}.tif"
)
_DEFAULT_BUFFER_DEG = 0.35   # degrees: initial DEM download radius around dam
_BUFFER_INCREMENT_DEG = 0.25 # degrees: expansion step when catchment hits boundary
_MAX_BUFFER_DEG = 2.5        # degrees: hard cap (~250 km radius) to prevent runaway
_BOUNDARY_MARGIN_PX = 5      # pixels: catchment must stay this far from DEM edge


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

class HAVCurve(NamedTuple):
    """Height-area-volume curve result."""
    wse_m: np.ndarray          # water surface elevation (masl)
    area_m2: np.ndarray        # inundated area (m^2)
    volume_m3: np.ndarray      # cumulative reservoir volume (m^3)
    dam_elevation_m: float     # DEM elevation at dam toe (masl)
    crest_elevation_m: float   # dam crest elevation (masl)


# ---------------------------------------------------------------------------
# DEM acquisition
# ---------------------------------------------------------------------------

def _tile_name(lat_deg: int, lon_deg: int) -> str:
    """Return GLO-30 tile name for the 1-degree tile whose SW corner is
    (lat_deg, lon_deg)."""
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    return (
        f"Copernicus_DSM_COG_10_{ns}{abs(lat_deg):02d}_00_"
        f"{ew}{abs(lon_deg):03d}_00_DEM"
    )


def _required_tiles(south: float, north: float, west: float, east: float):
    """Return list of (lat, lon) SW-corner integers for all tiles covering bbox."""
    lat_start = math.floor(south)
    lat_end = math.floor(north)
    lon_start = math.floor(west)
    lon_end = math.floor(east)
    tiles = []
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            tiles.append((lat, lon))
    return tiles


def _download_tile(lat: int, lon: int, dest_dir: str) -> str | None:
    """Download a single GLO-30 tile to dest_dir. Returns local path or None."""
    name = _tile_name(lat, lon)
    url = _GLO30_URL.format(tile=name)
    dest = os.path.join(dest_dir, f"{name}.tif")
    if os.path.exists(dest):
        return dest
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dambuhala/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return dest
    except Exception as exc:
        # Tile may not exist (ocean/void); skip silently
        print(f"  [warn] Could not download tile ({lat},{lon}): {exc}")
        return None


def fetch_dem(
    lat: float,
    lon: float,
    buffer_deg: float = _DEFAULT_BUFFER_DEG,
    cache_dir: str | None = None,
) -> tuple[np.ndarray, rasterio.transform.Affine, CRS]:
    """
    Download and mosaic Copernicus GLO-30 tiles covering the bounding box.

    Parameters
    ----------
    lat, lon : float
        Dam location in decimal degrees (WGS84).
    buffer_deg : float
        Degrees of padding around the dam (default 0.35 ~ 35 km).
    cache_dir : str, optional
        Directory to cache downloaded tiles. Uses a temp dir if None.

    Returns
    -------
    data : np.ndarray  (2-D float32)
    transform : rasterio.Affine
    crs : rasterio.CRS
    """
    south = lat - buffer_deg
    north = lat + buffer_deg
    west = lon - buffer_deg
    east = lon + buffer_deg

    work_dir = cache_dir or tempfile.mkdtemp(prefix="dambuhala_dem_")
    os.makedirs(work_dir, exist_ok=True)

    tiles = _required_tiles(south, north, west, east)
    paths = []
    for t_lat, t_lon in tiles:
        p = _download_tile(t_lat, t_lon, work_dir)
        if p:
            paths.append(p)

    if not paths:
        raise RuntimeError("No DEM tiles could be downloaded for this location.")

    if len(paths) == 1:
        with rasterio.open(paths[0]) as src:
            data = src.read(1).astype(np.float32)
            data[data == src.nodata] = np.nan
            transform = src.transform
            crs = src.crs
    else:
        datasets = [rasterio.open(p) for p in paths]
        mosaic, transform = rasterio.merge.merge(datasets)
        crs = datasets[0].crs
        for ds in datasets:
            ds.close()
        data = mosaic[0].astype(np.float32)
        nodata = datasets[0].nodata if datasets[0].nodata is not None else -32768
        data[data == nodata] = np.nan

    # Clip to bbox
    with rasterio.open(paths[0]) as ref:
        pass  # just needed crs above

    return data, transform, crs


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _rowcol(transform: rasterio.transform.Affine, lon: float, lat: float):
    """Convert geographic lon/lat to row, col indices."""
    col, row = ~transform * (lon, lat)
    return int(round(row)), int(round(col))


def _pixel_area_m2(transform: rasterio.transform.Affine, lat: float) -> float:
    """Approximate pixel area in m^2 using the equirectangular pixel dimensions
    at the given latitude. Accurate to <1% for 30 m pixels."""
    R = 6_378_137.0  # WGS84 equatorial radius, metres
    deg_to_rad = math.pi / 180.0
    dx_deg = abs(transform.a)   # pixel width in degrees
    dy_deg = abs(transform.e)   # pixel height in degrees
    dx_m = dx_deg * deg_to_rad * R * math.cos(lat * deg_to_rad)
    dy_m = dy_deg * deg_to_rad * R
    return dx_m * dy_m


# ---------------------------------------------------------------------------
# Boundary check
# ---------------------------------------------------------------------------

def _catchment_hits_boundary(mask: np.ndarray, margin: int = _BOUNDARY_MARGIN_PX) -> bool:
    """Return True if any catchment pixel lies within `margin` pixels of the
    DEM grid edge, indicating the watershed may extend beyond the current DEM."""
    return bool(
        mask[:margin, :].any()
        or mask[-margin:, :].any()
        or mask[:, :margin].any()
        or mask[:, -margin:].any()
    )


# ---------------------------------------------------------------------------
# Watershed delineation
# ---------------------------------------------------------------------------

def _delineate_catchment(
    dem: np.ndarray,
    transform: rasterio.transform.Affine,
    dam_lat: float,
    dam_lon: float,
    tmp_dir: str,
) -> np.ndarray:
    """
    Return a boolean mask (same shape as dem) of the upstream catchment above
    the dam location using pysheds D8 flow routing.
    """
    # Write DEM to temp file for pysheds
    dem_path = os.path.join(tmp_dir, "dem_conditioned.tif")
    rows, cols = dem.shape
    with rasterio.open(
        dem_path, "w",
        driver="GTiff", height=rows, width=cols,
        count=1, dtype="float32",
        crs=CRS.from_epsg(4326), transform=transform,
        nodata=-9999.0,
    ) as dst:
        arr = dem.copy()
        arr[np.isnan(arr)] = -9999.0
        dst.write(arr, 1)

    grid = Grid.from_raster(dem_path)
    raw = grid.read_raster(dem_path)

    filled = grid.fill_depressions(raw)
    conditioned = grid.resolve_flats(filled)
    fdir = grid.flowdir(conditioned)

    # Snap pour point to highest-accumulation cell nearby
    acc = grid.accumulation(fdir)
    dam_row, dam_col = _rowcol(transform, dam_lon, dam_lat)
    # Snap within a 5-pixel search window
    r0, r1 = max(0, dam_row - 5), min(rows, dam_row + 6)
    c0, c1 = max(0, dam_col - 5), min(cols, dam_col + 6)
    local_acc = np.array(acc)[r0:r1, c0:c1]
    local_acc[np.isnan(local_acc)] = 0
    lr, lc = np.unravel_index(np.argmax(local_acc), local_acc.shape)
    snap_row, snap_col = r0 + lr, c0 + lc

    # Convert snapped pixel back to geographic coords for pysheds catchment()
    snap_lon, snap_lat = transform * (snap_col + 0.5, snap_row + 0.5)

    catch = grid.catchment(
        x=snap_lon, y=snap_lat,
        fdir=fdir, xytype="coordinate",
    )
    mask = np.array(catch).astype(bool)
    return mask


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(curve: HAVCurve, path: str) -> None:
    """Write HAV curve to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "wse_masl", "area_m2", "volume_m3",
            "area_km2", "volume_Mm3",
        ])
        for wse, area, vol in zip(curve.wse_m, curve.area_m2, curve.volume_m3):
            writer.writerow([
                f"{wse:.2f}",
                f"{area:.1f}",
                f"{vol:.1f}",
                f"{area / 1e6:.4f}",
                f"{vol / 1e6:.4f}",
            ])
    print(f"CSV written: {path}")


def write_geojson(
    curve: HAVCurve,
    dem: np.ndarray,
    transform: rasterio.transform.Affine,
    catchment_mask: np.ndarray,
    dam_lat: float,
    dam_lon: float,
    path: str,
) -> None:
    """
    Write GeoJSON with two features:
      1. dam_location  – Point at (dam_lon, dam_lat)
      2. reservoir_extent – Polygon(s) of inundated area at full reservoir level
    """
    import rasterio.features
    from shapely.geometry import shape as shp_shape

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    crest = curve.crest_elevation_m
    dem_masked = dem.copy()
    dem_masked[~catchment_mask] = np.nan
    flood_mask = (dem_masked <= crest).astype(np.uint8)

    # Vectorise flood raster
    shapes = list(rasterio.features.shapes(
        flood_mask, mask=flood_mask, transform=transform
    ))
    polys = [shp_shape(s) for s, v in shapes if v == 1]
    if not polys:
        reservoir_geom = None
    elif len(polys) == 1:
        reservoir_geom = polys[0]
    else:
        reservoir_geom = unary_union(polys)

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [dam_lon, dam_lat]},
            "properties": {
                "name": "dam_location",
                "dam_elevation_masl": round(curve.dam_elevation_m, 2),
                "crest_elevation_masl": round(curve.crest_elevation_m, 2),
            },
        }
    ]
    if reservoir_geom is not None:
        features.append({
            "type": "Feature",
            "geometry": mapping(reservoir_geom),
            "properties": {
                "name": "reservoir_extent_full_level",
                "wse_masl": round(crest, 2),
                "area_m2": round(float(curve.area_m2[-1]), 1),
                "volume_m3": round(float(curve.volume_m3[-1]), 1),
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"GeoJSON written: {path}")


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run(
    dam_lat: float,
    dam_lon: float,
    dam_height_m: float,
    output_dir: str = "outputs",
    step_m: float = 2.0,
    buffer_deg: float = _DEFAULT_BUFFER_DEG,
    cache_dir: str | None = None,
) -> HAVCurve:
    """
    Full pipeline: fetch DEM, compute HAV curve, write CSV and GeoJSON.

    The DEM bounding box is expanded iteratively until the delineated upstream
    catchment fits entirely within the DEM extent (dynamic coverage fitting).
    Each expansion step adds _BUFFER_INCREMENT_DEG (~25 km) until the catchment
    no longer touches the grid boundary or _MAX_BUFFER_DEG is reached.

    Parameters
    ----------
    dam_lat, dam_lon : float
        Dam location (decimal degrees, WGS84).
    dam_height_m : float
        Maximum dam height above foundation (m).
    output_dir : str
        Directory for output files.
    step_m : float
        WSE sweep step in metres (default 2.0).
    buffer_deg : float
        Initial DEM download radius in degrees (default 0.35).
    cache_dir : str, optional
        Tile cache directory (reuse between runs to avoid re-downloading).

    Returns
    -------
    HAVCurve
    """
    tmp_dir = cache_dir or tempfile.mkdtemp(prefix="dambuhala_")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # --- Dynamic coverage fitting -------------------------------------------
    current_buffer = buffer_deg
    iteration = 0

    while True:
        iteration += 1
        print(f"[iter {iteration}] Fetching DEM (buffer={current_buffer:.2f} deg)...")
        dem, transform, crs = fetch_dem(dam_lat, dam_lon, current_buffer, tmp_dir)

        dam_row, dam_col = _rowcol(transform, dam_lon, dam_lat)
        dam_elev = float(dem[dam_row, dam_col])
        if np.isnan(dam_elev):
            raise ValueError(
                f"DEM value at dam location ({dam_lat}, {dam_lon}) is NaN. "
                "The site may be in a void area of the GLO-30 dataset."
            )

        print(f"[iter {iteration}] Delineating upstream catchment...")
        catchment_mask = _delineate_catchment(dem, transform, dam_lat, dam_lon, tmp_dir)

        if not _catchment_hits_boundary(catchment_mask):
            print(f"[iter {iteration}] Catchment contained within DEM extent. Proceeding.")
            break

        if current_buffer >= _MAX_BUFFER_DEG:
            print(
                f"[warn] Catchment still touches DEM boundary at maximum buffer "
                f"({_MAX_BUFFER_DEG} deg). Results may underestimate reservoir "
                f"volume. Consider increasing _MAX_BUFFER_DEG."
            )
            break

        next_buffer = min(current_buffer + _BUFFER_INCREMENT_DEG, _MAX_BUFFER_DEG)
        print(
            f"[iter {iteration}] Catchment touches DEM boundary — "
            f"expanding buffer to {next_buffer:.2f} deg."
        )
        current_buffer = next_buffer
    # ------------------------------------------------------------------------

    crest_elev = dam_elev + dam_height_m
    cell_area = _pixel_area_m2(transform, dam_lat)

    print(f"\nDam toe elevation:   {dam_elev:.1f} m (masl)")
    print(f"Dam crest elevation: {crest_elev:.1f} m (masl)")
    print(f"Final buffer:        {current_buffer:.2f} deg")
    print(f"Catchment pixels:    {catchment_mask.sum()}  |  Cell area: {cell_area:.1f} m^2")

    # --- HAV sweep -----------------------------------------------------------
    dem_masked = dem.copy()
    dem_masked[~catchment_mask] = np.nan

    wse_levels = np.arange(dam_elev, crest_elev + step_m, step_m)
    wse_levels = wse_levels[wse_levels <= crest_elev + 1e-6]

    areas = np.zeros(len(wse_levels))
    volumes = np.zeros(len(wse_levels))
    prev_wse, prev_area = dam_elev, 0.0

    for i, wse in enumerate(wse_levels):
        area = float(np.sum(dem_masked <= wse)) * cell_area
        areas[i] = area
        if i > 0:
            volumes[i] = volumes[i - 1] + 0.5 * (prev_area + area) * (wse - prev_wse)
        prev_wse, prev_area = wse, area

    curve = HAVCurve(
        wse_m=wse_levels,
        area_m2=areas,
        volume_m3=volumes,
        dam_elevation_m=dam_elev,
        crest_elevation_m=crest_elev,
    )

    # --- Outputs -------------------------------------------------------------
    csv_path = os.path.join(output_dir, "hav_curve.csv")
    geojson_path = os.path.join(output_dir, "reservoir.geojson")

    write_csv(curve, csv_path)
    write_geojson(curve, dem, transform, catchment_mask, dam_lat, dam_lon, geojson_path)

    print("\n--- HAV Curve Summary ---")
    print(f"  WSE range:  {wse_levels[0]:.1f} – {wse_levels[-1]:.1f} m (masl)")
    print(f"  Max area:   {areas[-1] / 1e6:.3f} km^2")
    print(f"  Max volume: {volumes[-1] / 1e6:.3f} Mm^3")

    return curve


# ---------------------------------------------------------------------------
# CLI usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Gened-2 Hydropower Project
    # RCC gravity dam, Kabugao, Apayao, Philippines
    run(
        dam_lat=18 + 3/60 + 32.4/3600,    # 18°03'32.4" N
        dam_lon=121 + 7/60 + 40.8/3600,   # 121°07'40.8" E
        dam_height_m=94.0,
        output_dir="outputs/gened2",
        step_m=2.0,
        cache_dir=".dem_cache",
    )

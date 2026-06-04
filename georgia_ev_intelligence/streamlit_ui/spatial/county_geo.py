"""GeoJSON helpers: county centroid lookup and point-in-polygon county inference.

Extracted from PAST_GEO_MAP_VIEW/backend/ingestion.py so the runtime spatial engine
can import these helpers without pulling in the full ingestion pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _iter_coordinates(geometry: dict) -> Iterable[Tuple[float, float]]:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if geom_type == "Polygon":
        for ring in coords:
            for lon, lat in ring:
                yield float(lat), float(lon)
    elif geom_type == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                for lon, lat in ring:
                    yield float(lat), float(lon)


def _geometry_rings(geometry: dict) -> Iterable[List[List[Tuple[float, float]]]]:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if geom_type == "Polygon":
        rings = [[(float(lon), float(lat)) for lon, lat in ring] for ring in coords]
        if rings:
            yield rings
    elif geom_type == "MultiPolygon":
        for polygon in coords:
            rings = [[(float(lon), float(lat)) for lon, lat in ring] for ring in polygon]
            if rings:
                yield rings


def _mean_lat_lon(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not points:
        return None
    arr = np.array(points, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def load_county_centroids(geojson_path: Path) -> Dict[str, Tuple[float, float]]:
    with Path(geojson_path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    centroids: Dict[str, Tuple[float, float]] = {}
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        centroid = _mean_lat_lon(list(_iter_coordinates(feature.get("geometry", {}))))
        if not centroid:
            continue
        county_name = (
            props.get("NAME10")
            or props.get("NAME")
            or props.get("NAMELSAD10", "").replace("County", "").strip()
        )
        if county_name:
            centroids[str(county_name).strip().lower()] = centroid
    return centroids


def load_county_geometries(geojson_path: Path) -> List[dict]:
    with Path(geojson_path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    geometries: List[dict] = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {}) or {}
        county_name = (
            props.get("NAME10")
            or props.get("NAME")
            or props.get("NAMELSAD10", "").replace("County", "").strip()
        )
        if not county_name:
            continue

        polygons = list(_geometry_rings(feature.get("geometry", {}) or {}))
        if not polygons:
            continue

        lon_values = [lon for polygon in polygons for ring in polygon for lon, _ in ring]
        lat_values = [lat for polygon in polygons for ring in polygon for _, lat in ring]
        geometries.append({
            "county": str(county_name).strip().title(),
            "county_key": str(county_name).strip().lower(),
            "polygons": polygons,
            "bbox": (min(lon_values), min(lat_values), max(lon_values), max(lat_values)),
        })
    return geometries


def _point_on_segment(lon: float, lat: float, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    ax, ay = a
    bx, by = b
    cross = (lat - ay) * (bx - ax) - (lon - ax) * (by - ay)
    if abs(cross) > 1e-10:
        return False
    dot = (lon - ax) * (bx - ax) + (lat - ay) * (by - ay)
    if dot < 0:
        return False
    squared_len = (bx - ax) ** 2 + (by - ay) ** 2
    return dot <= squared_len + 1e-10


def _point_in_ring(lon: float, lat: float, ring: Sequence[Tuple[float, float]]) -> bool:
    if len(ring) < 3:
        return False
    inside = False
    prev = ring[-1]
    for curr in ring:
        if _point_on_segment(lon, lat, prev, curr):
            return True
        xi, yi = curr
        xj, yj = prev
        intersects = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) + 1e-15) + xi
        )
        if intersects:
            inside = not inside
        prev = curr
    return inside


def _point_in_county_polygons(lon: float, lat: float, polygons: Sequence[Sequence[Tuple[float, float]]]) -> bool:
    for rings in polygons:
        if not rings:
            continue
        if not _point_in_ring(lon, lat, rings[0]):
            continue
        if any(_point_in_ring(lon, lat, hole) for hole in rings[1:]):
            continue
        return True
    return False


def infer_county_from_point(
    lat: Optional[float],
    lon: Optional[float],
    county_geometries: Sequence[dict],
) -> Optional[str]:
    if lat is None or lon is None:
        return None
    lat_f, lon_f = float(lat), float(lon)
    for county in county_geometries:
        min_lon, min_lat, max_lon, max_lat = county["bbox"]
        if not (min_lon <= lon_f <= max_lon and min_lat <= lat_f <= max_lat):
            continue
        if _point_in_county_polygons(lon_f, lat_f, county["polygons"]):
            return str(county["county"])
    return None

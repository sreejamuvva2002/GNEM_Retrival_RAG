"""Map domain models — the shape consumed by components/map_view.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MapContext:
    """Map view metadata used to drive overlays + dashboard widgets."""

    map_mode: str = "standard"
    focus_label: Optional[str] = None
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    radius_km: Optional[float] = None
    counties: List[str] = field(default_factory=list)
    county_coverage_count: int = 0
    gap_report: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "map_mode": self.map_mode,
            "focus_label": self.focus_label,
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "radius_km": self.radius_km,
            "counties": list(self.counties),
            "county_coverage_count": self.county_coverage_count,
            "gap_report": dict(self.gap_report),
        }


@dataclass
class MapResult:
    """Map view result combining the company records and the surrounding context."""

    records: List[Dict[str, Any]] = field(default_factory=list)
    context: MapContext = field(default_factory=MapContext)

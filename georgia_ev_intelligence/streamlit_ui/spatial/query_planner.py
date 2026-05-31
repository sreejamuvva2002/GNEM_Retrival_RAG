"""Query planner — classifies a user question and extracts spatial hints.

Ported verbatim from PAST_GEO_MAP_VIEW/backend/query_planner.py with no
external backend dependency. The planner consumes only Python stdlib + dataclasses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class QueryPlan:
    classification: str
    sql: bool
    geo: bool
    vector: bool
    hints: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "classification": self.classification,
            "sql": self.sql,
            "geo": self.geo,
            "vector": self.vector,
            "hints": self.hints,
        }


class QueryPlanner:
    GEO_KEYWORDS = {
        "near", "within", "distance", "km", "mile", "miles", "mi",
        "closest", "coordinate", "coordinates", "radius",
        "county", "counties", "map", "geospatial", "spatial",
    }
    SQL_KEYWORDS = {"top", "highest", "most", "employment", "employees", "industry", "list", "count", "show"}
    VECTOR_KEYWORDS = {
        "supplier", "suppliers", "battery", "oem", "relationship", "supply",
        "products", "manufacturers", "alternatives", "components",
    }
    GAP_KEYWORDS = {"gap", "gaps", "white space", "coverage", "under-served", "underserved", "missing"}
    DISRUPTION_KEYWORDS = {
        "disruption", "interrupt", "interruption", "shutdown", "outage",
        "alternative", "alternatives", "backup", "substitute", "resilience", "reroute",
    }
    FACILITY_CITY_ALIASES = {
        "kia west point": "West Point",
        "west point facility": "West Point",
        "port of savannah": "Savannah",
        "savannah port": "Savannah",
    }
    OEM_NAMES = [
        "general motors", "hyundai motor group", "mercedes-benz", "blue bird",
        "ford", "gm", "tesla", "rivian", "hyundai", "kia",
        "toyota", "honda", "nissan", "bmw", "mercedes",
        "stellantis", "volkswagen", "vw",
    ]
    CAPABILITY_SYNONYMS = {
        "battery": ["battery", "cell", "pack"],
        "thermal management": ["thermal", "hvac", "cooling", "thermal management"],
        "power electronics": ["power electronics", "inverter", "converter", "charging", "electronics"],
        "charging": ["charging", "charger", "charging infrastructure"],
        "wiring": ["wiring", "harness", "cable", "connector", "electrical architecture"],
        "stamping": ["stamping", "stamp", "pressed", "welded"],
        "seating": ["seating", "seat", "interior trim"],
        "materials": ["materials", "chemicals", "alloys", "rubber", "textile"],
        "sensors": ["sensor", "sensors", "braking", "control systems"],
    }

    def __init__(
        self,
        company_names: Optional[Sequence[str]] = None,
        county_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.company_names = sorted(
            {str(name).strip() for name in (company_names or []) if str(name).strip()},
            key=lambda value: len(value),
            reverse=True,
        )
        self.company_lookup = [
            (self._normalize_text(name), name)
            for name in self.company_names
            if self._normalize_text(name)
        ]
        self.county_names = sorted(
            {str(name).strip().title() for name in (county_names or []) if str(name).strip()},
            key=lambda value: len(value),
            reverse=True,
        )

    def plan(self, question: str) -> Dict[str, object]:
        text = question.strip()
        lower = text.lower()

        hints: Dict[str, object] = {}
        coords = self._extract_coordinates(lower)
        radius = self._extract_radius_km(lower)
        city = self._extract_city(text)
        counties = self._extract_counties(text)
        metric = self._extract_metric(lower)
        oem = self._extract_oem(lower)
        industry = self._extract_industry(text)
        category_term = self._extract_category_term(lower)
        capability_term = self._extract_capability_term(lower)
        facility_city = self._extract_facility_city(lower)
        company_name = self._extract_company_name(text)
        analysis_intent = self._extract_analysis_intent(lower)

        if coords:
            hints["coordinates"] = {"lat": coords[0], "lon": coords[1]}
        if radius:
            hints["radius_km"] = radius
        if city:
            hints["city"] = city
        if counties:
            hints["counties"] = counties
        if metric:
            hints["metric"] = metric
        if oem:
            hints["oem"] = oem
        if industry:
            hints["industry_group"] = industry
        if category_term:
            hints["category_term"] = category_term
        if capability_term:
            hints["capability_term"] = capability_term
        if facility_city:
            hints["city"] = facility_city
        if company_name:
            hints["company_name"] = company_name
        hints["analysis_intent"] = analysis_intent

        geo_signal = (
            bool(coords or city or facility_city or counties)
            or self._contains_keyword(lower, self.GEO_KEYWORDS)
            or analysis_intent in {"gap_analysis", "disruption_alternatives"}
        )
        sql_signal = (
            bool(metric or industry or oem or category_term or capability_term or company_name or counties)
            or ("top" in lower and "company" in lower)
            or ("list" in lower and "company" in lower)
            or analysis_intent in {"gap_analysis", "disruption_alternatives"}
        )
        vector_signal = (
            bool(oem or capability_term or company_name)
            or self._contains_keyword(lower, self.VECTOR_KEYWORDS)
            or analysis_intent in {"gap_analysis", "disruption_alternatives"}
        )

        if not any([sql_signal, geo_signal, vector_signal]):
            vector_signal = True

        classification = self._classify_query(
            geo_signal=geo_signal,
            sql_signal=sql_signal,
            vector_signal=vector_signal,
            analysis_intent=analysis_intent,
        )

        if geo_signal and "radius_km" not in hints and analysis_intent == "disruption_alternatives":
            hints["radius_km"] = 250.0
        elif geo_signal and "radius_km" not in hints and (coords or city or facility_city):
            hints["radius_km"] = 100.0

        return QueryPlan(
            classification=classification,
            sql=sql_signal,
            geo=geo_signal,
            vector=vector_signal,
            hints=hints,
        ).to_dict()

    @staticmethod
    def _contains_keyword(text: str, keywords: set) -> bool:
        return any(word in text for word in keywords)

    @staticmethod
    def _normalize_text(value: object) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _classify_query(geo_signal: bool, sql_signal: bool, vector_signal: bool, analysis_intent: str) -> str:
        if analysis_intent == "gap_analysis":
            return "SUPPLY_CHAIN_GAP_QUERY"
        if analysis_intent == "disruption_alternatives":
            return "DISRUPTION_ALTERNATIVES_QUERY"
        true_count = sum([sql_signal, geo_signal, vector_signal])
        if true_count > 1:
            return "HYBRID_QUERY"
        if sql_signal:
            return "SQL_QUERY"
        if geo_signal:
            return "GEO_QUERY"
        return "VECTOR_QUERY"

    @staticmethod
    def _extract_analysis_intent(text: str) -> str:
        if any(keyword in text for keyword in QueryPlanner.DISRUPTION_KEYWORDS):
            return "disruption_alternatives"
        if any(keyword in text for keyword in QueryPlanner.GAP_KEYWORDS):
            return "gap_analysis"
        return "standard_retrieval"

    @staticmethod
    def _extract_coordinates(text: str) -> Optional[Tuple[float, float]]:
        pattern = re.compile(r"(-?\d{1,2}(?:\.\d+)?)\s*[, ]\s*(-?\d{1,3}(?:\.\d+)?)")
        for match in pattern.finditer(text):
            lat = float(match.group(1))
            lon = float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        return None

    @staticmethod
    def _extract_radius_km(text: str) -> Optional[float]:
        match = re.search(r"(\d+(?:\.\d+)?)\s*km\b", text)
        if match:
            return float(match.group(1))
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:miles?|mi)\b", text)
        if match:
            return float(match.group(1)) * 1.60934
        return None

    @staticmethod
    def _extract_city(text: str) -> Optional[str]:
        patterns = [
            r"\bnear\s+([A-Za-z][A-Za-z\s\-']+?)(?:[?.!,]|$)",
            r"\baround\s+([A-Za-z][A-Za-z\s\-']+?)(?:[?.!,]|$)",
            r"\bclosest\s+to\s+([A-Za-z][A-Za-z\s\-']+?)(?:[?.!,]|$)",
            r"\bwithin\s+\d+(?:\.\d+)?\s*(?:km|miles?|mi)\s+of\s+([A-Za-z][A-Za-z\s\-']+?)(?:[?.!,]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                city = match.group(1).strip()
                if "km" not in city.lower() and "mile" not in city.lower() and not city.lower().endswith("county"):
                    return city.title()
        return None

    def _extract_counties(self, text: str) -> List[str]:
        out: List[str] = []
        normalized_text = f" {self._normalize_text(text)} "
        if self.county_names:
            for county_name in self.county_names:
                normalized_county = self._normalize_text(county_name)
                if not normalized_county or county_name in out:
                    continue
                if (
                    f" {normalized_county} county " in normalized_text
                    or f" {normalized_county} counties " in normalized_text
                    or f" {normalized_county} " in normalized_text
                ):
                    out.append(county_name)
            return out
        county_matches = re.findall(
            r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,2})\s+County\b",
            text,
        )
        for value in county_matches:
            county = value.strip().title()
            if county and county not in out:
                out.append(county)
        return out

    @staticmethod
    def _extract_metric(text: str) -> Optional[str]:
        if "employment" in text or "employees" in text:
            return "employment"
        return None

    @staticmethod
    def _extract_category_term(text: str) -> Optional[str]:
        if "tier 2/3" in text or "tier 2 3" in text:
            return "Tier 2/3"
        if "tier 1/2" in text or "tier 1 2" in text:
            return "Tier 1/2"
        if "tier 2" in text:
            return "Tier 2"
        if "tier 1" in text:
            return "Tier 1"
        if "oem footprint" in text or "oem operations" in text:
            return "OEM"
        return None

    def _extract_capability_term(self, text: str) -> Optional[str]:
        for canonical, variants in self.CAPABILITY_SYNONYMS.items():
            if any(variant in text for variant in variants):
                return canonical
        return None

    def _extract_oem(self, text: str) -> Optional[str]:
        for oem in self.OEM_NAMES:
            if oem in text:
                if oem == "gm":
                    return "GM"
                if oem == "vw":
                    return "VW"
                return oem.title()
        return None

    def _extract_facility_city(self, text: str) -> Optional[str]:
        for phrase, city in self.FACILITY_CITY_ALIASES.items():
            if phrase in text:
                return city
        return None

    def _extract_company_name(self, text: str) -> Optional[str]:
        normalized_text = f" {self._normalize_text(text)} "
        for normalized_company, company_name in self.company_lookup:
            if normalized_company and f" {normalized_company} " in normalized_text:
                return company_name
        return None

    @staticmethod
    def _extract_industry(text: str) -> Optional[str]:
        match = re.search(
            r"industry(?:\s+group)?\s+(?:is\s+|=|:)?([A-Za-z0-9\s/&-]+)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip().title()
        return None

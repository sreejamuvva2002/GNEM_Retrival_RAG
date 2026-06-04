"""Seed URL sets for all three source tiers.

Priority order: A (company sites) → B (news) → C (government/regulatory).
Each seed is a dict: {url, source_type, linked_company_id (optional)}.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Tier A — Company websites from the Excel KB
# ---------------------------------------------------------------------------

def _excel_kb_path() -> Path:
    """Locate the normalised KB workbook (falls back to raw GNEM file)."""
    from georgia_ev_intelligence.shared import config  # avoid circular at module load
    normalised = config.OUTPUTS_DIR / "Normalized_kb.xlsx"
    if normalised.exists():
        return normalised
    raw = config.KB_DIR / "GNEM - Auto Landscape Lat Long Updated.xlsx"
    if raw.exists():
        return raw
    raise FileNotFoundError(
        "Cannot find GNEM Excel KB. Run the normalisation step first."
    )


def company_site_seeds() -> list[dict]:
    """Read the 'Website' column from the Excel KB and return seed dicts."""
    path = _excel_kb_path()
    df = pd.read_excel(path, engine="openpyxl")

    # Normalise column names to lower-case stripped
    df.columns = [str(c).strip().lower() for c in df.columns]

    website_col = next(
        (c for c in df.columns if "website" in c or "url" in c),
        None,
    )
    company_col = next(
        (c for c in df.columns if "company" in c),
        None,
    )
    row_id_col = next(
        (c for c in df.columns if "row" in c and "id" in c),
        None,
    )

    seeds: list[dict] = []
    for idx, row in df.iterrows():
        url = str(row.get(website_col, "")).strip() if website_col else ""
        if not url or url.lower() in ("nan", "none", ""):
            continue
        if not url.startswith("http"):
            url = "https://" + url

        company_id: Optional[str] = None
        if row_id_col:
            rid = row.get(row_id_col)
            if rid is not None:
                company_id = f"KB_ROW_{int(rid):04d}"

        seeds.append({
            "url": url,
            "source_type": "company_site",
            "linked_company_id": company_id,
        })

    return seeds


# ---------------------------------------------------------------------------
# Tier B — News / press sites
# ---------------------------------------------------------------------------

NEWS_SEEDS: list[dict] = [
    # Georgia Power EV programme
    {"url": "https://www.georgiapower.com/company/green-power-energy/electric-vehicles.html",
     "source_type": "news"},
    {"url": "https://www.georgiapower.com/news.html",
     "source_type": "news"},

    # Georgia Environmental Finance Authority (GEFA) — state EV grants
    {"url": "https://gefa.georgia.gov/energy/electric-vehicles",
     "source_type": "news"},

    # Electrek Georgia tag
    {"url": "https://electrek.co/tag/georgia/",
     "source_type": "news"},

    # InsideEVs manufacturer news
    {"url": "https://insideevs.com/news/category/ev-news/",
     "source_type": "news"},

    # Green Car Reports
    {"url": "https://www.greencarreports.com/electric-car",
     "source_type": "news"},
]


# ---------------------------------------------------------------------------
# Tier C — Government / regulatory documents
# ---------------------------------------------------------------------------

GOV_SEEDS: list[dict] = [
    # DOE Alternative Fuels Station Locator (Georgia)
    {"url": "https://afdc.energy.gov/stations#/find/nearest?fuel=ELEC&country=US&state=GA",
     "source_type": "gov_doc"},

    # DOE / NREL EV resources
    {"url": "https://www.energy.gov/eere/vehicles/electric-vehicles",
     "source_type": "gov_doc"},

    # EPA vehicle emissions
    {"url": "https://www.epa.gov/greenvehicles",
     "source_type": "gov_doc"},

    # FHWA EV infrastructure (NEVI)
    {"url": "https://www.fhwa.dot.gov/environment/alternative_fuel_vehicles/",
     "source_type": "gov_doc"},

    # Georgia DCA EV / transportation
    {"url": "https://www.dca.ga.gov/community-economic-development/",
     "source_type": "gov_doc"},

    # GEMA (Georgia Emergency Management & Homeland Security) energy resilience
    {"url": "https://gema.georgia.gov/",
     "source_type": "gov_doc"},

    # Georgia Dept of Transportation — freight / supply chain
    {"url": "https://www.dot.ga.gov/PartnerSmart/FreightLogistics",
     "source_type": "gov_doc"},
]


# ---------------------------------------------------------------------------
# Combined — A → B → C priority
# ---------------------------------------------------------------------------

def all_seeds() -> list[dict]:
    """Return all seed URLs in priority order: company → news → gov."""
    tier_a = company_site_seeds()
    tier_b = NEWS_SEEDS
    tier_c = GOV_SEEDS
    return tier_a + tier_b + tier_c

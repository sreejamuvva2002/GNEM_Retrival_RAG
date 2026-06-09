# Map and Boundary Sources

The Streamlit UI keeps simplified boundary files locally so the boundary
overlays render quickly and do not depend on a live GIS request.

## Basemap

- OpenStreetMap standard tiles
- URL: https://tile.openstreetmap.org/
- Attribution: https://www.openstreetmap.org/copyright

## Georgia County Boundaries

- ArcGIS `Counties_Georgia` feature layer
- Source: https://services5.arcgis.com/Mm1BAVEiAXrYE9vn/ArcGIS/rest/services/Counties_Georgia/FeatureServer/0
- Local UI file: `Counties_Georgia_simplified.geojson`
- Contains all 159 Georgia counties and county names.

## Georgia State Boundary

- U.S. Census Bureau TIGERweb `State_County` map service, States layer
- Source: https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/0
- Local UI file: `Georgia_state_boundary.geojson`
- Filter: Georgia (`STATE='13'`)

The local files were exported on June 9, 2026 with geometry simplification for
web display. Source links are also shown in the map attribution inside the UI.

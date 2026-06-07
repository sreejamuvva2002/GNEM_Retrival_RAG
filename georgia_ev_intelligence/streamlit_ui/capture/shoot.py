"""Drive the capture harness with Playwright and save one PNG per screen state.

Prereq: the harness must be running (see capture_app.py), e.g.

    streamlit run georgia_ev_intelligence/streamlit_ui/capture/capture_app.py \
        --server.port 8531 --server.headless true

Then:

    python georgia_ev_intelligence/streamlit_ui/capture/shoot.py \
        --base-url http://localhost:8531 --out outputs/figma_shots

Each shot waits for Streamlit to finish running and for the map/leaflet iframe
tiles to settle before screenshotting the app container.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# (state, viewport_w, viewport_h, filename, expand_sidebar)
DESKTOP_W, DESKTOP_H = 1440, 900
MOBILE_W, MOBILE_H = 390, 844

SHOTS = [
    ("empty", DESKTOP_W, DESKTOP_H, "01-empty.png", False),
    ("chat", DESKTOP_W, DESKTOP_H, "02-chat.png", False),
    ("loading", DESKTOP_W, DESKTOP_H, "03-loading.png", False),
    ("sources", DESKTOP_W, DESKTOP_H, "04-sources.png", False),
    ("settings", DESKTOP_W, DESKTOP_H, "05-settings.png", False),
    ("sidebar", DESKTOP_W, DESKTOP_H, "06-sidebar.png", True),
    ("empty", MOBILE_W, MOBILE_H, "07-empty-mobile.png", False),
    ("chat", MOBILE_W, MOBILE_H, "08-chat-mobile.png", False),
]


def _wait_ready(page) -> None:
    """Wait until Streamlit has finished its run and the layout has settled."""
    page.wait_for_selector('[data-testid="stApp"]', timeout=30000)
    # Streamlit sets data-test-script-state="running" while a rerun is in flight.
    try:
        page.wait_for_selector(
            '[data-testid="stApp"][data-test-script-state="notRunning"]',
            timeout=15000,
        )
    except Exception:
        pass
    page.wait_for_load_state("networkidle")


def shoot(base_url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for state, vw, vh, fname, expand_sidebar in SHOTS:
            ctx = browser.new_context(
                viewport={"width": vw, "height": vh},
                device_scale_factor=2,  # retina-quality PNGs
            )
            page = ctx.new_page()
            page.goto(f"{base_url}/?state={state}", wait_until="domcontentloaded")
            _wait_ready(page)

            if expand_sidebar:
                # Sidebar starts expanded for this state, but make sure the collapse
                # control didn't hide it; click the expand control if present.
                for sel in (
                    '[data-testid="stSidebarCollapsedControl"]',
                    '[data-testid="collapsedControl"]',
                ):
                    try:
                        el = page.query_selector(sel)
                        if el and el.is_visible():
                            el.click()
                            page.wait_for_timeout(400)
                    except Exception:
                        pass

            # Let the divider script size columns + Leaflet finish tiling, and the
            # loading-card / settings-dialog animations reach a stable frame.
            page.wait_for_timeout(2600 if state in {"chat", "sources"} else 1800)

            target = page.query_selector('[data-testid="stApp"]') or page
            target.screenshot(path=str(out_dir / fname))
            print(f"  saved {fname}  ({state} @ {vw}x{vh})")
            ctx.close()
        browser.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8531")
    ap.add_argument("--out", default="outputs/figma_shots")
    args = ap.parse_args()
    t0 = time.time()
    shoot(args.base_url, Path(args.out))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

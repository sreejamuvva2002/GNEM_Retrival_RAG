"""Draggable divider + full-height flex layout (React parity).

React's layout is a full-height flex row: a left chat panel (input pinned at its
bottom, messages and inline sources scrolling above) + a `w-1` divider + a
right map panel. Streamlit can't do resizable, full-height columns,
so one injected script (runs on load / window-resize / drag, holding refs to the
two columns) does it all:

  * a fixed, full-height divider bar centered in the inter-column gap, draggable
    to resize the two `stColumn` flex widths live (client-side, no rerun), with
    a temporary drag shield so the map iframe cannot swallow mouse events;
  * `layoutHeights()` makes the columns row fill the viewport, the chat messages
    container scroll (`.st-key-chat_scroll`) with the inline input pinned below,
    and the map iframe fill its column — sized in px on the iframe + its ancestor
    chain (anchored by the dependable `.st-key-gnem_map` class). Resizing the
    iframe element natively fires `resize` inside it, so Leaflet re-tiles.

State lives on `document` so the latest rerun's script always controls the
persistent divider element; the split ratio is persisted in localStorage.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

_ANCHOR_ID = "gnem-split-anchor"


def anchor() -> None:
    """Render the sentinel marker — call at the top of the chat column."""
    st.markdown(f"<div id='{_ANCHOR_ID}'></div>", unsafe_allow_html=True)


def render(min_pct: int = 20, max_pct: int = 80) -> None:
    """Inject the divider + layout script — call once after the columns build."""
    components.html(
        f"""
        <script>
        (function() {{
            const doc = window.parent.document;
            const win = window.parent;
            const STORE = 'gnemSplitPct';
            const MIN = {min_pct}, MAX = {max_pct};
            const GAP = 8;

            function clamp(v) {{ return Math.min(MAX, Math.max(MIN, v)); }}
            function stored() {{
                const v = parseFloat(localStorage.getItem(STORE));
                return isNaN(v) ? 50 : clamp(v);
            }}
            function setPx(el, h) {{
                el.style.setProperty('height', h + 'px', 'important');
                el.style.setProperty('min-height', h + 'px', 'important');
            }}

            // Persistent fixed divider bar (top→bottom), created once.
            function divider() {{
                let d = doc.getElementById('gnem-divider');
                if (!d) {{
                    d = doc.createElement('div');
                    d.id = 'gnem-divider';
                    d.style.cssText = 'position:fixed;top:0;height:100vh;width:14px;'
                        + 'margin-left:-7px;cursor:col-resize;z-index:1000;display:flex;'
                        + 'align-items:center;justify-content:center;';
                    const bar = doc.createElement('div');
                    bar.className = 'gnem-divider-bar';
                    bar.style.cssText = 'width:2px;height:100%;background:#e2e8f0;'
                        + 'transition:background .15s ease;';
                    d.appendChild(bar);
                    doc.body.appendChild(d);
                }}
                return d;
            }}
            function bar() {{ return divider().querySelector('.gnem-divider-bar'); }}
            function dragShield() {{
                let shield = doc.getElementById('gnem-drag-shield');
                if (!shield) {{
                    shield = doc.createElement('div');
                    shield.id = 'gnem-drag-shield';
                    shield.style.cssText = 'position:fixed;inset:0;display:none;'
                        + 'cursor:col-resize;z-index:999;background:transparent;';
                    doc.body.appendChild(shield);
                }}
                return shield;
            }}

            function applyWidths(cols, p) {{
                cols[0].style.flex = '0 0 calc(' + p + '% - 0.5rem)';
                cols[0].style.width = 'calc(' + p + '% - 0.5rem)';
                cols[1].style.flex = '0 0 calc(' + (100 - p) + '% - 0.5rem)';
                cols[1].style.width = 'calc(' + (100 - p) + '% - 0.5rem)';
            }}

            // Position the divider bar centered in the inter-column gap.
            function placeDivider() {{
                const stt = doc.__gnemSplitState;
                if (!stt) return;
                const r = stt.cols[0].getBoundingClientRect();
                const r2 = stt.cols[1].getBoundingClientRect();
                divider().style.left = ((r.right + r2.left) / 2) + 'px';
            }}

            // Full-height flex layout: columns fill the viewport, chat messages
            // scroll with the input pinned below, map fills its column.
            function layoutHeights() {{
                const stt = doc.__gnemSplitState;
                if (!stt) return;
                const block = stt.block;
                const top = block.getBoundingClientRect().top;
                const H = Math.max(360, win.innerHeight - top - GAP);

                block.style.setProperty('height', H + 'px', 'important');
                block.style.setProperty('overflow', 'hidden', 'important');
                stt.cols.forEach(function(c) {{
                    c.style.setProperty('height', H + 'px', 'important');
                    c.style.setProperty('display', 'flex', 'important');
                    c.style.setProperty('flex-direction', 'column', 'important');
                    const vb = c.querySelector('[data-testid="stVerticalBlock"]');
                    if (vb) {{
                        vb.style.setProperty('height', '100%', 'important');
                        vb.style.setProperty('flex', '1 1 auto', 'important');
                    }}
                }});

                // Chat: messages scroll, input pinned at the bottom of the column.
                const scroll = doc.querySelector('.st-key-chat_scroll');
                if (scroll) {{
                    // Streamlit wraps keyed containers in an stLayoutWrapper.
                    // That wrapper must grow too, otherwise the scroll area
                    // collapses to its content and leaves the input mid-column.
                    const scrollHost = scroll.parentElement;
                    if (scrollHost) {{
                        scrollHost.style.setProperty('display', 'flex', 'important');
                        scrollHost.style.setProperty('flex', '1 1 0', 'important');
                        scrollHost.style.setProperty('min-height', '0', 'important');
                        scrollHost.style.setProperty('overflow', 'hidden', 'important');
                    }}
                    const input = doc.querySelector('.st-key-chat_input');
                    const colRect = stt.cols[0].getBoundingClientRect();
                    const scrollTop = scroll.getBoundingClientRect().top;
                    const inputH = input ? input.getBoundingClientRect().height : 64;
                    const avail = Math.max(160, colRect.bottom - scrollTop - inputH - GAP);
                    if (scrollHost) {{
                        scrollHost.style.setProperty('height', avail + 'px', 'important');
                        scrollHost.style.setProperty('max-height', avail + 'px', 'important');
                    }}
                    scroll.style.setProperty('flex', '1 1 0', 'important');
                    scroll.style.setProperty('height', avail + 'px', 'important');
                    scroll.style.setProperty('max-height', avail + 'px', 'important');
                    scroll.style.setProperty('min-height', '0', 'important');
                    scroll.style.setProperty('overflow-y', 'auto', 'important');
                    scroll.style.setProperty('overflow-x', 'hidden', 'important');
                }}

                // Map: size the iframe + its ancestor chain (up to .st-key-gnem_map).
                const mapWrap = doc.querySelector('.st-key-gnem_map');
                if (mapWrap) {{
                    const iframe = mapWrap.querySelector('iframe');
                    if (iframe) {{
                        const mTop = iframe.getBoundingClientRect().top;
                        const avail = Math.max(240, win.innerHeight - mTop - GAP);
                        const mapH = avail;
                        let n = iframe;
                        while (n) {{
                            setPx(n, mapH);
                            n.style.setProperty('width', '100%', 'important');
                            if (n === mapWrap) break;
                            n = n.parentElement;
                        }}
                    }}
                }}
            }}

            function relayout() {{ placeDivider(); layoutHeights(); }}

            function onMove(e) {{
                if (!doc.__gnemDragging) return;
                const stt = doc.__gnemSplitState;
                if (!stt) return;
                const rect = stt.block.getBoundingClientRect();
                const np = clamp(((e.clientX - rect.left) / rect.width) * 100);
                stt.p = np;
                applyWidths(stt.cols, np);
                placeDivider();  // widths drive the map iframe (100%) → Leaflet re-tiles
            }}
            function onUp() {{
                if (!doc.__gnemDragging) return;
                doc.__gnemDragging = false;
                doc.body.style.cursor = '';
                doc.body.style.userSelect = '';
                dragShield().style.display = 'none';
                bar().style.background = '#e2e8f0';
                if (doc.__gnemSplitState) localStorage.setItem(STORE, doc.__gnemSplitState.p);
            }}

            function setup() {{
                if (doc.__gnemDragging) return true;  // never clobber an active drag
                const a = doc.getElementById('{_ANCHOR_ID}');
                if (!a) return false;
                const leftCol = a.closest('[data-testid="stColumn"]');
                if (!leftCol) return false;
                const block = leftCol.parentElement;
                const cols = block.querySelectorAll(':scope > [data-testid="stColumn"]');
                if (cols.length < 2) return false;

                const p = stored();
                doc.__gnemSplitState = {{ cols: cols, block: block, p: p }};
                applyWidths(cols, p);

                const d = divider();
                const b = bar();
                d.onmouseenter = function() {{ b.style.background = '#94a3b8'; }};
                d.onmouseleave = function() {{ if (!doc.__gnemDragging) b.style.background = '#e2e8f0'; }};
                d.onmousedown = function(e) {{
                    doc.__gnemDragging = true;
                    doc.body.style.cursor = 'col-resize';
                    doc.body.style.userSelect = 'none';
                    dragShield().style.display = 'block';
                    b.style.background = '#64748b';
                    e.preventDefault();
                }};

                if (doc.__gnemMove) doc.removeEventListener('mousemove', doc.__gnemMove);
                if (doc.__gnemUp) doc.removeEventListener('mouseup', doc.__gnemUp);
                if (doc.__gnemResize) win.removeEventListener('resize', doc.__gnemResize);
                if (doc.__gnemBlur) win.removeEventListener('blur', doc.__gnemBlur);
                doc.__gnemMove = onMove;
                doc.__gnemUp = onUp;
                doc.__gnemResize = relayout;
                doc.__gnemBlur = onUp;
                doc.addEventListener('mousemove', onMove);
                doc.addEventListener('mouseup', onUp);
                win.addEventListener('resize', relayout);
                win.addEventListener('blur', onUp);

                relayout();
                return true;
            }}

            // Retry until the columns (and the async-rendered map iframe) exist,
            // re-applying layout as late reflows land, then stop.
            let tries = 0;
            const iv = setInterval(function() {{
                tries += 1;
                const ok = setup();
                if ((ok && tries > 14) || tries > 80) clearInterval(iv);
            }}, 80);
        }})();
        </script>
        """,
        height=0,
    )

"""CSS injection — emit one <style> block matching the React rag-chat-ui look.

The CSS is generated against the active palette so theme switches re-render
with the right colors. We hide Streamlit chrome (menu, footer, deploy button)
and rebuild the top header / sidebar / message bubbles to match the React UI.

Design tokens (radius / spacing / shadow / z-index) live in colors.TOKENS and
are emitted here as CSS variables so component rules consume `var(--radius-*)`
etc. instead of scattering magic numbers.
"""
from __future__ import annotations

import streamlit as st

from .colors import TOKENS, palette


def inject_styles(is_dark: bool = False, compact: bool = False) -> None:
    p = palette(False)
    t = TOKENS
    chat_font_size = "0.95rem"

    st.markdown(
        f"""
        <style>
        #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{
            visibility: hidden;
        }}

        :root {{
            --bg: {p['background']};
            --fg: {p['foreground']};
            --card: {p['card']};
            --card-fg: {p['card_foreground']};
            --primary: {p['primary']};
            --primary-fg: {p['primary_foreground']};
            --secondary: {p['secondary']};
            --muted: {p['muted']};
            --muted-fg: {p['muted_foreground']};
            --accent: {p['accent']};
            --border: {p['border']};
            --sidebar: {p['sidebar']};
            --sidebar-border: {p['sidebar_border']};
            --sidebar-accent: {p['sidebar_accent']};
            --glass-bg: {p['glass_bg']};
            --glass-border: {p['glass_border']};
            --user-bubble: {p['user_bubble']};
            --user-bubble-text: {p['user_bubble_text']};
            --assistant-bubble: {p['assistant_bubble']};
            --citation-bg: {p['citation_bg']};
            --citation-text: {p['citation_text']};
            --success: {p['success']};
            --warning: {p['warning']};
            --destructive: {p['destructive']};

            /* Loading-card accents (React from-blue-500 to-indigo-600) */
            --accent-blue: {p['accent_blue']};
            --accent-indigo: {p['accent_indigo']};
            --accent-blue-light: {p['accent_blue_light']};
            --accent-emerald: {p['accent_emerald']};
            --loading-track: {p['loading_track']};

            /* ===== Design tokens ===== */
            --radius-sm: {t['radius_sm']};
            --radius-md: {t['radius_md']};
            --radius-lg: {t['radius_lg']};
            --radius-xl: {t['radius_xl']};
            --radius-pill: {t['radius_pill']};
            --space-1: {t['space_1']};
            --space-2: {t['space_2']};
            --space-3: {t['space_3']};
            --space-4: {t['space_4']};
            --space-5: {t['space_5']};
            --space-6: {t['space_6']};
            --shadow-sm: {t['shadow_sm']};
            --shadow-md: {t['shadow_md']};
            --shadow-lg: {t['shadow_lg']};
            --z-header: {t['z_header']};
            --z-overlay: {t['z_overlay']};
        }}

        html, body, .stApp {{
            color: var(--fg) !important;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}

        /* Chat-panel background stack (React chat-panel.tsx): base slate→white
           gradient + two radial tints + a faint dot grid. Applied app-wide; the
           map iframe covers the right half so the gradient reads as the chat bg. */
        .stApp {{
            background-color: #f1f5f9 !important;
            background-image:
                radial-gradient(ellipse at top right, rgba(239, 246, 255, 0.4), transparent 60%),
                radial-gradient(ellipse at bottom left, rgba(238, 242, 255, 0.3), transparent 60%),
                radial-gradient(circle at 1px 1px, rgba(203, 213, 225, 0.4) 1px, transparent 0),
                linear-gradient(to bottom right, #f1f5f9, #f8fafc 50%, #ffffff) !important;
            background-size: 100% 100%, 100% 100%, 24px 24px, 100% 100% !important;
            background-attachment: fixed !important;
        }}

        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        [data-testid="stMainBlockContainer"] {{
            padding: 0.75rem 1.25rem !important;
            max-width: 100% !important;
        }}

        /* ===== Accessibility: visible keyboard focus everywhere ===== */
        a:focus-visible,
        button:focus-visible,
        input:focus-visible,
        select:focus-visible,
        [role="button"]:focus-visible {{
            outline: 2px solid var(--primary) !important;
            outline-offset: 2px !important;
            border-radius: var(--radius-sm);
        }}

        /* ===== Respect reduced-motion preferences ===== */
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                animation-duration: 0.001ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.001ms !important;
                scroll-behavior: auto !important;
            }}
        }}

        [data-testid="stSidebar"] {{
            display: none !important;
        }}

        /* Apply Inter only to text containers. Including span/div forces
           Streamlit's Material-Icons spans (e.g. expander chevrons) to use
           Inter, which renders the literal ligature text "keyboard_arrow_right"
           instead of the chevron glyph. */
        h1, h2, h3, h4, h5, h6, p, label, button, input, textarea {{
            font-family: 'Inter', 'Geist', ui-sans-serif, system-ui, -apple-system, sans-serif !important;
        }}

        /* ===== Header / GNEM badge =====
           The full glass header bar is gone — only the small blue GNEM pill
           remains. The wrapper just provides a top anchor and a positioning
           context for the hover tooltip. */
        .gnem-header,
        .gnem-header__brand {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
        }}
        /* ===== Chat-panel header (badge + title + subtitle) ===== */
        .chat-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.55rem 0.25rem 0.85rem 0.25rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }}
        .chat-header__title {{
            font-weight: 600;
            font-size: 1rem;
            color: #1e293b;
            line-height: 1.2;
        }}
        .chat-header__subtitle {{
            font-size: 0.75rem;
            color: var(--muted-fg);
            margin-top: 1px;
        }}
        .gnem-logo {{
            min-width: 56px;
            height: 34px;
            padding: 0 0.75rem;
            border-radius: var(--radius-sm);
            background: linear-gradient(135deg, #1e293b, #0f172a);
            color: #ffffff;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.8rem;
            letter-spacing: 0.04em;
            box-shadow: 0 4px 6px -1px rgba(148, 163, 184, 0.5);
        }}
        /* Hover tooltip for the GNEM badge — uses a sibling <span> because
           Streamlit's HTML sanitizer strips custom data-* attributes (so
           `content: attr(data-tooltip)` resolved to empty). */
        .gnem-logo-wrap {{
            position: relative;
            display: inline-flex;
            outline: none;
        }}
        .gnem-tooltip {{
            position: absolute;
            top: calc(100% + 8px);
            left: 0;
            padding: 0.4rem 0.7rem;
            border-radius: var(--radius-md);
            background: var(--fg);
            color: var(--bg);
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            white-space: nowrap;
            box-shadow: var(--shadow-md);
            opacity: 0;
            pointer-events: none;
            transform: translateY(-4px);
            transition: opacity 160ms ease, transform 160ms ease;
            z-index: var(--z-overlay);
        }}
        .gnem-logo-wrap:hover .gnem-tooltip,
        .gnem-logo-wrap:focus-within .gnem-tooltip,
        .gnem-logo-wrap:focus-visible .gnem-tooltip {{
            opacity: 1;
            transform: translateY(0);
        }}
        .gnem-header__title {{
            font-weight: 700;
            font-size: 0.92rem;
            color: var(--fg);
            letter-spacing: -0.01em;
        }}

        /* ===== Glass cards ===== */
        .glass-card {{
            background: var(--glass-bg);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            padding: var(--space-3) var(--space-4);
            box-shadow: var(--shadow-sm);
            transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
        }}
        .glass-card:hover {{
            border-color: var(--primary);
            box-shadow: var(--shadow-lg);
        }}

        /* ===== Chat bubbles ===== */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .chat-row {{
            display: flex;
            gap: 0.7rem;
            margin-bottom: 1.25rem;
            animation: fadeInUp 0.4s ease both;
        }}
        .chat-row--user {{
            justify-content: flex-end;
        }}
        .chat-col {{
            display: flex;
            flex-direction: column;
            max-width: 85%;
        }}
        .chat-row--user .chat-col {{ align-items: flex-end; }}
        .chat-avatar {{
            width: 32px;
            height: 32px;
            border-radius: var(--radius-pill);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            font-weight: 800;
            flex-shrink: 0;
        }}
        .chat-avatar--user {{
            background: var(--user-bubble);
            color: var(--user-bubble-text);
        }}
        .chat-avatar--assistant {{
            background: var(--citation-bg);
            color: var(--primary);
        }}
        .chat-bubble {{
            padding: 0.9rem 1.15rem;
            border-radius: var(--radius-lg);
            font-size: {chat_font_size};
            line-height: 1.65;
            color: var(--fg);
            overflow-wrap: anywhere;
        }}
        .chat-bubble--user {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            color: #ffffff;
            box-shadow: 0 10px 15px -3px rgba(148, 163, 184, 0.3);
        }}
        .chat-bubble--assistant {{
            background: #ffffff;
            color: #334155;
            border: 1px solid rgba(226, 232, 240, 0.8);
            box-shadow: 0 4px 6px -1px rgba(226, 232, 240, 0.5);
        }}
        .chat-bubble p {{ margin: 0 0 0.45rem 0; }}
        .chat-bubble ul, .chat-bubble ol {{ margin: 0.2rem 0 0.5rem 1.1rem; padding: 0; }}
        .chat-bubble strong {{ color: var(--fg); }}
        .chat-meta {{
            font-size: 0.7rem;
            color: var(--muted-fg);
            margin-top: 0.2rem;
            padding: 0 0.25rem;
        }}

        /* ===== Citation chips ===== */
        .citation-chip {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 20px;
            height: 18px;
            margin: 0 1px;
            padding: 0 6px;
            border-radius: var(--radius-sm);
            background: var(--citation-bg);
            color: var(--citation-text);
            font-size: 0.74rem;
            font-weight: 700;
            text-decoration: none;
        }}

        /* ===== Empty state ===== */
        .empty-shell {{
            text-align: center;
            padding: 4rem 1rem 0;
            max-width: 760px;
            margin: 0 auto;
        }}
        .empty-logo {{
            width: 64px;
            height: 64px;
            border-radius: var(--radius-lg);
            background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
            color: #94a3b8;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 2px 4px 0 rgba(15, 23, 42, 0.06);
        }}
        .empty-title {{
            margin-top: 1.25rem;
            font-size: 1.25rem;
            font-weight: 600;
            color: #334155;
            letter-spacing: -0.01em;
        }}
        .empty-subtitle {{
            margin-top: 0.5rem;
            margin-bottom: 0.9rem;
            font-size: 0.875rem;
            line-height: 1.6;
            color: #64748b;
            max-width: 24rem;
            margin-left: auto;
            margin-right: auto;
        }}

        /* ===== Suggested-question cards (rendered as st.button, key=suggest_*) ===== */
        [class*="st-key-suggest_"] {{
            margin-top: 0.25rem;
        }}
        [class*="st-key-suggest_"] button {{
            min-height: 58px !important;
            height: 100% !important;
            justify-content: flex-start !important;
            text-align: left !important;
            padding: 0.7rem 0.95rem !important;
            border-radius: var(--radius-md) !important;
            background: var(--glass-bg) !important;
            border: 1px solid var(--glass-border) !important;
            color: var(--fg) !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
            white-space: normal !important;
            transition: background 140ms ease, border-color 140ms ease, transform 140ms ease, box-shadow 140ms ease;
        }}
        [class*="st-key-suggest_"] button p {{
            text-align: left !important;
            white-space: normal !important;
        }}
        [class*="st-key-suggest_"] button:hover {{
            background: var(--citation-bg) !important;
            border-color: var(--primary) !important;
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }}

        /* ===== Sources panel ===== */
        .sources-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.7rem 0.5rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 0.75rem;
        }}
        .sources-header__title {{
            font-weight: 800;
            font-size: 0.95rem;
        }}
        .sources-header__subtitle {{
            font-size: 0.75rem;
            color: var(--muted-fg);
        }}

        /* ===== Sources field grid (inside each expander) ===== */
        .source-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem 1rem;
            font-size: 0.8rem;
        }}
        .source-field {{ display: flex; flex-direction: column; min-width: 0; }}
        .source-field--full {{ grid-column: span 2; }}
        .source-field__label {{ color: var(--muted-fg); font-size: 0.75rem; }}
        .source-field__value {{ color: var(--fg); overflow-wrap: anywhere; }}

        .source-card {{
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.6rem;
            transition: border-color 160ms ease;
        }}
        .source-card:hover {{ border-color: var(--primary); }}
        .source-card__title {{
            font-weight: 700;
            font-size: 0.86rem;
            color: var(--fg);
            margin: 0 0 0.25rem 0;
        }}
        .source-card__meta {{
            font-size: 0.7rem;
            color: var(--muted-fg);
            margin: 0 0 0.4rem 0;
        }}
        .source-card__snippet {{
            font-size: 0.78rem;
            line-height: 1.55;
            color: var(--muted-fg);
            margin: 0;
            overflow-wrap: anywhere;
        }}
        .source-rank-bar {{
            margin-top: 0.45rem;
            height: 6px;
            border-radius: 4px;
            background: var(--muted);
            overflow: hidden;
        }}
        .source-rank-bar__fill {{
            height: 100%;
            background: var(--accent);
            transition: width 320ms ease;
        }}
        .source-rank-label {{
            font-size: 0.7rem;
            color: var(--muted-fg);
            margin-top: 0.2rem;
        }}

        /* ===== Loading card (replica of loading-steps.tsx) ===== */
        @keyframes lcBounce {{
            0%, 100% {{ transform: translateY(-25%); animation-timing-function: cubic-bezier(0.8,0,1,1); }}
            50% {{ transform: none; animation-timing-function: cubic-bezier(0,0,0.2,1); }}
        }}
        @keyframes lcPulse {{
            0%, 100% {{ opacity: 0.6; }}
            50% {{ opacity: 0.2; }}
        }}
        .lc-wrap {{ display: flex; justify-content: flex-start; margin-bottom: 1.25rem; }}
        .lc-card {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: var(--radius-lg);
            box-shadow: 0 10px 15px -3px rgba(226, 232, 240, 0.5);
            overflow: hidden;
            width: 100%;
            max-width: 24rem;
            animation: fadeInUp 0.3s ease both;
        }}
        .lc-body {{ display: flex; align-items: center; gap: 1rem; padding: 1.25rem; }}
        .lc-iconwrap {{ position: relative; flex-shrink: 0; }}
        .lc-icon {{
            width: 3rem;
            height: 3rem;
            border-radius: var(--radius-md);
            background: linear-gradient(135deg, #3b82f6, #4f46e5);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 15px -3px #bfdbfe;
        }}
        .lc-ring {{
            position: absolute;
            inset: -4px;
            border-radius: 14px;
            border: 2px solid #93c5fd;
            animation: lcPulse 2s cubic-bezier(0.4,0,0.6,1) infinite;
        }}
        .lc-text {{ flex: 1; min-width: 0; }}
        .lc-label {{ font-size: 0.875rem; font-weight: 600; color: #1e293b; margin: 0; }}
        .lc-desc {{ font-size: 0.75rem; color: #64748b; margin: 0.125rem 0 0 0; }}
        .lc-dots {{ display: flex; gap: 0.25rem; align-items: center; }}
        .lc-dot {{
            width: 6px;
            height: 6px;
            background: #3b82f6;
            border-radius: 9999px;
            animation: lcBounce 1s infinite;
        }}
        .lc-footer {{ padding: 0 1.25rem 1rem 1.25rem; }}
        .lc-stepdots {{ display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 0.75rem; }}
        .lc-stepdot {{ width: 8px; height: 8px; border-radius: 9999px; transition: all 0.3s ease; }}
        .lc-stepdot--done {{ background: #10b981; }}
        .lc-stepdot--current {{ background: #3b82f6; width: 24px; }}
        .lc-stepdot--pending {{ background: #e2e8f0; }}
        .lc-track {{ height: 4px; background: #f1f5f9; border-radius: 9999px; overflow: hidden; }}
        .lc-fill {{
            height: 100%;
            border-radius: 9999px;
            background: linear-gradient(to right, #3b82f6, #6366f1);
            transition: width 0.5s ease-out;
        }}
        .lc-count {{ font-size: 11px; color: #94a3b8; text-align: center; margin: 0.5rem 0 0 0; }}

        /* ===== Dashboard widgets ===== */
        .widget-card {{
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            padding: 0.75rem 0.9rem;
            transition: border-color 160ms ease, transform 160ms ease;
        }}
        .widget-card:hover {{ border-color: var(--primary); transform: translateY(-1px); }}
        .widget-card__icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border-radius: var(--radius-sm);
            background: var(--citation-bg);
            color: var(--primary);
            margin-bottom: 0.45rem;
            font-size: 1rem;
        }}
        .widget-card__value {{
            font-size: 1.45rem;
            font-weight: 800;
            color: var(--fg);
            letter-spacing: -0.02em;
        }}
        .widget-card__label {{
            font-size: 0.74rem;
            color: var(--muted-fg);
            font-weight: 600;
        }}

        /* ===== Buttons ===== */
        .stButton > button {{
            border-radius: var(--radius-md);
            font-weight: 600;
            font-size: 0.85rem;
        }}
        button[kind="primary"], button[data-testid="baseButton-primary"] {{
            background: var(--primary) !important;
            color: var(--primary-fg) !important;
            border: 1px solid var(--primary) !important;
        }}
        button[kind="secondary"], button[data-testid="baseButton-secondary"] {{
            background: var(--glass-bg) !important;
            color: var(--fg) !important;
            border: 1px solid var(--glass-border) !important;
        }}

        /* "View Sources (n)" — React outline button under the assistant bubble */
        [class*="st-key-chat_sources_toggle"] button {{
            width: auto !important;
            background: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            color: #475569 !important;
            font-size: 0.75rem !important;
            font-weight: 500 !important;
            border-radius: var(--radius-sm) !important;
            padding: 0.4rem 0.8rem !important;
            min-height: 0 !important;
        }}
        [class*="st-key-chat_sources_toggle"] button:hover {{
            background: #f1f5f9 !important;
            border-color: #cbd5e1 !important;
            color: #0f172a !important;
        }}

        /* ===== Text inputs ===== */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stChatInput"] textarea {{
            background: var(--glass-bg) !important;
            color: var(--fg) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: var(--radius-md) !important;
        }}

        /* ===== Map legend (lighter pills, wraps cleanly) ===== */
        .map-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.3rem 0 0.7rem 0;
        }}
        .map-legend__item {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.22rem 0.55rem;
            border-radius: var(--radius-pill);
            background: transparent;
            border: 1px solid var(--border);
            font-size: 0.72rem;
            color: var(--muted-fg);
        }}
        .map-legend__dot {{
            width: 9px;
            height: 9px;
            border-radius: var(--radius-pill);
            flex-shrink: 0;
        }}

        /* ===== Tuck the per-message copy icon right under its bubble ===== */
        [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {{
            gap: 0.3rem;
        }}
        [data-testid="stMainBlockContainer"] [data-testid="stCustomComponentV1"] {{
            margin-top: -0.55rem !important;
            margin-bottom: 0 !important;
        }}

        /* ===== Inline search bar (lives at the bottom of the left chat column) =====
           A top border + padding mimics React's `border-t … bg-white/80` input
           area. The input itself is white with a slate focus ring; the send
           button is the dark slate gradient. */
        .st-key-chat_input {{
            border-top: 1px solid var(--border);
            padding-top: 0.75rem;
            margin-top: 0.25rem;
        }}
        [data-testid="stChatInput"] {{
            max-width: 100%;
            margin: 0;
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--border) !important;
            background: #ffffff !important;
            box-shadow: var(--shadow-sm) !important;
            padding: 0.1rem 0.3rem;
            transition: border-color 140ms ease, box-shadow 140ms ease;
        }}
        [data-testid="stChatInput"]:focus-within {{
            border-color: #94a3b8 !important;
        }}
        [data-testid="stChatInput"] textarea {{
            background: transparent !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            outline: none !important;
        }}
        [data-testid="stChatInput"] textarea::placeholder {{ color: var(--muted-fg) !important; }}
        [data-testid="stChatInput"] button {{
            background: linear-gradient(135deg, #1e293b, #0f172a) !important;
            color: #ffffff !important;
            border-radius: var(--radius-md) !important;
            min-width: 44px;
            min-height: 44px;
            box-shadow: 0 4px 6px -1px rgba(148, 163, 184, 0.5);
        }}
        [data-testid="stChatInput"] button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}

        /* ===== Map (folium) =====
           Height is owned by resizable_split.render() (JS), which sizes the
           iframe + its ancestor chain to fill the column (or 50% when the
           sources panel is open) and re-tiles Leaflet. We only ensure it can't
           collapse before the script runs. */
        .st-key-gnem_map iframe {{
            width: 100% !important;
            min-height: 360px;
        }}

        /* ===== Responsive: phones ===== */
        @media (max-width: 640px) {{
            [data-testid="stMainBlockContainer"] {{ padding: 0.5rem 0.75rem !important; }}
            .gnem-header {{
                flex-wrap: wrap;
                gap: var(--space-2);
                padding: var(--space-2) var(--space-3);
            }}
            .gnem-header__title {{ font-size: 0.82rem; }}
            .empty-title {{ font-size: 1.3rem; }}
            .empty-subtitle {{ font-size: 0.9rem; }}
            [data-testid="stChatInput"] {{ max-width: 100%; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

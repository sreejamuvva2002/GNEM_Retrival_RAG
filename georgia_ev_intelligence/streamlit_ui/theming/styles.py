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


def inject_styles(is_dark: bool, compact: bool = False) -> None:
    p = palette(is_dark)
    t = TOKENS
    body_padding = "0.2rem 0.6rem" if compact else "0.4rem 1rem"
    chat_font_size = "0.88rem" if compact else "0.95rem"

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
            background: var(--bg) !important;
            color: var(--fg) !important;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}

        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        [data-testid="stMainBlockContainer"] {{
            padding: {body_padding} !important;
            max-width: 100% !important;
        }}

        /* ===== Accessibility: visible keyboard focus everywhere ===== */
        a:focus-visible,
        button:focus-visible,
        input:focus-visible,
        textarea:focus-visible,
        select:focus-visible,
        [role="button"]:focus-visible,
        [data-testid="stSegmentedControl"] button:focus-visible,
        [data-testid="stChatInput"] textarea:focus-visible {{
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
            background: var(--sidebar) !important;
            border-right: 1px solid var(--sidebar-border);
        }}

        /* Color sidebar *text* with --fg, but do NOT blanket-override every
           descendant — that would force button labels to --fg and wreck the
           contrast on the active (primary) history button. */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] strong,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
            color: var(--fg) !important;
        }}
        /* Primary (active) sidebar button keeps readable on-primary text. */
        [data-testid="stSidebar"] button[kind="primary"],
        [data-testid="stSidebar"] button[kind="primary"] * {{
            color: var(--primary-fg) !important;
        }}

        /* Apply Inter only to text containers. Including span/div forces
           Streamlit's Material-Icons spans (e.g. expander chevrons) to use
           Inter, which renders the literal ligature text "keyboard_arrow_right"
           instead of the chevron glyph. */
        h1, h2, h3, h4, h5, h6, p, label, button, input, textarea {{
            font-family: 'Inter', 'Geist', ui-sans-serif, system-ui, -apple-system, sans-serif !important;
        }}

        /* ===== Header bar ===== */
        .gnem-header {{
            position: sticky;
            top: 0;
            z-index: var(--z-header);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: var(--space-3);
            padding: var(--space-3) var(--space-4);
            background: var(--glass-bg);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-sm);
            margin-bottom: var(--space-3);
        }}
        .gnem-header__brand {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}
        .gnem-logo {{
            min-width: 64px;
            height: 30px;
            padding: 0 0.6rem;
            border-radius: var(--radius-sm);
            background: var(--primary);
            color: var(--primary-fg);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.8rem;
            letter-spacing: 0.06em;
            box-shadow: var(--shadow-sm);
        }}
        .gnem-header__title {{
            font-weight: 700;
            font-size: 0.92rem;
            color: var(--fg);
            letter-spacing: -0.01em;
        }}

        /* ===== Mode tabs (st.segmented_control) ===== */
        /* primaryColor in config.toml already paints the active pill brand-blue;
           here we add hover/spacing polish and a non-color active cue (weight). */
        [data-testid="stSegmentedControl"] {{
            display: flex;
            justify-content: center;
        }}
        [data-testid="stSegmentedControl"] button {{
            border-radius: var(--radius-md) !important;
            font-weight: 600 !important;
            transition: background 140ms ease, color 140ms ease, border-color 140ms ease;
        }}
        [data-testid="stSegmentedControl"] button:hover {{
            border-color: var(--primary) !important;
        }}
        [data-testid="stSegmentedControl"] button[aria-checked="true"],
        [data-testid="stSegmentedControl"] button[kind="primary"] {{
            font-weight: 700 !important;
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
        .chat-row {{
            display: flex;
            gap: 0.7rem;
            margin-bottom: 0.9rem;
        }}
        .chat-row--user {{
            flex-direction: row-reverse;
        }}
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
            padding: 0.7rem 0.95rem;
            border-radius: var(--radius-lg);
            max-width: 78%;
            font-size: {chat_font_size};
            line-height: 1.65;
            color: var(--fg);
            overflow-wrap: anywhere;
        }}
        .chat-bubble--user {{
            background: var(--user-bubble);
            color: var(--user-bubble-text);
            border-top-right-radius: 4px;
        }}
        .chat-bubble--assistant {{
            background: var(--assistant-bubble);
            border: 1px solid var(--glass-border);
            border-top-left-radius: 4px;
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
            padding: 1.4rem 1rem 0;
            max-width: 760px;
            margin: 0 auto;
        }}
        .empty-logo {{
            width: 64px;
            height: 64px;
            border-radius: var(--radius-xl);
            background: var(--citation-bg);
            color: var(--primary);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 1.7rem;
        }}
        .empty-logo [data-testid="stIconMaterial"] {{ font-size: 2rem; }}
        .empty-title {{
            margin-top: 1.1rem;
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--fg);
            letter-spacing: -0.02em;
        }}
        .empty-subtitle {{
            margin-top: 0.5rem;
            font-size: 0.98rem;
            line-height: 1.55;
            color: var(--muted-fg);
        }}

        /* ===== Suggested-question cards (rendered as st.button, key=suggest_*) ===== */
        [class*="st-key-suggest_"] {{
            margin-top: 0.4rem;
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

        /* ===== Searching indicator ===== */
        .search-step {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.6rem 0.8rem;
            border-radius: var(--radius-md);
            border: 1px solid var(--glass-border);
            background: var(--glass-bg);
            margin-bottom: 0.5rem;
        }}
        .search-step__spinner {{
            width: 18px;
            height: 18px;
            border-radius: var(--radius-pill);
            border: 2px solid var(--muted);
            border-top-color: var(--primary);
            animation: spin 0.8s linear infinite;
        }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        .search-step__label {{ font-size: 0.85rem; font-weight: 600; color: var(--fg); }}
        .search-progress {{
            display: flex;
            gap: 0.25rem;
            margin-top: 0.45rem;
        }}
        .search-progress__cell {{
            flex: 1;
            height: 4px;
            border-radius: 2px;
            background: var(--muted);
        }}
        .search-progress__cell--active {{ background: var(--primary); }}

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

        /* ===== Sidebar header + new-chat + history ===== */
        .sidebar-heading {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.6rem;
        }}
        .sidebar-heading__badge {{
            width: 32px;
            height: 32px;
            border-radius: var(--radius-sm);
            background: var(--primary);
            color: var(--primary-fg);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
        }}
        .sidebar-heading__label {{ font-size: 0.95rem; font-weight: 800; }}
        .sidebar-empty {{
            color: var(--muted-fg);
            font-size: 0.8rem;
            text-align: center;
            margin-top: 2.4rem;
        }}
        .sidebar-footer {{
            color: var(--muted-fg);
            font-size: 0.7rem;
            text-align: center;
            margin-top: 1.4rem;
            letter-spacing: 0.02em;
        }}
        /* New Chat button — intentional primary-tinted treatment */
        .st-key-sb_new_chat button {{
            border-radius: var(--radius-md) !important;
            font-weight: 700 !important;
            border: 1px solid var(--primary) !important;
            color: var(--primary) !important;
            background: var(--citation-bg) !important;
            transition: background 140ms ease, transform 140ms ease;
        }}
        .st-key-sb_new_chat button:hover {{
            background: var(--primary) !important;
            color: var(--primary-fg) !important;
        }}
        /* History item buttons — consistent radius + active cue */
        [class*="st-key-sb_open_"] button {{
            border-radius: var(--radius-md) !important;
            justify-content: flex-start !important;
            text-align: left !important;
            font-weight: 600 !important;
        }}
        [class*="st-key-sb_del_"] button {{
            border-radius: var(--radius-md) !important;
            color: var(--muted-fg) !important;
        }}
        [class*="st-key-sb_del_"] button:hover {{
            color: var(--destructive) !important;
            border-color: var(--destructive) !important;
        }}

        .history-item {{
            padding: 0.5rem 0.6rem;
            border-radius: var(--radius-md);
            border: 1px solid transparent;
            transition: background 140ms ease, border-color 140ms ease;
            margin-bottom: 0.3rem;
        }}
        .history-item:hover {{ background: var(--sidebar-accent); border-color: var(--sidebar-border); }}
        .history-item--active {{
            background: var(--citation-bg);
            border-color: var(--primary);
        }}
        .history-item__title {{ font-size: 0.82rem; font-weight: 700; color: var(--fg); margin: 0; }}
        .history-item__preview {{ font-size: 0.7rem; color: var(--muted-fg); margin: 0.1rem 0 0 0; }}
        .history-item__meta {{ font-size: 0.66rem; color: var(--muted-fg); margin-top: 0.1rem; }}

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

        /* ===== Header: compact settings icon button (#2) ===== */
        .st-key-hdr_settings button {{
            width: 40px !important;
            height: 40px !important;
            min-height: 40px !important;
            padding: 0 !important;
            border-radius: var(--radius-md) !important;
            background: var(--glass-bg) !important;
            border: 1px solid var(--glass-border) !important;
            color: var(--fg) !important;
            font-size: 1.05rem !important;
        }}
        .st-key-hdr_settings button:hover {{ border-color: var(--primary) !important; }}

        /* ===== Header: sun/moon sliding theme toggle (#3) ===== */
        .st-key-hdr_theme_toggle {{
            display: flex !important;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
        }}
        .st-key-hdr_theme_toggle::before {{ content: "☀"; font-size: 0.95rem; line-height: 1; }}
        .st-key-hdr_theme_toggle::after {{ content: "🌙"; font-size: 0.9rem; line-height: 1; }}
        .st-key-hdr_theme_toggle [data-baseweb="checkbox"] > div {{
            transition: all 0.25s ease !important;
        }}

        /* ===== Tuck the per-message copy icon right under its bubble (#1) ===== */
        [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {{
            gap: 0.3rem;
        }}
        [data-testid="stMainBlockContainer"] [data-testid="stCustomComponentV1"] {{
            margin-top: -0.55rem !important;
            margin-bottom: 0 !important;
        }}

        /* ===== Theme the docked bottom chrome so light mode has no dark band (#5) ===== */
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"] {{
            background: var(--bg) !important;
            padding-bottom: env(safe-area-inset-bottom, 0px);
        }}

        /* ===== ChatGPT-style bottom input pill (#7) ===== */
        [data-testid="stChatInput"] {{
            max-width: 820px;
            margin: 0 auto;
            border-radius: var(--radius-pill) !important;
            border: 1px solid var(--glass-border) !important;
            background: var(--card) !important;
            box-shadow: var(--shadow-md);
            padding: 0.15rem 0.4rem 0.15rem 0.4rem;
            transition: border-color 140ms ease, box-shadow 140ms ease;
        }}
        [data-testid="stChatInput"]:focus-within {{
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px var(--citation-bg);
        }}
        [data-testid="stChatInput"] textarea {{
            background: transparent !important;
            border: none !important;
            border-radius: var(--radius-pill) !important;
        }}
        [data-testid="stChatInput"] textarea::placeholder {{ color: var(--muted-fg) !important; }}
        [data-testid="stChatInput"] button {{
            background: var(--primary) !important;
            color: var(--primary-fg) !important;
            border-radius: var(--radius-pill) !important;
            min-width: 40px;
            min-height: 40px;
        }}
        [data-testid="stChatInput"] button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}

        /* ===== Responsive: phones ===== */
        @media (max-width: 640px) {{
            [data-testid="stMainBlockContainer"] {{ padding: 0.3rem 0.6rem !important; }}
            .gnem-header {{
                flex-wrap: wrap;
                gap: var(--space-2);
                padding: var(--space-2) var(--space-3);
            }}
            .gnem-header__title {{ font-size: 0.82rem; }}
            .empty-title {{ font-size: 1.3rem; }}
            .empty-subtitle {{ font-size: 0.9rem; }}
            /* Suggestion cards collapse to one column (Streamlit columns stack). */
            [class*="st-key-suggest_"] button {{ min-height: 52px !important; }}
            [data-testid="stChatInput"] {{ max-width: 100%; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

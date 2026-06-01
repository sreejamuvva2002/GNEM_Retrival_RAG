"""CSS injection — emit one <style> block matching the React rag-chat-ui look.

The CSS is generated against the active palette so theme switches re-render
with the right colors. We hide Streamlit chrome (menu, footer, deploy button)
and rebuild the top header / sidebar / message bubbles to match the React UI.
"""
from __future__ import annotations

import streamlit as st

from .colors import palette


def inject_styles(is_dark: bool, compact: bool = False) -> None:
    p = palette(is_dark)
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
        }}

        html, body, .stApp {{
            background: var(--bg) !important;
            color: var(--fg) !important;
        }}

        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        [data-testid="stMainBlockContainer"] {{
            padding: {body_padding} !important;
            max-width: 100% !important;
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
            z-index: 50;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.55rem 1rem;
            background: var(--glass-bg);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid var(--glass-border);
            border-radius: 14px;
            margin-bottom: 0.7rem;
        }}
        .gnem-header__brand {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}
        .gnem-logo {{
            width: 76px;
            height: 30px;
            border-radius: 9px;
            background: var(--primary);
            color: var(--primary-fg);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.8rem;
            letter-spacing: 0.04em;
        }}
        .gnem-header__title {{
            font-weight: 700;
            font-size: 0.92rem;
            color: var(--fg);
        }}

        /* ===== Glass cards ===== */
        .glass-card {{
            background: var(--glass-bg);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid var(--glass-border);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
        }}
        .glass-card:hover {{
            border-color: var(--primary);
            box-shadow: 0 18px 36px rgba(15, 23, 42, 0.18);
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
            border-radius: 999px;
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
            border-radius: 16px;
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
            border-radius: 6px;
            background: var(--citation-bg);
            color: var(--citation-text);
            font-size: 0.74rem;
            font-weight: 700;
            text-decoration: none;
        }}

        /* ===== Empty state ===== */
        .empty-shell {{
            text-align: center;
            padding: 2.4rem 1rem 0;
        }}
        .empty-logo {{
            width: 78px;
            height: 78px;
            border-radius: 22px;
            background: var(--citation-bg);
            color: var(--primary);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
        }}
        .empty-title {{
            margin-top: 1.3rem;
            font-size: 1.7rem;
            font-weight: 800;
            color: var(--fg);
            letter-spacing: -0.02em;
        }}
        .empty-subtitle {{
            margin-top: 0.4rem;
            font-size: 1rem;
            color: var(--muted-fg);
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
            border-radius: 14px;
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
            border-radius: 12px;
            border: 1px solid var(--glass-border);
            background: var(--glass-bg);
            margin-bottom: 0.5rem;
        }}
        .search-step__spinner {{
            width: 18px;
            height: 18px;
            border-radius: 999px;
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
            border-radius: 14px;
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
            border-radius: 10px;
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

        /* ===== Sidebar chat history item ===== */
        .history-item {{
            padding: 0.5rem 0.6rem;
            border-radius: 10px;
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
            border-radius: 10px;
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
            border-radius: 12px !important;
        }}

        /* ===== Suggested-question cards ===== */
        .suggest-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.7rem;
            margin: 1.4rem auto 0;
            max-width: 720px;
        }}

        /* ===== Selectbox / view toggle ===== */
        .gnem-view-toggle {{
            display: inline-flex;
            border-radius: 12px;
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            padding: 3px;
            gap: 2px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

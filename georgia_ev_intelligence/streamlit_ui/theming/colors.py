"""Color palette derived from rag-chat-ui/app/globals.css (oklch → hex).

Two palettes: dark (matches React default) and light. Streamlit lacks a
runtime theme switch, so we emit both as CSS variables and toggle a body
class to switch between them at render time.
"""
from __future__ import annotations


DARK_PALETTE = {
    "background": "#0f1419",
    "foreground": "#e7eef5",
    "card": "#1a2230",
    "card_foreground": "#e7eef5",
    "primary": "#5e7fff",
    "primary_foreground": "#0f1419",
    "secondary": "#1d2735",
    "secondary_foreground": "#e7eef5",
    "muted": "#1d2735",
    "muted_foreground": "#8b9bb3",
    "accent": "#3ecfaf",
    "accent_foreground": "#0f1419",
    "destructive": "#d8485c",
    "destructive_foreground": "#e7eef5",
    "border": "#2a3548",
    "input": "#1d2735",
    "ring": "#5e7fff",
    "sidebar": "#121823",
    "sidebar_foreground": "#e7eef5",
    "sidebar_border": "#2a3548",
    "sidebar_accent": "#1d2735",
    "success": "#3ecfaf",
    "warning": "#f5c054",
    "glass_bg": "rgba(26, 34, 48, 0.7)",
    "glass_border": "rgba(94, 127, 255, 0.18)",
    "user_bubble": "#5e7fff",
    "user_bubble_text": "#0f1419",
    "assistant_bubble": "rgba(29, 39, 53, 0.78)",
    "citation_bg": "rgba(94, 127, 255, 0.18)",
    "citation_text": "#9ab1ff",
}

LIGHT_PALETTE = {
    "background": "#f4f7f9",
    "foreground": "#11263a",
    "card": "#ffffff",
    "card_foreground": "#11263a",
    "primary": "#3863c4",
    "primary_foreground": "#ffffff",
    "secondary": "#eaf0f6",
    "secondary_foreground": "#11263a",
    "muted": "#eaf0f6",
    # Darkened from #5d7187 to clear WCAG AA (4.5:1) on the #f4f7f9 background.
    "muted_foreground": "#4d6175",
    "accent": "#12897f",
    "accent_foreground": "#ffffff",
    "destructive": "#b94b5c",
    "destructive_foreground": "#ffffff",
    "border": "#d7e1ea",
    "input": "#ffffff",
    "ring": "#3863c4",
    "sidebar": "#ffffff",
    "sidebar_foreground": "#11263a",
    "sidebar_border": "#d7e1ea",
    "sidebar_accent": "#eaf0f6",
    "success": "#12897f",
    "warning": "#d8902f",
    "glass_bg": "rgba(255, 255, 255, 0.78)",
    "glass_border": "rgba(56, 99, 196, 0.18)",
    "user_bubble": "#3863c4",
    "user_bubble_text": "#ffffff",
    "assistant_bubble": "rgba(255, 255, 255, 0.85)",
    "citation_bg": "rgba(56, 99, 196, 0.12)",
    "citation_text": "#3863c4",
}


def palette(is_dark: bool) -> dict:
    return DARK_PALETTE if is_dark else LIGHT_PALETTE


# Theme-agnostic design tokens (radius / spacing / shadow / z-index). Emitted as
# CSS variables by theming/styles.py so component CSS stops hardcoding magic
# numbers and stays internally consistent.
TOKENS = {
    # Radius scale
    "radius_sm": "8px",
    "radius_md": "12px",
    "radius_lg": "16px",
    "radius_xl": "20px",
    "radius_pill": "9999px",
    # Spacing scale (rem)
    "space_1": "0.25rem",
    "space_2": "0.5rem",
    "space_3": "0.75rem",
    "space_4": "1rem",
    "space_5": "1.5rem",
    "space_6": "2rem",
    # Soft, consistent shadows
    "shadow_sm": "0 1px 2px rgba(15, 23, 42, 0.06)",
    "shadow_md": "0 8px 24px rgba(15, 23, 42, 0.10)",
    "shadow_lg": "0 18px 36px rgba(15, 23, 42, 0.16)",
    # Z-index layers
    "z_header": "50",
    "z_overlay": "100",
}


# Source-type colors are stable across themes (match the React palette).
SOURCE_TYPE_COLORS = {
    "company": ("#5e7fff", "rgba(94, 127, 255, 0.12)"),
    "government": ("#f5c054", "rgba(245, 192, 84, 0.12)"),
    "supply_chain": ("#3ecfaf", "rgba(62, 207, 175, 0.12)"),
    "infrastructure": ("#b07cf0", "rgba(176, 124, 240, 0.12)"),
    "manufacturing": ("#f08a5d", "rgba(240, 138, 93, 0.12)"),
    "web": ("#8b9bb3", "rgba(139, 155, 179, 0.16)"),
    "unknown": ("#8b9bb3", "rgba(139, 155, 179, 0.16)"),
}

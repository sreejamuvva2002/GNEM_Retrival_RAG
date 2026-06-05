"""Screenshot-capture harness for exporting UI screen states to Figma.

These modules are NOT part of the production app. They render the real UI
components with seeded mock data (no RAG backend / DB required) so each screen
state can be screenshotted with Playwright and uploaded into Figma as image
frames. See ``capture_app.py`` (the Streamlit harness) and ``shoot.py`` (the
Playwright driver).
"""

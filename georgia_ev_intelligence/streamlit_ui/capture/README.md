# UI → Figma capture harness

Reproduces every screen state of the Streamlit chat+map UI as **pixel-exact image
frames** in Figma. Not part of the production app — used only to export designs.

## Pipeline

1. **`capture_app.py`** — a Streamlit harness that mirrors `app.py`'s layout but
   renders one state per page load from seeded mock data (no RAG backend / DB).
   Select the state with the `?state=` query param:
   `empty | chat | loading | sources | settings | sidebar`.

   ```bash
   .venv/bin/python -m streamlit run \
     georgia_ev_intelligence/streamlit_ui/capture/capture_app.py \
     --server.port 8531 --server.headless true
   ```

2. **`shoot.py`** — drives the harness with Playwright across all states +
   desktop/mobile viewports and writes PNGs to `outputs/figma_shots/`.

   ```bash
   .venv/bin/python georgia_ev_intelligence/streamlit_ui/capture/shoot.py \
     --base-url http://localhost:8531 --out outputs/figma_shots
   ```

3. **Upload to Figma** — frames live in file `czdmNReNDTAlRPysQ1hlAT`
   ("Georgia EV Intelligence — UI Screens"). Each screenshot is placed as the
   image fill of a pre-built, labeled frame via the `upload_assets` MCP tool +
   an HTTP POST of the PNG bytes to the returned single-use `submitUrl`.

## Status (2026-06-05)

Filled: `01-empty` (1:5), `02-chat` (1:7), `03-loading` (1:9), `04-sources`
(1:11), `05-settings` (1:13).

**Pending** (blocked by the Figma **Starter plan monthly MCP tool-call cap** —
all tool calls count, not just reads; `upload_assets` is NOT exempt):

| Screenshot              | Target frame |
|-------------------------|--------------|
| `06-sidebar.png`        | `1:15`       |
| `07-empty-mobile.png`   | `1:17`       |
| `08-chat-mobile.png`    | `1:19`       |

### Finishing the last 3 frames (after the quota resets or an upgrade)

For each pending row, in the Claude/MCP session:

1. `upload_assets({ fileKey: "czdmNReNDTAlRPysQ1hlAT", count: 1, nodeId: "<id>", scaleMode: "FILL" })`
   → returns a `submitUrl`.
2. POST the PNG bytes to it (no MCP quota cost — plain HTTP):
   ```bash
   curl -s -X POST "<submitUrl>" -F "file=@outputs/figma_shots/06-sidebar.png;type=image/png"
   ```
   A `{ "success": true, "placedOnNodeId": "1:15" }` response means it landed.

The frames already exist and are positioned/labeled; only their image fill is missing.

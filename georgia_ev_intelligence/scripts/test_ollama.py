"""
Ollama Health Check — tests both LLM and embedding models.
Run: venv\\Scripts\\python scripts\\test_ollama.py
"""
import sys
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from shared.config import Config

SEP = "=" * 65

def check_ollama_running(base_url: str) -> bool:
    """Check if Ollama server is up."""
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False

def list_models(base_url: str) -> list[str]:
    """List available Ollama models."""
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        return []

def test_embedding(base_url: str, model: str) -> dict:
    """Test nomic-embed-text — should return 768-dim vector."""
    print(f"\n  Testing embed: {model}")
    start = time.monotonic()
    try:
        r = httpx.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": ["Georgia EV supply chain battery manufacturer Tier 1"]},
            timeout=60.0,
        )
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            return {"ok": False, "error": "No embeddings returned", "time_s": 0}
        vec = embeddings[0]
        elapsed = time.monotonic() - start
        return {
            "ok": len(vec) == 768,
            "dims": len(vec),
            "time_s": round(elapsed, 2),
            "error": None if len(vec) == 768 else f"Wrong dims: {len(vec)} (expected 768)",
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "time_s": round(time.monotonic() - start, 2)}

def test_llm(base_url: str, model: str) -> dict:
    """Test LLM model — should return valid JSON."""
    print(f"\n  Testing LLM: {model}")
    start = time.monotonic()
    try:
        payload = {
            "model": model,
            "prompt": (
                'Extract facts from this text. Return JSON array.\n'
                'Text: "Hanwha Q Cells invested $2.5 billion in Hall County, Georgia in 2023, creating 2,500 jobs."\n'
                'Return: [{"fact_type": "investment", "fact_value_text": "...", "fact_value_numeric": ..., "location_county": "..."}]'
            ),
            "stream": False,
            "think": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_predict": 300,
                "num_ctx": 2048,
            },
        }
        r = httpx.post(f"{base_url}/api/generate", json=payload, timeout=120.0)
        r.raise_for_status()
        data = r.json()
        response_text = data.get("response", "")
        elapsed = time.monotonic() - start

        # Try to parse as JSON
        parsed = None
        parse_ok = False
        try:
            parsed = json.loads(response_text)
            parse_ok = True
        except Exception:
            pass

        return {
            "ok": parse_ok,
            "time_s": round(elapsed, 2),
            "response_preview": response_text[:200],
            "json_valid": parse_ok,
            "parsed": parsed,
            "error": None if parse_ok else "Could not parse as JSON",
        }
    except httpx.TimeoutException:
        elapsed = time.monotonic() - start
        return {"ok": False, "error": f"TIMEOUT after {elapsed:.0f}s", "time_s": round(elapsed, 2)}
    except Exception as e:
        return {"ok": False, "error": str(e), "time_s": round(time.monotonic() - start, 2)}

def main():
    cfg = Config.get()
    base_url = cfg.ollama_base_url
    llm_model = cfg.ollama_llm_model
    embed_model = cfg.ollama_embed_model

    print(f"\n{'#'*65}")
    print("  OLLAMA HEALTH CHECK")
    print(f"{'#'*65}")
    print(f"\n  Ollama URL   : {base_url}")
    print(f"  LLM model    : {llm_model}")
    print(f"  Embed model  : {embed_model}")

    # 1. Is Ollama running?
    print(f"\n{SEP}")
    print("  1. Is Ollama server running?")
    print(SEP)
    running = check_ollama_running(base_url)
    if not running:
        print(f"\n  ❌ Ollama is NOT running at {base_url}")
        print(f"  → Start it with:  ollama serve")
        print(f"  → Or it may already be running as a background service")
        sys.exit(1)
    print(f"  ✅ Ollama is running")

    # 2. Available models
    print(f"\n{SEP}")
    print("  2. Available models")
    print(SEP)
    models = list_models(base_url)
    if models:
        for m in models:
            marker = "✅" if m.split(":")[0] in [llm_model.split(":")[0], embed_model.split(":")[0]] else "  "
            print(f"  {marker} {m}")
    else:
        print("  ⚠️  No models found — run: ollama pull nomic-embed-text")

    # Check required models exist
    model_names = [m.split(":")[0] for m in models]
    llm_base = llm_model.split(":")[0]
    embed_base = embed_model.split(":")[0]

    if llm_base not in model_names:
        print(f"\n  ⚠️  LLM model '{llm_model}' not found. Pull it: ollama pull {llm_model}")
    if embed_base not in model_names:
        print(f"\n  ⚠️  Embed model '{embed_model}' not found. Pull it: ollama pull {embed_model}")

    # 3. Test embedding
    print(f"\n{SEP}")
    print(f"  3. Embedding Test ({embed_model})")
    print(SEP)
    if embed_base in model_names:
        embed_result = test_embedding(base_url, embed_model)
        if embed_result["ok"]:
            print(f"  ✅ Embedding works: {embed_result['dims']} dims in {embed_result['time_s']}s")
        else:
            print(f"  ❌ Embedding FAILED: {embed_result['error']} ({embed_result['time_s']}s)")
    else:
        print(f"  ⏭️  Skipped (model not installed)")
        embed_result = {"ok": False, "error": "model not installed"}

    # 4. Test LLM
    print(f"\n{SEP}")
    print(f"  4. LLM Test ({llm_model})")
    print(SEP)
    if llm_base in model_names:
        llm_result = test_llm(base_url, llm_model)
        if llm_result["ok"]:
            print(f"  ✅ LLM works in {llm_result['time_s']}s")
            print(f"  ✅ JSON output valid: {llm_result['json_valid']}")
            if llm_result.get("parsed"):
                print(f"  ✅ Extracted facts: {json.dumps(llm_result['parsed'], indent=4)[:400]}")
        else:
            print(f"  ❌ LLM FAILED: {llm_result['error']} ({llm_result['time_s']}s)")
            if llm_result.get("response_preview"):
                print(f"  Raw output: {llm_result['response_preview'][:200]}")
    else:
        print(f"  ⏭️  Skipped (model not installed)")
        llm_result = {"ok": False, "error": "model not installed"}

    # Summary
    print(f"\n{'#'*65}")
    print("  SUMMARY")
    print(f"{'#'*65}")
    print(f"  Ollama server : ✅ Running")
    print(f"  Embed model   : {'✅ OK  (' + str(embed_result.get('dims', 0)) + ' dims, ' + str(embed_result.get('time_s', 0)) + 's)' if embed_result.get('ok') else '❌ ' + str(embed_result.get('error', ''))}")
    print(f"  LLM model     : {'✅ OK  (' + str(llm_result.get('time_s', 0)) + 's)' if llm_result.get('ok') else '❌ ' + str(llm_result.get('error', ''))}")

    all_ok = embed_result.get("ok") and llm_result.get("ok")
    print(f"\n  Overall: {'🎉 ALL GOOD — ready for Phase 2 + 3' if all_ok else '⚠️  Fix issues above before proceeding'}")
    print()

if __name__ == "__main__":
    main()

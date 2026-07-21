"""Check certification status (ISO 9001 / IATF 16949 / ... configurable) for each company
using web evidence judged by a LOCAL LLM, and write ONE PAGE PER COMPANY covering all
certifications found.

Per company+certification the model extracts:
    org_status          confirmed_active | likely_active | confirmed_inactive | not_found
    facility_status     confirmed_facility | different_facility | unknown_facility
    certification_body  e.g. DQS, TUV, SGS ("" if unknown)
    reference_no        certificate/registration number ("" if unknown)
    scope               certificate scope text ("" if unknown)
    expiry_date         ISO date if visible, else ""
and the script derives final_label (e.g. "confirmed_active_facility / active_expiry_unknown").

Outputs:
    outputs/certification_status.csv     flat summary (one row per company+cert, resumable)
    outputs/evidence_log.jsonl           raw search/page text per verdict (audit trail)
    outputs/companies/<company>.md       one page per company, all certs, in the agreed format

Browsing backends:
    --backend browser  browser-use/browser-harness driving your real Chrome via CDP
    --backend http     plain HTTPS (DuckDuckGo HTML endpoint + page fetches), headless-friendly
    --backend auto     try browser, fall back to http   [default]

The sheet's location/address are UNVERIFIED hints: never added to search queries unless
you pass --use-location, and flagged as unverified in the LLM prompt.

Usage:
    python scripts/check_certifications.py --limit 3 --backend http     # smoke test
    python scripts/check_certifications.py --backend browser            # full run
    python scripts/check_certifications.py --cert "ISO 14001" --cert "ISO 9001:2015"
"""
from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import re
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import llm_client  # noqa: E402

DEFAULT_COMPANIES = PKG_ROOT / "data" / "companies.csv"
DEFAULT_OUT = PKG_ROOT / "outputs" / "certification_status.csv"
DEFAULT_LOG = PKG_ROOT / "outputs" / "evidence_log.jsonl"
DEFAULT_PAGES_DIR = PKG_ROOT / "outputs" / "companies"
DEFAULT_CERTS = ["ISO 9001:2015", "IATF 16949:2016"]

OUT_FIELDS = [
    "company", "certification", "org_status", "facility_status", "final_label",
    "certification_body", "reference_no", "scope", "expiry_date",
    "confidence", "evidence", "source_url", "checked_at", "model", "backend",
]

ORG_STATUSES = {"confirmed_active", "likely_active", "confirmed_inactive", "not_found"}
FACILITY_STATUSES = {"confirmed_facility", "different_facility", "unknown_facility"}

SYSTEM_PROMPT = """You are a compliance analyst. You are given web search results (and text \
from certificate-registry pages) about ONE company, and must extract its status for ONE \
specific certification.

Respond with ONLY a JSON object, no prose, exactly this shape:
{"org_status": "confirmed_active" | "likely_active" | "confirmed_inactive" | "not_found",
 "facility_status": "confirmed_facility" | "different_facility" | "unknown_facility",
 "certification_body": "<registrar name, e.g. DQS, TUV, SGS, BSI, or empty string>",
 "reference_no": "<certificate or registration number, or empty string>",
 "scope": "<certificate scope text, or empty string>",
 "expiry_date": "<expiry/valid-until date if visible, or empty string>",
 "confidence": <float 0.0-1.0>,
 "evidence": "<one sentence citing the strongest evidence>",
 "source_url": "<the single URL best supporting the verdict, or empty string>"}

Rules:
- org_status "confirmed_active": a registrar, certificate directory, or the company's own site
  explicitly shows a current certificate for this certification.
- "likely_active": credible secondary mentions (press releases, supplier directories) without
  direct certificate evidence.
- "confirmed_inactive": explicit evidence the certificate is withdrawn/expired/never held.
- "not_found": no relevant evidence, or results are about a different company.
- facility_status "confirmed_facility": evidence ties the certificate to the specific facility/
  address given. "different_facility": the certificate is clearly for another site of the same
  organization. "unknown_facility": evidence does not say which site.
- The location/address given for the company is an UNVERIFIED hint and may be wrong — never
  downgrade org_status because a result shows a different city.
- Company names collide; ignore results clearly about unrelated companies.
- Copy reference numbers, scope, dates, and registrar names VERBATIM from the evidence. Never
  invent them; use empty strings when not visible.
"""


def derive_final_label(v: dict) -> str:
    org = v.get("org_status", "not_found")
    fac = v.get("facility_status", "unknown_facility")
    if org == "not_found":
        return "not_found"
    if org == "confirmed_inactive":
        return "confirmed_inactive"
    base = {
        ("confirmed_active", "confirmed_facility"): "confirmed_active_facility",
        ("confirmed_active", "different_facility"): "confirmed_active_other_facility",
        ("confirmed_active", "unknown_facility"): "confirmed_active_org",
        ("likely_active", "confirmed_facility"): "likely_active_facility",
        ("likely_active", "different_facility"): "likely_active_other_facility",
        ("likely_active", "unknown_facility"): "likely_active_org",
    }[(org, fac)]
    expiry = (v.get("expiry_date") or "").strip()
    suffix = f"active_until_{expiry}" if expiry else "active_expiry_unknown"
    return f"{base} / {suffix}"


# ---------------------------------------------------------------- browsing backends

def _load_env():
    env = PKG_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, val = line.split("=", 1)
                os.environ.setdefault(k.strip(), val.strip().strip('"').strip("'"))


def _strip_html(page: str, limit: int = 4000) -> str:
    page = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", page, flags=re.DOTALL | re.IGNORECASE)
    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", page))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


class HttpBackend:
    """DuckDuckGo HTML endpoint + plain page fetches. Headless-friendly."""
    name = "http"

    def search(self, query: str) -> tuple[str, list[str]]:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            page = resp.read().decode("utf-8", errors="replace")
        return self._parse(page)

    def fetch_page(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return _strip_html(resp.read().decode("utf-8", errors="replace"))

    @staticmethod
    def _parse(page: str, max_results: int = 8) -> tuple[str, list[str]]:
        def clean(s: str) -> str:
            return html_lib.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()

        results, links = [], []
        # Each organic result sits in its own <div class="result ..."> block.
        for block in re.split(r'<div[^>]+class="[^"]*\bresult\b', page)[1:]:
            a = re.search(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                          block, flags=re.DOTALL)
            if not a:
                continue
            href, title = a.groups()
            snip = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, flags=re.DOTALL)
            # DDG wraps URLs as /l/?uddg=<encoded>
            m = re.search(r"uddg=([^&]+)", href)
            real = urllib.parse.unquote(m.group(1)) if m else href
            links.append(real)
            results.append(f"URL: {real}\nTITLE: {clean(title)}\n"
                           f"SNIPPET: {clean(snip.group(1)) if snip else ''}")
            if len(results) >= max_results:
                break
        return ("\n\n".join(results) if results else "(no search results)"), links


class BrowserBackend:
    """browser-use/browser-harness: drives your real Chrome over CDP."""
    name = "browser"

    def __init__(self):
        from browser_harness.admin import ensure_daemon
        from browser_harness import helpers
        ensure_daemon()
        self.bh = helpers
        self._tab_open = False

    def _goto(self, url: str) -> None:
        if not self._tab_open:
            self.bh.new_tab(url)
            self._tab_open = True
        else:
            self.bh.goto_url(url)
        self.bh.wait_for_load(timeout=20.0)

    def search(self, query: str) -> tuple[str, list[str]]:
        self._goto("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query))
        text = str(self.bh.js("document.body.innerText"))[:6000]
        raw = self.bh.js(
            "JSON.stringify([...document.querySelectorAll('a.result__a')]"
            ".slice(0,8).map(a=>a.href))")
        try:
            links = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except Exception:
            links = []
        return text, links

    def fetch_page(self, url: str) -> str:
        self._goto(url)
        text = str(self.bh.js("document.body.innerText"))
        return re.sub(r"\s+", " ", text).strip()[:4000]


def make_backend(kind: str):
    if kind == "http":
        return HttpBackend()
    if kind == "browser":
        return BrowserBackend()
    try:
        b = BrowserBackend()
        print("[backend] using browser-harness (real Chrome via CDP)")
        return b
    except Exception as e:
        print(f"[backend] browser-harness unavailable ({e.__class__.__name__}: {e}); "
              f"falling back to plain HTTP search")
        return HttpBackend()


# ---------------------------------------------------------------- evidence gathering

# Prefer registrar/registry pages when deciding which result links to open.
REGISTRY_HINTS = ("dqs", "tuv", "sgs", "bsi", "bureauveritas", "intertek", "certcheck",
                  "iatfglobaloversight", "iaf", "certificate", "certification", "cert")


def gather_evidence(backend, query: str, fetch_pages: int) -> tuple[str, str]:
    """Return (evidence_text, search_text) for one query."""
    search_text, links = backend.search(query)
    parts = [f"SEARCH RESULTS for query {query!r}:\n{search_text}"]
    if fetch_pages and links:
        ranked = sorted(links, key=lambda u: 0 if any(h in u.lower() for h in REGISTRY_HINTS) else 1)
        for url in ranked[:fetch_pages]:
            try:
                page_text = backend.fetch_page(url)
                parts.append(f"PAGE TEXT from {url}:\n{page_text}")
            except Exception as e:
                parts.append(f"PAGE TEXT from {url}: (fetch failed: {e})")
    return "\n\n".join(parts), search_text


# ---------------------------------------------------------------- pipeline

def classify(company: dict, cert: str, evidence_text: str, model: str | None) -> dict:
    hint = ", ".join(filter(None, [company.get("location", ""), company.get("address", ""),
                                   company.get("industry_group", "")]))
    user = (
        f"Company: {company['company']}\n"
        f"Unverified location/address hints (may be wrong): {hint or 'none'}\n"
        f"Certification to assess: {cert}\n\n"
        f"Evidence:\n{evidence_text[:12000]}"
    )
    raw = llm_client.chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        model=model,
    )
    try:
        verdict = llm_client.extract_json(raw)
    except ValueError:
        verdict = {"org_status": "not_found", "facility_status": "unknown_facility",
                   "confidence": 0.0, "evidence": f"unparseable model output: {raw[:150]}"}
    if verdict.get("org_status") not in ORG_STATUSES:
        verdict["org_status"] = "not_found"
    if verdict.get("facility_status") not in FACILITY_STATUSES:
        verdict["facility_status"] = "unknown_facility"
    verdict["final_label"] = derive_final_label(verdict)
    return verdict


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "company"


def write_company_page(pages_dir: Path, company: dict, rows: list[dict]) -> Path:
    """One page per company, in the agreed format, covering all certifications."""
    lines = [
        f"company: {company['company']}",
        f"facility: {company.get('location', '') or 'unknown (sheet hint missing)'}",
        f"address: {company.get('address', '') or 'unknown'}",
        "",
        "NOTE: facility/address above come from the source sheet and are unverified hints.",
        "",
    ]
    for r in rows:
        lines.append(f"{r['certification']}:")
        lines.append(f"  org_status: {r['org_status']}")
        lines.append(f"  facility_status: {r['facility_status']}")
        lines.append(f"  final_label: {r['final_label']}")
        lines.append(f"  certification_body: {r['certification_body'] or 'not identified'}")
        lines.append(f"  reference_no: {r['reference_no'] or 'not visible in sources checked'}")
        lines.append(f"  scope: {r['scope'] or 'not visible in sources checked'}")
        lines.append(f"  expiry_date: {r['expiry_date'] or 'not visible in sources checked'}")
        lines.append(f"  confidence: {r['confidence']}")
        lines.append(f"  evidence: {r['evidence']}")
        lines.append(f"  source: {r['source_url'] or 'n/a'}")
        lines.append("")
    lines.append(f"checked_at: {rows[-1]['checked_at']}  model: {rows[-1]['model']}  "
                 f"backend: {rows[-1]['backend']}")
    pages_dir.mkdir(parents=True, exist_ok=True)
    path = pages_dir / f"{slugify(company['company'])}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def load_done(out_path: Path) -> dict[tuple[str, str], dict]:
    if not out_path.exists():
        return {}
    with out_path.open(newline="", encoding="utf-8") as f:
        return {(r["company"], r["certification"]): r for r in csv.DictReader(f)}


def search_cert_name(cert: str) -> str:
    """'ISO 9001:2015' -> 'ISO 9001' for the search query."""
    return cert.split(":")[0].strip()


def main() -> None:
    _load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--companies", type=Path, default=DEFAULT_COMPANIES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--pages-dir", type=Path, default=DEFAULT_PAGES_DIR)
    ap.add_argument("--cert", action="append",
                    help=f"certification(s) to check (repeatable); default: {DEFAULT_CERTS}")
    ap.add_argument("--backend", choices=["auto", "browser", "http"], default="auto")
    ap.add_argument("--model", default=None, help="override LLM_MODEL env var")
    ap.add_argument("--limit", type=int, default=0, help="only process first N companies (0 = all)")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between web requests (be polite)")
    ap.add_argument("--fetch-pages", type=int, default=2,
                    help="how many top result pages to open per query (default 2; 0 = snippets only)")
    ap.add_argument("--use-location", action="store_true",
                    help="include the sheet's location in the search query "
                         "(off by default: sheet locations may be wrong)")
    ap.add_argument("--fresh", action="store_true", help="ignore existing results and start over")
    args = ap.parse_args()

    certs = args.cert or DEFAULT_CERTS

    if not llm_client.ping():
        raise SystemExit(
            f"Local LLM server not reachable at {llm_client.DEFAULT_BASE_URL}.\n"
            f"Start it first, e.g.:  ollama serve   (then: ollama pull qwen3:14b)")

    if not args.companies.exists():
        raise SystemExit(f"{args.companies} not found — run scripts/extract_companies.py first.")
    with args.companies.open(newline="", encoding="utf-8") as f:
        companies = list(csv.DictReader(f))
    if args.limit:
        companies = companies[: args.limit]

    backend = make_backend(args.backend)
    model = args.model or llm_client.DEFAULT_MODEL

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {} if args.fresh else load_done(args.out)
    mode = "w" if (args.fresh or not args.out.exists()) else "a"
    out_f = args.out.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=OUT_FIELDS)
    if mode == "w":
        writer.writeheader()
    log_f = args.log.open("w" if args.fresh else "a", encoding="utf-8")

    total = len(companies) * len(certs)
    n = 0
    try:
        for comp in companies:
            comp_rows = []
            for cert in certs:
                n += 1
                key = (comp["company"], cert)
                if key in done:
                    comp_rows.append(done[key])
                    continue
                query = f'"{comp["company"]}" {search_cert_name(cert)} certificate'
                if args.use_location and comp.get("location"):
                    query += f' {comp["location"]}'
                try:
                    evidence_text, search_text = gather_evidence(backend, query, args.fetch_pages)
                except Exception as e:
                    evidence_text = search_text = f"(search failed: {e})"
                verdict = classify(comp, cert, evidence_text, model)
                row = {
                    "company": comp["company"],
                    "certification": cert,
                    "org_status": verdict["org_status"],
                    "facility_status": verdict["facility_status"],
                    "final_label": verdict["final_label"],
                    "certification_body": str(verdict.get("certification_body") or ""),
                    "reference_no": str(verdict.get("reference_no") or ""),
                    "scope": str(verdict.get("scope") or "")[:500],
                    "expiry_date": str(verdict.get("expiry_date") or ""),
                    "confidence": verdict.get("confidence", ""),
                    "evidence": str(verdict.get("evidence") or "")[:500],
                    "source_url": str(verdict.get("source_url") or ""),
                    "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "model": model,
                    "backend": backend.name,
                }
                writer.writerow(row)
                out_f.flush()
                log_f.write(json.dumps({"query": query, "evidence_text": evidence_text[:6000], **row}) + "\n")
                log_f.flush()
                comp_rows.append(row)
                print(f"[{n}/{total}] {comp['company']} | {cert} -> {row['final_label']} "
                      f"({row['confidence']})")
                time.sleep(args.delay)
            if comp_rows:
                page = write_company_page(args.pages_dir, comp, comp_rows)
                print(f"    page: {page}")
    finally:
        out_f.close()
        log_f.close()

    print(f"\nDone. Summary CSV: {args.out}\nPer-company pages: {args.pages_dir}/\n"
          f"Evidence log: {args.log}")


if __name__ == "__main__":
    main()

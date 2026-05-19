"""
Analyze company coverage across dense, sparse, and reranker (retrieved context) retrievals
for the 50 rewritten queries in 20260519_015033_retrieval_only.xlsx.

For each row we:
  1) Take a hand-curated set of golden companies (from the golden answer column).
  2) Extract retrieved company names from each retrieval column.
  3) Compute coverage: # found / # total + missed names.
"""

import re
import openpyxl
from copy import copy

SRC = '/home/sm11926/GNEM_Retrival_RAG/georgia_ev_intelligence/outputs/hybrid_retrieval_rewritten_50/20260519_015033_retrieval_only.xlsx'
DST = '/home/sm11926/GNEM_Retrival_RAG/georgia_ev_intelligence/outputs/hybrid_retrieval_rewritten_50/20260519_015033_retrieval_only_with_coverage.xlsx'


# Hand-curated golden company sets per row (row 2-51 = question 1-50).
# Empty list means the golden answer is not a company list (county only, count only, "none", etc.).
GOLDEN: dict[int, list[str]] = {
    2: ['F&P Georgia Manufacturing', 'Fouts Brothers Fire Equipment', 'Hitachi Astemo',
        'Hitachi Astemo Americas Inc.', 'Hollingsworth & Vose Co.', 'Honda Development & Manufacturing',
        'Hwashin', 'Hyundai & LG Energy Solution (LGES)', 'Hyundai Industrial Co.',
        'Hyundai MOBIS (Georgia)', 'Hyundai Motor Group', 'Hyundai Transys Georgia Powertrain',
        'Hyundai Transys Georgia Seating Systems', 'IMMI', 'IMS Gear Georgia Inc.',
        'Inalfa Roof Systems Inc.', 'JAC Products Inc.', 'Jefferson Southern Corp.'],
    3: ['F&P Georgia Manufacturing', 'Hitachi Astemo Americas Inc.', 'Hollingsworth & Vose Co.',
        'Honda Development & Manufacturing', 'Hyundai Motor Group', 'IMMI'],
    4: ['ZF Gainesville LLC', 'Freudenberg-NOK', 'Hyundai Transys Georgia Powertrain',
        'Novelis Inc.', 'Peerless-Winsmith Inc.'],
    5: ['GSC Steel Stamping LLC', 'Morgan Corp.', 'Yazaki North America', 'ZF Gainesville LLC'],
    6: ['Kia Georgia Inc.', 'Minebea AccessSolutions USA Inc.', 'Superior Essex Inc.',
        'Suzuki Manufacturing of America Corp.', 'TCI Powder Coatings', 'TDK Components USA Inc.',
        'TE Connectivity', 'Teklas USA', 'Textron Specialized Vehicles', 'Thermal Ceramics Inc.',
        'Thomson Plastics Inc.'],
    7: ['Novelis Inc.'],
    8: ['WIKA USA'],
    9: [],   # County-level answer, no specific company listed
    10: [],  # County total only
    11: ['Kia Georgia Inc.', 'Superior Essex Inc.', 'Suzuki Manufacturing of America Corp.',
         'TCI Powder Coatings', 'TDK Components USA Inc.', 'TE Connectivity', 'Teklas USA',
         'Textron Specialized Vehicles', 'Thermal Ceramics Inc.', 'Thomson Plastics Inc.'],
    12: ['Sewon America Inc.'],
    13: ['Duckyang', 'GSC Steel Stamping LLC', 'Enchem America Inc.'],
    14: ['Duckyang', 'Enchem America Inc.', 'GSC Steel Stamping LLC',
         'Hyundai Transys Georgia Powertrain', 'Racemark International LLC',
         'Suzuki Manufacturing of America Corp.'],
    15: ['Duckyang', 'F&P Georgia Manufacturing', 'Hollingsworth & Vose Co.', 'IMMI'],
    16: ['WIKA USA', 'Woodbridge Foam Corp.'],
    17: ['Hollingsworth & Vose Co.', 'IMMI'],
    18: ['WIKA USA', 'Woodbridge Foam Corp.'],
    19: [],   # "There are no ..." — empty golden set
    20: ['Duckyang'],
    21: [],   # "No Georgia Tier 1/2 companies..." — empty
    22: ['F&P Georgia Manufacturing', 'Duckyang', 'GSC Steel Stamping LLC',
         'Hitachi Astemo Americas Inc.', 'Hollingsworth & Vose Co.',
         'Honda Development & Manufacturing', 'Hyundai Motor Group',
         'Hyundai Transys Georgia Powertrain', 'IMMI', 'Enchem America Inc.',
         'Racemark International LLC'],
    23: ['GSC Steel Stamping LLC', 'Hyundai MOBIS (Georgia)', 'Hyundai Transys Georgia Seating Systems'],
    24: ['Archer Aviation Inc.'],
    25: ['F&P Georgia Manufacturing', 'Hitachi Astemo Americas Inc.', 'Hollingsworth & Vose Co.',
         'Honda Development & Manufacturing', 'Hyundai Motor Group', 'IMMI'],
    26: ['Arising Industries Inc.', 'Dinex Emissions Inc.', 'ACM Georgia LLC',
         'Dongwon Autopart Technology Georgia LLC', 'FOX Factory', 'AVS', 'Flambeau Inc.', 'Grudem'],
    27: ['Archer Aviation Inc.', 'Arising Industries Inc.'],
    28: ['Yamaha Motor Manufacturing Corp.', 'Vanguard National Trailer Corp.', 'Morgan Corp.',
         'Vista Metals Corp.', 'Voestalpine Automotive Body Parts Inc.', 'ZF Gainesville LLC',
         'Yachiyo Manufacturing of America LLC', 'YKK USA Inc.', 'Woodbridge Foam Corp.',
         'WIKA USA', 'Woory Industrial Co.', 'Wabash National Corp.', 'Wheelabrator Group Inc.',
         'Trenton Pressing Inc.', 'Tie Down Engineering', 'Trenton Pressing', 'TI Fluid Systems',
         'TN Americas Holding Inc.', 'Toyota Industries Group (TACG-TICA)', 'Volvo Cars USA',
         'GSC Steel Stamping LLC', 'Yazaki North America', 'Vernay', 'Volvo Group North America',
         'Valeo'],
    29: ['F&P Georgia Manufacturing', 'Hitachi Astemo Americas Inc.', 'Hollingsworth & Vose Co.',
         'Honda Development & Manufacturing', 'Hyundai Motor Group', 'IMMI'],
    30: ['Enchem America Inc.', 'F&P Georgia Manufacturing', 'Hyundai Motor Group',
         'Hyundai Transys Georgia Powertrain', 'IMMI', 'Racemark International LLC'],
    31: ['Duckyang', 'Enchem America Inc.'],
    32: ['Fouts Brothers Fire Equipment', 'Hitachi Astemo', 'Hwashin',
         'Hyundai & LG Energy Solution (LGES)', 'IMS Gear Georgia Inc.', 'JAC Products Inc.',
         'Hyundai Industrial Co.', 'Hyundai MOBIS (Georgia)', 'Hyundai Transys Georgia Seating Systems',
         'Inalfa Roof Systems Inc.', 'Jefferson Southern Corp.'],
    33: ['Freudenberg-NOK', 'Hyundai Transys Georgia Powertrain', 'Peerless-Winsmith Inc.'],
    34: ['TI Fluid Systems', 'Tie Down Engineering', 'TN Americas Holding Inc.',
         'Toyota Industries Group (TACG-TICA)', 'Trenton Pressing', 'Yazaki North America',
         'YKK USA Inc.', 'ZF Gainesville LLC', 'Trenton Pressing Inc.', 'Valeo',
         'Vanguard National Trailer Corp.', 'Vernay', 'Vista Metals Corp.',
         'Voestalpine Automotive Body Parts Inc.', 'Volvo Cars USA', 'Volvo Group North America',
         'Wabash National Corp.', 'Wheelabrator Group Inc.', 'WIKA USA', 'Woodbridge Foam Corp.',
         'Woory Industrial Co.', 'Yachiyo Manufacturing of America LLC',
         'Yamaha Motor Manufacturing Corp.'],
    35: ['JTEKT North America Corp.', 'Kautex Inc.', 'Lark United Manufacturing Inc.',
         'Fouts Brothers Fire Equipment', 'Arising Industries Inc.', 'Dinex Emissions Inc.',
         'Lund International Inc.', 'Mack Trucks', 'Mando America Corp.', 'ACM Georgia LLC'],
    36: ['F&P Georgia Manufacturing', 'Hitachi Astemo Americas Inc.', 'Hollingsworth & Vose Co.',
         'Honda Development & Manufacturing', 'IMMI'],
    37: ['Bridgestone Bandag', 'Dinex Emissions Inc.', 'Down 2 Earth Trailers', 'Eaton Corp.',
         'Ecoplastic America Corporation', 'Erdrich USA Inc.', 'Global Powertrain Systems LLC'],
    38: ['GSC Steel Stamping LLC'],
    39: ['TCI Powder Coatings', 'Teklas USA', 'Textron Specialized Vehicles'],
    40: ['ZF Gainesville LLC', 'Novelis Inc.', 'Freudenberg-NOK', 'Peerless-Winsmith Inc.'],
    41: ['Freudenberg-NOK', 'Hyundai Transys Georgia Powertrain', 'Novelis Inc.',
         'Peerless-Winsmith Inc.', 'ZF Gainesville LLC'],
    42: ['Fouts Brothers Fire Equipment', 'Hyundai Industrial Co.', 'Hyundai MOBIS (Georgia)',
         'Hyundai Transys Georgia Seating Systems', 'Inalfa Roof Systems Inc.',
         'Jefferson Southern Corp.'],
    43: ['Freudenberg-NOK', 'Hyundai Transys Georgia Powertrain', 'Novelis Inc.',
         'Peerless-Winsmith Inc.'],
    44: ['Enplas USA Inc.', 'EVCO Plastics', 'F&P Georgia Manufacturing'],
    45: ['Racemark International LLC'],
    46: ['Duckyang', 'GSC Steel Stamping LLC', 'Hyundai Transys Georgia Powertrain',
         'Enchem America Inc.', 'Racemark International LLC'],
    47: [],   # County list only
    48: [],   # Location count only
    49: [],   # Location list only
    50: ['Archer Aviation Inc.', 'Arising Industries Inc.'],
    51: ['Racemark International LLC'],
}


def normalize_name(name: str) -> str:
    """Lowercase, strip trailing punctuation/whitespace and collapse spaces."""
    s = name.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.rstrip('.,;:|')
    return s.strip()


def extract_retrieved_companies(text: str) -> list[str]:
    """Pull every `company: <name>` line out of a retrieval column."""
    if not text:
        return []
    found = set()
    for line in text.splitlines():
        m = re.search(r'(?i)^\s*company\s*:\s*(.+?)\s*$', line)
        if m:
            found.add(normalize_name(m.group(1)))
    return sorted(found)


def is_match(golden: str, retrieved_set: set[str]) -> bool:
    """Match a golden company against the normalised retrieved set.

    Strategy:
      1. Exact normalised match.
      2. Substring match in both directions, but only if the longer name's first word
         matches the shorter one's first word — this stops 'Hitachi Astemo' falsely
         matching 'Hitachi Astemo Americas Inc.' (and vice versa).
    """
    g = normalize_name(golden)
    if g in retrieved_set:
        return True
    g_tokens = g.split()
    for r in retrieved_set:
        r_tokens = r.split()
        # Strict exact-after-normalisation already failed; check for parenthetical /
        # punctuation-only differences by removing parens.
        g_clean = re.sub(r'[()]', '', g).strip()
        r_clean = re.sub(r'[()]', '', r).strip()
        g_clean = re.sub(r'\s+', ' ', g_clean)
        r_clean = re.sub(r'\s+', ' ', r_clean)
        if g_clean == r_clean:
            return True
    return False


def coverage_cell(total: int, missed: list[str]) -> str:
    if total == 0:
        return 'N/A (golden answer has no company list)'
    if not missed:
        return f'Total companies: {total} — all companies covered.'
    return (f'Total companies: {total} — missed {len(missed)}: '
            + ', '.join(missed))


def main():
    wb = openpyxl.load_workbook(SRC)
    ws = wb['Sheet1']

    # Preserve original column widths / row heights / cell styles by writing into a copy.
    # We append three new columns at positions 7, 8, 9.
    new_headers = ['dense coverage', 'retrieved context coverage', 'sparse coverage']
    base_col = 7
    for i, h in enumerate(new_headers):
        c = ws.cell(1, base_col + i, h)
        # Copy header style from an existing header cell
        src_style = ws.cell(1, 1)
        c.font = copy(src_style.font)
        c.fill = copy(src_style.fill)
        c.alignment = copy(src_style.alignment)
        c.border = copy(src_style.border)

    summary_rows = []
    for r in range(2, ws.max_row + 1):
        sno = ws.cell(r, 1).value
        retrieved_ctx_text = ws.cell(r, 4).value or ''
        dense_text = ws.cell(r, 5).value or ''
        sparse_text = ws.cell(r, 6).value or ''

        golden_companies = GOLDEN.get(r, [])
        # de-dup preserving order
        seen = set()
        golden_unique = []
        for g in golden_companies:
            key = normalize_name(g)
            if key not in seen:
                seen.add(key)
                golden_unique.append(g)

        dense_set = set(extract_retrieved_companies(dense_text))
        ctx_set = set(extract_retrieved_companies(retrieved_ctx_text))
        sparse_set = set(extract_retrieved_companies(sparse_text))

        missed_dense = [g for g in golden_unique if not is_match(g, dense_set)]
        missed_ctx = [g for g in golden_unique if not is_match(g, ctx_set)]
        missed_sparse = [g for g in golden_unique if not is_match(g, sparse_set)]

        total = len(golden_unique)
        ws.cell(r, base_col + 0, coverage_cell(total, missed_dense))
        ws.cell(r, base_col + 1, coverage_cell(total, missed_ctx))
        ws.cell(r, base_col + 2, coverage_cell(total, missed_sparse))

        # Apply wrap-text and top-align like the rest of the sheet
        for col_offset in range(3):
            cell = ws.cell(r, base_col + col_offset)
            cell.alignment = openpyxl.styles.Alignment(wrap_text=True, vertical='top')

        summary_rows.append((sno, total,
                             total - len(missed_dense),
                             total - len(missed_ctx),
                             total - len(missed_sparse)))

    # Set sensible widths for the new columns
    for i in range(3):
        col_letter = openpyxl.utils.get_column_letter(base_col + i)
        ws.column_dimensions[col_letter].width = 55

    wb.save(DST)

    print(f'Wrote {DST}\n')
    print(f"{'sno':>4} {'#gold':>6} {'dense':>6} {'ctx':>6} {'sparse':>6}")
    for sno, total, d, c, s in summary_rows:
        print(f'{sno!s:>4} {total:>6} {d:>6} {c:>6} {s:>6}')


if __name__ == '__main__':
    main()

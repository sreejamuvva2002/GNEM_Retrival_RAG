import sys; sys.path.insert(0, '.')
from phase1_extraction.kb_loader import load_companies_from_excel
companies = load_companies_from_excel()
print(f'Total loaded: {len(companies)}')
print()
print('First 5 companies with locations:')
for c in companies[:5]:
    name = c.get("company_name", "")[:40]
    tier = c.get("tier", "") or ""
    city = c.get("location_city", "") or ""
    county = c.get("location_county", "") or ""
    prod = (c.get("products_services", "") or "")[:40]
    print(f'  {name:<40}  {tier:<15}  {city} | {county}')
    print(f'  {"":40}  Products: {prod}')
    print()

with_city = [c for c in companies if c.get('location_city')]
print(f'Companies with location_city set: {len(with_city)}/{len(companies)}')

from ddgs import DDGS

ddgs = DDGS()
res = list(ddgs.text("Duckyang Georgia", backend="html", max_results=2))
print("Results length:", len(res))
if res:
    print("First result:", res[0])
else:
    # Try default backend too
    res_default = list(ddgs.text("Duckyang Georgia", max_results=2))
    print("Default backend results length:", len(res_default))

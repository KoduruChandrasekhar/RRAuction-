"""Step 9 - render the report HTML to PDF with headless Edge/Chromium (Playwright).
Usage: python -I analysis/09_render.py <in_html> <out_pdf>"""
import sys, pathlib
from playwright.sync_api import sync_playwright
src, dst = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve()
with sync_playwright() as pw:
    try:
        br = pw.chromium.launch(channel='msedge')
    except Exception:
        br = pw.chromium.launch()
    pg = br.new_page(viewport={'width': 1280, 'height': 720})
    pg.goto(src.as_uri(), wait_until='networkidle')
    pg.evaluate('document.fonts.ready')
    pg.wait_for_timeout(800)
    pg.pdf(path=str(dst), width='13.333in', height='7.5in', print_background=True, margin={'top': '0', 'bottom': '0', 'left': '0', 'right': '0'})
    br.close()
print('pdf written', dst, round(dst.stat().st_size / 1e6, 2), 'MB')

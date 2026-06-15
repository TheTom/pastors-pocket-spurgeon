"""Headless validation: confirm streaming is visible, bars gone, both tabs work."""
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"

def bot_len(pg):
    return pg.evaluate(r"""() => {
      const b = document.querySelectorAll('.bubble [data-testid="bot"]');
      if (!b.length) return 0;
      return (b[b.length-1].innerText || '').length;
    }""")

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 1000})
    pg.goto(URL, wait_until="domcontentloaded")
    pg.wait_for_selector(".inputrow textarea", timeout=20000)
    pg.wait_for_timeout(1000)

    ta = pg.query_selector(".inputrow textarea")
    ta.fill("Tell me about sola fide")
    ta.press("Enter")

    # poll bot message length over time -> proves visible streaming
    lens = []
    for _ in range(14):
        pg.wait_for_timeout(600)
        lens.append(bot_len(pg))
    print("bot text length over time (every 0.6s):", lens)
    growing = sum(1 for i in range(1, len(lens)) if lens[i] > lens[i-1])
    print("VISIBLE STREAMING:", "YES" if growing >= 4 else "NO", f"({growing} growth steps)")

    # zoom screenshot of the user bubble to eyeball bars
    u = pg.query_selector('.bubble [data-testid="user"]')
    if u:
        u.screenshot(path="/tmp/val_userbubble.png")
        print("saved user-bubble zoom")

    # bars check
    bars = pg.evaluate(r"""() => {
      let n = 0;
      const scan = el => { const cs = getComputedStyle(el);
        if (parseFloat(cs.borderLeftWidth)>0 && cs.borderLeftStyle!=='none' && el.tagName==='SPAN') n++;
        for (const c of el.children) scan(c); };
      document.querySelectorAll('.bubble [data-testid="user"], .bubble [data-testid="bot"]').forEach(scan);
      return n;
    }""")
    print("stray span-borders (bars) remaining:", bars)

    # test Sermon Review tab
    tabs = pg.query_selector_all("button[role='tab']")
    print("tabs:", [t.inner_text() for t in tabs])
    for t in tabs:
        if "Review" in t.inner_text():
            t.click(); break
    pg.wait_for_timeout(800)
    ta2 = pg.query_selector_all(".inputrow textarea")[-1]
    ta2.fill("Grade my sermon: God is love, so just be nice. Amen.")
    ta2.press("Enter")
    pg.wait_for_timeout(6000)
    rev = pg.evaluate(r"""() => {
      const b = document.querySelectorAll('.bubble [data-testid="bot"]');
      return b.length ? (b[b.length-1].innerText||'').length : 0; }""")
    print("Sermon Review reply length:", rev, "->", "WORKS" if rev > 50 else "EMPTY")
    pg.screenshot(path="/tmp/val_review.png", full_page=True)
    b.close()

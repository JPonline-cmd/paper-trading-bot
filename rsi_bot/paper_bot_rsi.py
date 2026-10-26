"""
PAPER TRADING BOT 2  —  same entries as bot 1, plus RSI exit  (30-minute chart)
Markets : Delta Exchange India  BTCUSD + ETHUSD perpetual  (public data only)
Capital : Rs 50,000 virtual per market.  NO real orders, NO API key.

Rules (same as the TradingView strategy "EMA 20/50/200 + Volume + 4H Trend"):
  LONG  : EMA20 crosses above EMA50 on a closed 30m candle, close > EMA200,
          volume > 20-candle average volume, 4H trend UP
  SHORT : mirror rules, 4H trend DOWN
  EXIT  : opposite EMA20/EMA50 cross (an opposite signal also reverses the trade)
           OR RSI(14) exit: SHORT -> RSI went below 30 during the trade, then crosses back above 60
                            LONG  -> RSI went above 70 during the trade, then crosses back below 40
           (backtest Jan-Oct 2026: +71.2% vs +52.6% for bot 1, max drawdown -14.1% vs -20.6%)
  4H trend UP   = last CLOSED 4H candle: EMA9 > EMA100 and close > EMA200
  4H trend DOWN = EMA9 < EMA100 and close < EMA200
  Orders fill at the NEXT 30m candle's open. Cost 0.05% fee + 0.02% slippage per side.

Run:  python paper_bot.py          -> runs forever, checks after every 30m candle closes
      python paper_bot.py --once   -> one check, then exit
      python paper_bot.py --status -> print current positions and P&L
Files written next to this script: state.json, trades.csv, equity.csv, bot.log
"""
import json, os, sys, time, csv, datetime as dt
import requests
import pandas as pd

# ---------------- settings ----------------
SYMBOLS      = ["BTCUSD", "ETHUSD"]
CAPITAL      = 50_000          # Rs per market
FEE          = 0.0005          # 0.05% per side
SLIP         = 0.0002          # 0.02% per side
RSI_N, RSI_ARM, RSI_EXIT = 14, 30, 60   # short: arm below 30, exit above 60 (long: above 70 / below 40)
TF, TF_SEC   = "30m", 1800
H4, H4_SEC   = "4h", 14400
API          = "https://api.india.delta.exchange/v2/history/candles"
HERE         = os.path.dirname(os.path.abspath(__file__))
STATE_F      = os.path.join(HERE, "state.json")
TRADES_F     = os.path.join(HERE, "trades.csv")
EQUITY_F     = os.path.join(HERE, "equity.csv")
LOG_F        = os.path.join(HERE, "bot.log")
IST          = dt.timezone(dt.timedelta(hours=5, minutes=30))


def ist(ts):  # unix seconds -> "2026-10-24 22:30" IST
    return dt.datetime.fromtimestamp(int(ts), IST).strftime("%Y-%m-%d %H:%M")


def log(msg):
    line = f"[{dt.datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_F, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------- data ----------------
def fetch(symbol, res, sec, bars):
    now = int(time.time())
    r = requests.get(API, params=dict(resolution=res, symbol=symbol, start=now - bars * sec, end=now), timeout=30)
    r.raise_for_status()
    j = r.json()
    if not j.get("success") or not j.get("result"):
        raise RuntimeError(f"Delta API returned no candles for {symbol} {res}: {str(j)[:200]}")
    df = pd.DataFrame(j["result"])[["time", "open", "high", "low", "close", "volume"]].astype(float)
    df["time"] = df.time.astype(int)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def prepare(df30, df4):
    """Indicators on 30m candles + the 4H trend known at each 30m candle's close."""
    d = df30.copy()
    c = d.close
    d["e20"], d["e50"], d["e200"] = ema(c, 20), ema(c, 50), ema(c, 200)
    d["volavg"] = d.volume.rolling(20).mean()
    d["up"] = (d.e20 > d.e50) & (d.e20.shift() <= d.e50.shift())
    d["dn"] = (d.e20 < d.e50) & (d.e20.shift() >= d.e50.shift())
    ch = c.diff()
    gain = ch.clip(lower=0).ewm(alpha=1 / RSI_N, adjust=False).mean()
    loss = (-ch).clip(lower=0).ewm(alpha=1 / RSI_N, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss)
    h = df4.copy()
    t9, t100, t200 = ema(h.close, 9), ema(h.close, 100), ema(h.close, 200)
    h["trend"] = 0
    h.loc[(t9 > t100) & (h.close > t200), "trend"] = 1
    h.loc[(t9 < t100) & (h.close < t200), "trend"] = -1
    h["avail"] = h.time + H4_SEC                       # usable only after the 4H candle closes
    d["close_time"] = d.time + TF_SEC
    d = pd.merge_asof(d.sort_values("close_time"), h[["avail", "trend"]].rename(columns={"avail": "close_time"}),
                      on="close_time", direction="backward")
    d["trend"] = d.trend.fillna(0).astype(int)
    vol_ok = d.volume > d.volavg
    d["long_sig"] = d.up & (c > d.e200) & vol_ok & (d.trend == 1)
    d["short_sig"] = d.dn & (c < d.e200) & vol_ok & (d.trend == -1)
    return d.reset_index(drop=True)


# ---------------- state / files ----------------
def new_market():
    return dict(equity=CAPITAL, pos=0, entry=None, entry_time=None, last_bar=None, trades=0, wins=0, armed=False)


def load_state():
    if os.path.exists(STATE_F):
        with open(STATE_F, encoding="utf-8") as f:
            return json.load(f)
    return dict(started=dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M"), markets={s: new_market() for s in SYMBOLS})


def save_state(st):
    tmp = STATE_F + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATE_F)


def append_csv(path, row, header):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row)


# ---------------- core: replay closed candles ----------------
def close_trade(sym, m, px_raw, t, reason):
    side = m["pos"]
    px = px_raw * (1 - SLIP * side)
    ret = side * (px - m["entry"]) / m["entry"] - 2 * FEE
    pnl = m["equity"] * ret
    m["equity"] += pnl
    m["trades"] += 1
    m["wins"] += 1 if pnl > 0 else 0
    append_csv(TRADES_F, [sym, "LONG" if side == 1 else "SHORT", ist(m["entry_time"]), round(m["entry"], 2),
                          ist(t), round(px, 2), round(100 * ret, 2), round(pnl, 2), round(m["equity"], 2), reason],
               ["market", "side", "entry_time_IST", "entry_price", "exit_time_IST", "exit_price",
                "return_pct", "pnl_rs", "equity_rs", "exit_reason"])
    log(f"{sym}: EXIT {'LONG' if side == 1 else 'SHORT'} @ {px:,.2f}  ({100 * ret:+.2f}%, Rs {pnl:+,.0f})  "
        f"equity Rs {m['equity']:,.0f}  [{reason}]")
    m["pos"], m["entry"], m["entry_time"], m["armed"] = 0, None, None, False


def open_trade(sym, m, side, px_raw, t):
    m["pos"] = side
    m["entry"] = px_raw * (1 + SLIP * side)
    m["entry_time"] = int(t)
    m["armed"] = False
    log(f"{sym}: ENTER {'LONG' if side == 1 else 'SHORT'} @ {m['entry']:,.2f}  (paper, Rs {m['equity']:,.0f})")


def process(sym, m, d, now):
    """Act on every closed candle after m['last_bar']. A signal on candle i fills at candle i+1's open."""
    closed = d[d.time + TF_SEC <= now].reset_index(drop=True)
    if len(closed) < 260:
        raise RuntimeError(f"{sym}: not enough history ({len(closed)} closed candles)")
    if m["last_bar"] is None:                      # first run: start from the latest closed candle, no back-trading
        m["last_bar"] = int(closed.time.iloc[-1])
        log(f"{sym}: paper trading starts after candle {ist(m['last_bar'])} IST")
        return
    todo = d.index[(d.time > m["last_bar"]) & (d.time + TF_SEC <= now)]
    for i in todo:
        if i + 1 >= len(d):                        # next candle not opened yet -> act next run
            break
        row, nxt = d.iloc[i], d.iloc[i + 1]
        fill, t = float(nxt.open), int(nxt.time)
        r, rp = float(row.rsi), float(d.rsi.iloc[i - 1])
        if m["pos"] == -1:
            if r < RSI_ARM: m["armed"] = True
            if m.get("armed") and r > RSI_EXIT and rp <= RSI_EXIT:
                close_trade(sym, m, fill, t, f"RSI went below {RSI_ARM}, now back above {RSI_EXIT}")
        elif m["pos"] == 1:
            if r > 100 - RSI_ARM: m["armed"] = True
            if m.get("armed") and r < 100 - RSI_EXIT and rp >= 100 - RSI_EXIT:
                close_trade(sym, m, fill, t, f"RSI went above {100 - RSI_ARM}, now back below {100 - RSI_EXIT}")
        if m["pos"] == 1 and row.dn:  close_trade(sym, m, fill, t, "EMA20 crossed below EMA50")
        if m["pos"] == -1 and row.up: close_trade(sym, m, fill, t, "EMA20 crossed above EMA50")
        if row.long_sig and m["pos"] != 1:
            if m["pos"] == -1: close_trade(sym, m, fill, t, "reverse to LONG")
            open_trade(sym, m, 1, fill, t)
        elif row.short_sig and m["pos"] != -1:
            if m["pos"] == 1: close_trade(sym, m, fill, t, "reverse to SHORT")
            open_trade(sym, m, -1, fill, t)
        m["last_bar"] = int(row.time)


def mark(m, price):
    if not m["pos"]:
        return m["equity"]
    return m["equity"] * (1 + m["pos"] * (price * (1 - SLIP * m["pos"]) - m["entry"]) / m["entry"] - 2 * FEE)


def run_once(st):
    now = int(time.time())
    total = 0.0
    lines = []
    for sym in SYMBOLS:
        m = st["markets"].setdefault(sym, new_market())
        d = prepare(fetch(sym, TF, TF_SEC, 1200), fetch(sym, H4, H4_SEC, 700))
        process(sym, m, d, now)
        last = float(d.close.iloc[-1])
        eq = mark(m, last)
        total += eq
        closed = d[d.time + TF_SEC <= now]
        trend, rsi_now = int(closed.trend.iloc[-1]), float(closed.rsi.iloc[-1])
        pos = {1: "LONG", -1: "SHORT", 0: "no position"}[m["pos"]]
        lines.append(f"{sym}: price {last:,.1f} | 4H trend {['DOWN','SIDEWAYS','UP'][trend + 1]} | {pos}"
                     + (f" from {m['entry']:,.1f}" if m["pos"] else "") + f" | RSI {rsi_now:.0f}"
                     + (" (RSI exit armed)" if m["pos"] and m.get("armed") else "")
                     + f" | equity Rs {eq:,.0f} ({100 * (eq / CAPITAL - 1):+.2f}%) | trades {m['trades']}, wins {m['wins']}")
        append_csv(EQUITY_F, [dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M"), sym, round(last, 2), pos, round(eq, 2)],
                   ["time_IST", "market", "price", "position", "equity_rs"])
    save_state(st)
    write_status(st, total, lines)
    log(" || ".join(lines))
    log(f"TOTAL paper equity Rs {total:,.0f} on Rs {CAPITAL * len(SYMBOLS):,} ({100 * (total / (CAPITAL * len(SYMBOLS)) - 1):+.2f}%)")


def write_status(st, total, lines):
    """Human-readable summary for phone viewing on GitHub."""
    cap = CAPITAL * len(SYMBOLS)
    out = ["# Paper Bot 2 — same entries + RSI exit (below 30, then back above 60)", "",
           "Compare with bot 1: [../STATUS.md](../STATUS.md)", "",
           f"**Last update:** {dt.datetime.now(IST).strftime('%d %b %Y, %H:%M')} IST  ",
           f"**Paper trading since:** {st['started']} IST  ",
           f"**Total paper equity:** Rs {total:,.0f} on Rs {cap:,} (**{100 * (total / cap - 1):+.2f}%**)", "",
           "## Markets", ""]
    out += [f"- {l}" for l in lines]
    out += ["", "## Last 15 closed trades", ""]
    if os.path.exists(TRADES_F):
        t = pd.read_csv(TRADES_F).tail(15).iloc[::-1]
        out.append("| Market | Side | Entry (IST) | Entry | Exit (IST) | Exit | Return % | P&L Rs |")
        out.append("|---|---|---|---|---|---|---|---|")
        for _, r in t.iterrows():
            out.append(f"| {r.market} | {r.side} | {r.entry_time_IST} | {r.entry_price:,.2f} | {r.exit_time_IST} | "
                       f"{r.exit_price:,.2f} | {r.return_pct:+.2f} | {r.pnl_rs:+,.0f} |")
    else:
        out.append("Abhi tak koi trade band nahi hua.")
    out += ["", "_Paper trading only — no real orders. Not financial advice._", ""]
    with open(os.path.join(HERE, "STATUS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def status(st):
    for sym, m in st["markets"].items():
        print(f"{sym}: equity Rs {m['equity']:,.0f}, position {m['pos']}, entry {m['entry']}, trades {m['trades']}, wins {m['wins']}")


def main():
    st = load_state()
    if "--status" in sys.argv:
        status(st); return
    log(f"Paper bot 2 (RSI exit) started (paper since {st['started']} IST). Markets: {', '.join(SYMBOLS)}. NO real orders.")
    while True:
        try:
            run_once(st)
        except Exception as e:
            log(f"ERROR: {e}  (will retry next cycle)")
        if "--once" in sys.argv:
            break
        now = time.time()
        wake = (int(now // TF_SEC) + 1) * TF_SEC + 20          # 20 s after the next 30m candle closes
        log(f"Next check at {ist(wake)} IST")
        time.sleep(max(5, wake - now))


if __name__ == "__main__":
    main()

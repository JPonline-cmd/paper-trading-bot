"""
PAPER TRADING BOT 3  —  9 EMA + VWAP Pullback  (5-minute chart)
Markets : Delta Exchange India  BTCUSD + ETHUSD perpetual  (public data only)
Capital : Rs 50,000 virtual per market.  NO real orders, NO API key.

Rules (same as the TradingView strategy "9 EMA + VWAP Pullback"):
  VWAP  : daily VWAP, resets 00:00 UTC (05:30 IST)
  BIAS  LONG  : close > VWAP, EMA9 > VWAP, EMA9 rising (EMA9 > EMA9 three candles ago)
  PULLBACK    : previous candle's low touched EMA9 (0.1% tolerance) and it closed above VWAP
  TRIGGER     : this candle closes above the pullback candle's high  -> BUY at next candle's open
  SHORT       : exact mirror
  CHOP FILTER : skip if price crossed VWAP 3+ times in the last 20 candles
  STOP        : below the lower low of pullback + trigger candle (0.05% buffer)
  SIZE        : risk 1% of equity per trade, notional capped at 3x equity
  TARGET      : 50% booked at 2R, then stop moves to entry (breakeven)
  EXIT REST   : first 5m close below EMA9 (or below VWAP)  -> exit at next open
  DAY LIMITS  : max 3 trades per market per UTC day, stop for the day after 2 losing trades
  COSTS       : 0.05% fee + 0.02% slippage per side
  If stop and target are both touched inside one candle, the stop is assumed first (conservative).

Run:  python vwap_bot/paper_bot_vwap.py          -> runs forever, checks after every 5m candle
      python vwap_bot/paper_bot_vwap.py --once   -> one check, then exit (GitHub Actions)
      python vwap_bot/paper_bot_vwap.py --status -> print current positions and P&L
Files written next to this script: state.json, trades.csv, equity.csv, bot.log, STATUS.md
"""
import json, os, sys, time, csv, datetime as dt
import requests
import pandas as pd

# ---------------- settings ----------------
SYMBOLS       = ["BTCUSD", "ETHUSD"]
CAPITAL       = 50_000          # Rs per market
FEE           = 0.0005          # 0.05% per side
SLIP          = 0.0002          # 0.02% per side
TF, TF_SEC    = "5m", 300
FETCH_BARS    = 1500            # ~5 days of 5m candles
EMA_LEN       = 9
SLOPE_BARS    = 3
TOUCH         = 0.0010          # 0.10% pullback touch tolerance
STOP_BUF      = 0.0005          # 0.05% beyond the swing low/high
RISK          = 0.01            # 1% of equity per trade
RR            = 2.0             # partial target at 2R
PARTIAL       = 0.5             # book 50% there
MAX_LEV       = 3.0
CHOP_LOOK     = 20
CHOP_MAX      = 3
MAX_TRADES    = 3
MAX_LOSSES    = 2
API           = "https://api.india.delta.exchange/v2/history/candles"
HERE          = os.path.dirname(os.path.abspath(__file__))
STATE_F       = os.path.join(HERE, "state.json")
TRADES_F      = os.path.join(HERE, "trades.csv")
EQUITY_F      = os.path.join(HERE, "equity.csv")
LOG_F         = os.path.join(HERE, "bot.log")
IST           = dt.timezone(dt.timedelta(hours=5, minutes=30))


def ist(ts):
    return dt.datetime.fromtimestamp(int(ts), IST).strftime("%Y-%m-%d %H:%M")


def utc_day(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d")


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
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def prepare(df):
    d = df.copy()
    c = d.close
    d["ema9"] = c.ewm(span=EMA_LEN, adjust=False).mean()
    # daily VWAP anchored at 00:00 UTC
    tp = (d.high + d.low + d.close) / 3
    day = pd.to_datetime(d.time, unit="s", utc=True).dt.date
    pv = (tp * d.volume).groupby(day).cumsum()
    vv = d.volume.groupby(day).cumsum()
    d["vwap"] = (pv / vv.replace(0, pd.NA)).astype(float).fillna(tp)
    up_slope = d.ema9 > d.ema9.shift(SLOPE_BARS)
    dn_slope = d.ema9 < d.ema9.shift(SLOPE_BARS)
    above = (c > d.vwap).astype(int)
    crosses = (above.diff().abs() == 1).astype(int)
    not_choppy = crosses.rolling(CHOP_LOOK).sum() < CHOP_MAX
    bull = (c > d.vwap) & (d.ema9 > d.vwap) & up_slope
    bear = (c < d.vwap) & (d.ema9 < d.vwap) & dn_slope
    pb_long = bull.shift(1, fill_value=False) & (d.low.shift(1) <= d.ema9.shift(1) * (1 + TOUCH)) & (c.shift(1) > d.vwap.shift(1))
    pb_short = bear.shift(1, fill_value=False) & (d.high.shift(1) >= d.ema9.shift(1) * (1 - TOUCH)) & (c.shift(1) < d.vwap.shift(1))
    d["long_sig"] = pb_long & (c > d.high.shift(1)) & bull & not_choppy
    d["short_sig"] = pb_short & (c < d.low.shift(1)) & bear & not_choppy
    d["stop_long"] = pd.concat([d.low, d.low.shift(1)], axis=1).min(axis=1) * (1 - STOP_BUF)
    d["stop_short"] = pd.concat([d.high, d.high.shift(1)], axis=1).max(axis=1) * (1 + STOP_BUF)
    return d


# ---------------- state / files ----------------
def new_market():
    return dict(equity=CAPITAL, pos=0, notional=0.0, entry=None, entry_time=None, sl=None, tp=None,
                tp1_done=False, trade_pnl=0.0, day=None, trades_today=0, losses_today=0,
                last_bar=None, trades=0, wins=0)


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


# ---------------- trade mechanics ----------------
def side_name(s):
    return "LONG" if s == 1 else "SHORT"


def exit_part(sym, m, px_raw, t, frac, reason):
    """Close `frac` of the open notional at px_raw (slippage + fee applied)."""
    side = m["pos"]
    px = px_raw * (1 - SLIP * side)
    part = m["notional"] * frac
    pnl = part * side * (px - m["entry"]) / m["entry"] - part * 2 * FEE
    m["equity"] += pnl
    m["trade_pnl"] += pnl
    m["notional"] -= part
    full = frac >= 0.999
    append_csv(TRADES_F, [sym, side_name(side), ist(m["entry_time"]), round(m["entry"], 2), ist(t), round(px, 2),
                          "FULL" if full else f"{int(frac * 100)}%", round(pnl, 2), round(m["equity"], 2), reason],
               ["market", "side", "entry_time_IST", "entry_price", "exit_time_IST", "exit_price",
                "portion", "pnl_rs", "equity_rs", "exit_reason"])
    log(f"{sym}: EXIT {'' if full else 'PARTIAL '}{side_name(side)} @ {px:,.2f}  (Rs {pnl:+,.0f})  "
        f"equity Rs {m['equity']:,.0f}  [{reason}]")
    if full:
        m["trades"] += 1
        if m["trade_pnl"] > 0:
            m["wins"] += 1
        else:
            m["losses_today"] += 1
        m.update(pos=0, notional=0.0, entry=None, entry_time=None, sl=None, tp=None, tp1_done=False, trade_pnl=0.0)


def open_trade(sym, m, side, px_raw, stop, t):
    entry = px_raw * (1 + SLIP * side)
    dist = side * (entry - stop)
    if dist <= 0:
        log(f"{sym}: {side_name(side)} signal skipped — opened beyond the stop")
        return
    frac = min(RISK / (dist / entry), MAX_LEV)
    m.update(pos=side, notional=m["equity"] * frac, entry=entry, entry_time=int(t), sl=stop,
             tp=entry + side * RR * dist, tp1_done=False, trade_pnl=0.0)
    m["trades_today"] += 1
    log(f"{sym}: ENTER {side_name(side)} @ {entry:,.2f}  stop {stop:,.2f}  target {m['tp']:,.2f}  "
        f"size {frac:.2f}x (paper, Rs {m['equity']:,.0f})")


def process(sym, m, d, now):
    """Replay every closed candle after m['last_bar'].
    Stops/targets are checked inside each candle; close-based exits and entries fill at the next candle's open."""
    closed = d[d.time + TF_SEC <= now].reset_index(drop=True)
    if len(closed) < 100:
        raise RuntimeError(f"{sym}: not enough history ({len(closed)} closed candles)")
    if m["last_bar"] is None:
        m["last_bar"] = int(closed.time.iloc[-1])
        log(f"{sym}: paper trading starts after candle {ist(m['last_bar'])} IST")
        return
    todo = d.index[(d.time > m["last_bar"]) & (d.time + TF_SEC <= now)]
    for i in todo:
        if i + 1 >= len(d):
            break
        row, nxt = d.iloc[i], d.iloc[i + 1]
        t_bar = int(row.time)
        day = utc_day(t_bar)
        if m["day"] != day:
            m.update(day=day, trades_today=0, losses_today=0)

        # 1) stop / partial target inside this candle
        if m["pos"]:
            s = m["pos"]
            hit_sl = (row.low <= m["sl"]) if s == 1 else (row.high >= m["sl"])
            hit_tp = (not m["tp1_done"]) and ((row.high >= m["tp"]) if s == 1 else (row.low <= m["tp"]))
            if hit_sl:
                gap = (row.open < m["sl"]) if s == 1 else (row.open > m["sl"])
                exit_part(sym, m, float(row.open) if gap else m["sl"], t_bar, 1.0,
                          "breakeven stop" if m["tp1_done"] else "stop loss")
            elif hit_tp:
                exit_part(sym, m, m["tp"], t_bar, PARTIAL, f"{RR:g}R target (50%)")
                m["tp1_done"] = True
                m["sl"] = m["entry"]

        # 2) close-based exits -> next open
        fill, t_fill = float(nxt.open), int(nxt.time)
        if m["pos"] == 1:
            if row.close < row.ema9:   exit_part(sym, m, fill, t_fill, 1.0, "close below EMA9")
            elif row.close < row.vwap: exit_part(sym, m, fill, t_fill, 1.0, "close below VWAP")
        elif m["pos"] == -1:
            if row.close > row.ema9:   exit_part(sym, m, fill, t_fill, 1.0, "close above EMA9")
            elif row.close > row.vwap: exit_part(sym, m, fill, t_fill, 1.0, "close above VWAP")

        # 3) new entry -> next open
        can = m["pos"] == 0 and m["trades_today"] < MAX_TRADES and m["losses_today"] < MAX_LOSSES
        if can and row.long_sig:
            open_trade(sym, m, 1, fill, float(row.stop_long), t_fill)
        elif can and row.short_sig:
            open_trade(sym, m, -1, fill, float(row.stop_short), t_fill)

        m["last_bar"] = t_bar


def mark(m, price):
    if not m["pos"]:
        return m["equity"]
    px = price * (1 - SLIP * m["pos"])
    return m["equity"] + m["notional"] * (m["pos"] * (px - m["entry"]) / m["entry"] - 2 * FEE)


def run_once(st):
    now = int(time.time())
    total, lines = 0.0, []
    for sym in SYMBOLS:
        m = st["markets"].setdefault(sym, new_market())
        d = prepare(fetch(sym, TF, TF_SEC, FETCH_BARS))
        process(sym, m, d, now)
        last_closed = d[d.time + TF_SEC <= now].iloc[-1]
        last = float(d.close.iloc[-1])
        eq = mark(m, last)
        total += eq
        bias = "UP" if (last_closed.close > last_closed.vwap and last_closed.ema9 > last_closed.vwap) else \
               "DOWN" if (last_closed.close < last_closed.vwap and last_closed.ema9 < last_closed.vwap) else "MIXED"
        pos = {1: "LONG", -1: "SHORT", 0: "no position"}[m["pos"]]
        lines.append(f"{sym}: price {last:,.1f} | EMA9 {last_closed.ema9:,.1f} | VWAP {last_closed.vwap:,.1f} | bias {bias} | {pos}"
                     + (f" from {m['entry']:,.1f}, stop {m['sl']:,.1f}" + ("" if m["tp1_done"] else f", target {m['tp']:,.1f}")
                        if m["pos"] else "")
                     + f" | equity Rs {eq:,.0f} ({100 * (eq / CAPITAL - 1):+.2f}%) | trades {m['trades']}, wins {m['wins']}"
                     + f" | today {m['trades_today']}/{MAX_TRADES} trades, {m['losses_today']} losses")
        append_csv(EQUITY_F, [dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M"), sym, round(last, 2), pos, round(eq, 2)],
                   ["time_IST", "market", "price", "position", "equity_rs"])
    save_state(st)
    write_status(st, total, lines)
    log(" || ".join(lines))
    cap = CAPITAL * len(SYMBOLS)
    log(f"TOTAL paper equity Rs {total:,.0f} on Rs {cap:,} ({100 * (total / cap - 1):+.2f}%)")


def write_status(st, total, lines):
    cap = CAPITAL * len(SYMBOLS)
    out = ["# Paper Bot 3 — 9 EMA + VWAP Pullback (5m)", "",
           "Compare with bot 1: [../STATUS.md](../STATUS.md) | bot 2: [../rsi_bot/STATUS.md](../rsi_bot/STATUS.md)", "",
           f"**Last update:** {dt.datetime.now(IST).strftime('%d %b %Y, %H:%M')} IST  ",
           f"**Paper trading since:** {st['started']} IST  ",
           f"**Total paper equity:** Rs {total:,.0f} on Rs {cap:,} (**{100 * (total / cap - 1):+.2f}%**)", "",
           "## Markets", ""]
    out += [f"- {l}" for l in lines]
    out += ["", "## Last 15 exits", ""]
    if os.path.exists(TRADES_F):
        t = pd.read_csv(TRADES_F).tail(15).iloc[::-1]
        out.append("| Market | Side | Entry (IST) | Entry | Exit (IST) | Exit | Portion | P&L Rs | Reason |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in t.iterrows():
            out.append(f"| {r.market} | {r.side} | {r.entry_time_IST} | {r.entry_price:,.2f} | {r.exit_time_IST} | "
                       f"{r.exit_price:,.2f} | {r.portion} | {r.pnl_rs:+,.0f} | {r.exit_reason} |")
    else:
        out.append("Abhi tak koi trade band nahi hua.")
    out += ["", "_Paper trading only — no real orders. Not financial advice._", ""]
    with open(os.path.join(HERE, "STATUS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def status(st):
    for sym, m in st["markets"].items():
        print(f"{sym}: equity Rs {m['equity']:,.0f}, position {m['pos']}, entry {m['entry']}, stop {m['sl']}, "
              f"trades {m['trades']}, wins {m['wins']}")


def main():
    st = load_state()
    if "--status" in sys.argv:
        status(st); return
    log(f"Paper bot 3 (9EMA+VWAP, 5m) started (paper since {st['started']} IST). Markets: {', '.join(SYMBOLS)}. NO real orders.")
    while True:
        try:
            run_once(st)
        except Exception as e:
            log(f"ERROR: {e}  (will retry next cycle)")
        if "--once" in sys.argv:
            break
        now = time.time()
        wake = (int(now // TF_SEC) + 1) * TF_SEC + 20
        log(f"Next check at {ist(wake)} IST")
        time.sleep(max(5, wake - now))


if __name__ == "__main__":
    main()

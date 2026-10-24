# Paper Trading Bot — BTC + ETH (Delta Exchange India, 30m)

**Paper trading only. No real orders, no API keys, no real money.**

- **Strategy:** EMA 20/50/200 crossover + Volume confirmation + 4H trend filter
- **Markets:** BTCUSD and ETHUSD perpetual (Delta Exchange India public price data)
- **Virtual capital:** Rs 50,000 per market
- **Runs:** automatically every 30 minutes on GitHub Actions

## Kahan dekhein
- **[STATUS.md](STATUS.md)** — abhi ki position, total P&L, last 15 trades
- `trades.csv` — har band hua trade
- `equity.csv` — har 30 minute ka hisaab
- `bot.log` — bot ne kya kiya

## Rules
- **BUY:** EMA20 crosses above EMA50, close > EMA200, volume > 20-candle average, 4H trend UP
- **SELL:** mirror rules, 4H trend DOWN
- **EXIT:** opposite EMA20/50 cross (an opposite signal reverses the trade)
- **4H trend:** last closed 4H candle — UP if EMA9 > EMA100 and close > EMA200, DOWN if the reverse
- Fee 0.05% + slippage 0.02% per side included

Not financial advice. Past results do not guarantee future results.

# Bottom up corn futures pricing

The idea is simple to state and hard to execute: if you can estimate the physical state of the corn market more accurately than the market has already priced in, there should be tradable gaps between futures prices and model implied fair value.

Corn is not one homogeneous good. A July contract and a December contract embed different economic exposures. July is mostly about old crop inventories and nearby tightness. December is about new crop acreage, yield expectations, and harvest pressure. Any serious model has to respect that term structure instead of treating all contracts as the same bet.

## How the pipeline is structured

Three layers:

1. **State estimation.** Build a daily panel of supply, demand, stocks, exports, ethanol use, positioning, weather, and seasonality from official sources (NASS, WASDE, export sales, CFTC, EIA, market prices).
2. **Fair value mapping.** Translate that state into contract specific fair values, because the same bullish inventory shock does not move every maturity equally.
3. **Trading rule.** Enter when market price is far enough from fair value to cover model error, execution cost, and risk.

The notebook `corn_futures_factor_frramework.ipynb` walks through the full research stack: raw file ingestion, canonical cleaning, forecast modules, balance sheet reconciliation, factor construction, and export of a strategy ready dataset. The QuantConnect script `Quantconnect_strategy_backtest.py` runs the live logic against CBOT corn futures using the processed state panel.

## What I took away

Most of the difficulty is not the regression or the optimizer. It is data alignment. Official releases arrive on different calendars, revisions rewrite history, and "the same" fundamental series can mean different things depending on whether you are looking at old crop or new crop. The project forced me to be explicit about state definitions before thinking about alpha.

I wrote this up properly on the blog: [Bottom up corn pricing](https://ahmadkoman.github.io/posts/bottom-up-corn/)

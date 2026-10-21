# agentic_trading

A small, dependency-free (stdlib `asyncio`) framework for a multi-agent trading system.
Agents never call each other. They publish typed messages to a bus, and share information such
as earnings announcements and market events with whoever subscribes.

**Paper trading only.** There is no live broker integration, by design. Do not connect one without
independent review of the risk layer.

## Layout

| Module | Role |
| --- | --- |
| `protocol.py` | Message envelope, topics, performatives, priorities and typed payloads |
| `bus.py` | Pub/sub bus: wildcard topics, directed messages, priority inboxes, dead letters |
| `agent.py` | `BaseAgent`: inbox loop, routing, periodic `tick`, heartbeat, error isolation |
| `agents/market_data.py` | Quotes, price and volume event detection, earnings announcements |
| `agents/geospatial.py` | Geospatial signals from pluggable sources, including the God's Eye adapter |
| `agents/risk.py` | Pre-trade limits, event-driven size tightening, daily-loss circuit breaker |
| `agents/strategy.py` | Fuses earnings, geo and market evidence into trade proposals |
| `agents/execution.py` | Paper broker; executes only risk-approved proposals |
| `manager.py` | `AgentManager`: lifecycle, watchdog restarts, global halt/resume |

## How agents share information

```text
market_data --events.earnings--> geo, strategy, risk      (all react independently)
geo         --intel.geo.signal-> strategy, risk           (correlated to the earnings event)
strategy    --intel.geo.request-> geo                     (asks for corroboration when borderline)
strategy    --trade.proposal---> risk, execution
risk        --trade.decision---> execution, strategy      (approve, shrink or reject)
execution   --trade.fill-------> risk, strategy
risk        --risk.alert-------> manager                  (critical alert halts everything)
```

Every message carries a `correlation_id`, so one earnings announcement can be traced through the
geo lookup, the proposal, the risk decision and the fill. Run `python examples/run_demo.py` to
see it.

## Safety properties

* Nothing executes without a `trade.decision` from a trusted approver (`risk` by default).
  Execution also checks that it saw the proposal, caps the size at what was proposed, and
  executes each proposal once.
* Risk shrinks orders to fit limits and rejects what cannot fit. Earnings, large price moves and
  strongly negative geo signals temporarily tighten limits for the affected ticker.
* A daily-loss breach trips the circuit breaker (only risk-reducing orders pass). The manager
  turns that critical alert into a system-wide halt.
* A failing handler is logged and counted. It never takes down its agent. The manager restarts
  agents whose tasks die or whose heartbeat goes stale, and halts the system if one cannot
  be recovered.

## Connecting God's Eye

The God's Eye repository was not accessible when this was written, so `GodsEyeSource` is an
adapter seam and not a finished integration. To wire it up, provide an async function that calls
into God's Eye and returns records for the requested tickers:

```python
from agentic_trading.agents import GeospatialAgent, GodsEyeSource

async def fetch_from_godseye(symbols):
    ...  # call God's Eye, return an iterable of dict records

source = GodsEyeSource(fetch_from_godseye)  # or GodsEyeSource(fetch_from_godseye, my_mapper)
geo = GeospatialAgent(bus, source, watchlist=["XOM", "WMT"])
```

`default_godseye_mapper` assumes each record has `id`, `region`, `lat`, `lon`, `category`,
`value`, `baseline` and `symbols` (optional: `bullish_when_higher`, `confidence`, `summary`,
`observed_at`). If God's Eye returns a different shape, pass your own mapper that returns a
`GeoObservation`. Nothing else in the framework needs to change.

## Extending

* **Real market data:** implement `MarketDataProvider` (`get_quotes`, `poll_earnings`).
* **New agent:** subclass `BaseAgent`, call `self.route(pattern, handler)` and `self.publish(...)`.
* **Register and run:** `manager.register(agent)`, then `await manager.start_all()`.
* **Another transport** (Redis, NATS): replace `MessageBus`. Agents depend only on `publish`,
  `register`, `subscribe`, `unregister`, `ack` and `dead_letter`.

## Tests

```bash
python -m unittest discover -s tests
```

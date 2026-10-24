# Hurricane fair-value trading bot

Competition research code and frozen model artifacts. **The live adapter has known safety defects and is not ready for unattended trading.** See [independent verification](verification/audit.txt).

## Quick start

Use Python 3.11 or newer and install dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python replay.py --data /absolute/path/to/hurricane-season-research --seasons 451-600 --out replay-local.json
HS_DATA=/absolute/path/to/hurricane-season-research python tests/test_engine.py
```

The research archive is not included. Supply your own competition archive containing `case.json` and `history/`.
The saved models report scikit-learn 1.9.1; dependency installation in a fresh environment has not been verified. The independent replay used 1.6.1 with compatibility and numerical warnings, while reproducing the saved P&Ls.

## Verified results and limitations

On seasons 451–600, the independent replay reproduced mean paper profit of $4,777.54, median $4,344.36, worst season -$2,719.32, and 144 profitable seasons out of 150. Assumptions: full fills at house marks, four-second delay, and $0.10 per share each way. This is not a live execution result or a pristine research holdout.

Known live defects: the news-driven clock misses the scheduled exit in all 450 archived test rounds; announcement flatten instructions are ignored by the adapter. Repair and test these before sending orders. Never commit API keys; use the `BOT_KEY` environment variable. No credentials are included here.

---

## Original research documentation

# Fair-value storm strategy (Hurricane Season / Systematic Equities)

Historical paper research only. Nothing here has placed a live order. Fills are hypothetical at the house
consensus mark ± a per-share cost; the archive has no order book, so depth, queue position and quote
skew are not modelled. Treat every dollar figure as "if we could trade at the mark".

## The idea in one paragraph

Every stock settles at a formula. It depends only on three things: how much damage each island company has
taken, how long its sites were closed, and the season's insured loss. We recovered that formula from the 600
past seasons (residual under $0.04 a share), along with the storm physics (how wind falls off with distance,
and how much damage it does). During a storm we can therefore work out **what each stock will be worth**
from the storm's live position, the official cone and what has already happened. The house price is a good
estimate, but it is slow and systematically off in the middle of a storm. By the time the storm dies, the
house has moved almost all the way to our number (measured slope ≈ 1.0). So the strategy has three steps:
work out the value, take positions toward it at fixed checkpoints, and close everything 10 seconds after the
storm dissipates. All the rules are fixed before the round starts. When a new observation arrives, it only
changes the inputs.

## What was recovered from the data (exact, see `research/value.py`, `research/damage.py`)

| Stock | Settlement value |
|---|---|
| PORT | 100 − 0.55·D_PORT − 0.02·closed_sec |
| ISLE | 60 − 0.38·D_ISLE |
| AQUA | 35 − 0.20·D_AQUA |
| GRID | 50 − 0.24·D_GRID − 0.05·closed_sec |
| FERRY | 25 − 0.05·D_FERRY − 0.06·closed_sec |
| RECON | 40 + 24·(1 − e^−(0.009·D_PORT + 0.007·D_GRID + 0.0012·D_FERRY)) |
| MAT | 28 + 14·(1 − e^−(0.0075·D_P + 0.005·D_I + 0.0021·D_A + 0.0058·D_G + 0.001·D_F)) |
| GEN | 32 + 18·(1 − e^−0.0082·GRID closed_sec) |
| HAVEN | 75 − 0.06·clip(L − 400, 0, 900) (layer $400M–$1,300M) |
| REIN | 90 − 0.035·clip(L − 1300, 0, 700) (layer starts at $1,300M) |

Here D is the cumulative season damage % (the final damage reports), and closed_sec is the site-weighted
number of seconds that the wind was above the hurricane threshold. That wind is
intensity·exp(−½(d/0.70R)²) > 0.34. The insured loss of a storm is L = 9·D_PORT + 6·D_ISLE + 2.5·D_AQUA
+ 7·D_GRID + 1.2·D_FERRY. Damage takes a fraction of the *remaining* value of a site, so it compounds.
The preliminary report is the final value × (1 ± ~33% noise). The formulas are identical when refitted on
seasons 1–450 only.

## The playbook (what the code does)

**Strategy A — Storm fair-value convergence (on by default).**

1. *Before the round.* Load the season state from earlier final reports (site health, damage, closures,
   loss). Load the frozen tables: for each checkpoint and stock, `beta` (the share of today's gap that
   historically closes by the exit) and the typical error.
2. *Checkpoints at 150, 190, …, 390 s after the open* (storm age 90–330 s). Fair value = expected settlement.
   It combines this storm's predicted final damage and closures (a gradient-boosted nowcast built on the
   physics: damage so far, constant-velocity and official-cone projections, advisory odds) with a
   Monte Carlo over future storms drawn from 1,350 past storms.
3. *Rule.* expected move = beta × (fair value − house price). If |expected move| > 2·cost + $0.25, add a
   tranche of up to 1/7 of the position limit, scaled by expected move ÷ typical error. `Playbook.ladder()`
   prints this as a price table before prices arrive: buy at or below X, sell at or above Y, with the size
   at each price. The live adapter uses those prices as limit prices, so an order never pays more than the
   edge the rule demands.
4. *Risk cap.* The model's one-standard-deviation risk for the whole book is kept at or below $1,500. New
   tranches are shrunk to fit.
5. *Exit.* Everything is closed at 550 s (10 s after the storm dissipates), before either damage report.
   This leaves no exposure to report noise and no exposure across rounds.
6. *Safety rules (pre-committed).*
   - **Gap widening:** a stock whose gap widens against the first signal by more than 3 typical gaps gets no
     more tranches this round.
   - **Model breach:** a stock whose fair value still disagrees with the house by more than 6 typical gaps
     after the storm is presumed mis-modelled and is disabled for the rest of the season.
   - **Announcement:** an `announcement` naming a stock disables it at once and closes the position.
   - **Stale data:** stale storm data at a checkpoint means no action.

**Strategy D — Round-3 settlement snap (optional, off).** When the round-3 final report arrives, settlement
is known exactly. Post limit orders at value ∓ (cost + $0.10). These fill only if the house quote is still
stale. They earn $898/season at 2 s latency and $0 at 4–6 s, and never lose in the replay. Enable it only if
the measured order round trip is under 2 s.

**Strategy C — ETF lag (optional, off).** ARCH and REBLD house marks equal their basket value from 6 s
earlier (residual sd $0.48). Trading the ETF toward its basket earns $10.6k/season at 2 s latency, goes
negative at 6 s, and depends on the real ETF book behaving like the marks. **Verify it on the testnet before
using it.**

Optional sleeve not enabled: holding fundamental value across rounds ("carry") had a small positive edge
(+$200–300/season) with $1.2–2k standard deviation. Mean-variance hedging of that carry lost money on
validation. Both are documented in `research/carry.py`.

## Results (simulated $ per season, hypothetical fills at mark ± cost)

Protocol: models were cross-fitted on seasons 1–450. Settings were chosen on 1–450 using out-of-fold
predictions only. The frozen configuration was then run on held-out seasons 451–600.

| Strategy A on test 451–600 | Mean | Median | Worst | 5th pct | Winning seasons |
|---|---:|---:|---:|---:|---:|
| Frozen (4 s delay, $0.10/side) | **4,778** | 4,344 | −2,719 | 143 | 96% |
| Delay 2 s / 10 s / 20 s | 4,897 / 4,672 / 4,466 | | −2,611 / −2,678 / −3,423 | | 97 / 96 / 97% |
| Cost $0.25 / $0.50 per share per side | 3,259 / 2,043 | | −2,010 / −2,386 | | 90 / 84% |
| Half size | 2,403 | 2,168 | −1,349 | 67 | 96% |
| Long trades only / short trades only | 983 / 3,812 | | | | 89 / 97% |

The streaming engine (`replay.py`, message by message) reproduces the batch result exactly ($4,777.54 vs
$4,777.5). PnL comes from every stock (test means: HAVEN 1,429; PORT 771; GRID 579; FERRY 439; RECON 416;
ISLE 337; GEN 322; AQUA 286; MAT 176; REIN 24). About 70% of tranches are shorts, but long trades alone are
also profitable, so the result is not just a "prices drift down" effect.

**Same risk box as the previous candidate.** With 10 shares per entry, at most 2 open stocks, a $3,000 gross
cap, and the same seasons and costs: $63/season and 77% winning seasons, from about 6 entries/season. That is
$1.05 per share after costs. The selective adaptive candidate made $75/season, 65% winning, from about 13
entries/season, about $0.58 per share. So the edge per share is about 1.8× larger, and the rest of the dollar
gain comes from using more of the position limits. That extra size is real risk: the worst test season is
−$2,719.

**Stress tests (all planned scenarios reported; test seasons, safety rules on):**

| Scenario | Mean | Worst | Winning seasons |
|---|---:|---:|---:|
| None | 4,778 | −2,719 | 96% |
| All prices drift **up** $0.02/s after 400 s | 876 | −5,960 | 60% |
| All prices drift down $0.02/s after 400 s | 6,945 | −2,211 | 98% |
| All stocks +3% jump at 300 s | 3,301 | −3,161 | 87% |
| All stocks −3% jump at 300 s | 4,282 | −3,120 | 95% |
| Insurers drift −$0.03/s after 300 s | 6,055 | −3,139 | 97% |
| PORT economics change from round 2 (not modelled) | 4,711 | −2,610 | 95% |
| Quote noise 0.3% | 4,769 | −2,181 | 95% |
| 30% of checkpoints stale or rejected | 3,396 | −2,755 | 95% |
| Adverse fills (we pay 20% of the gap) | 3,279 | −3,699 | 87% |
| Cost $0.25 + 10 s delay | 3,182 | −2,523 | 90% |

The weak spot is an unexplained market-wide rally after the entries. The book leans short, so it loses
in roughly 40% of those seasons. The model-breach rule limits the damage in later rounds. An in-round
mark-to-market stop was tested and did not help, so it was not adopted.

**What did not work (kept for honesty):**

- Rebalancing every 10 s. Costs and noise more than halved the risk-adjusted return.
- Full inverse-covariance (mean-variance) sizing. It flipped hedges and lost heavily to costs.
- De-risking when the storm goes "off-script". Our fair value already follows the observed track, so cutting
  positions just locked in losses.
- Trading after the storm. The house is better than us once the damage reports are out.

## Run it

```sh
pip install numpy scipy scikit-learn            # sklearn version should match model/ (1.9.x) or re-run train.py
python train.py  --data ~/Downloads/hurricane-season-research    # rebuilds model/ from seasons 1-450 (~5 min)
python replay.py --data ~/Downloads/hurricane-season-research --seasons 451-600            # Strategy A
python replay.py --data ~/Downloads/hurricane-season-research --seasons 451-600 --delay 2 --snap --etf
HS_DATA=~/Downloads/hurricane-season-research python tests/test_engine.py                  # causality tests
python bot_adapter.py --url https://<exchange> --key $BOT_KEY --dry-run                    # testnet, dry run
```

`tests/test_engine.py` checks five things:

- **Future mutation:** corrupting every later storm observation, advisory, report and price leaves all
  earlier fair values identical.
- **Duplicate messages** are ignored.
- **Dropped observations:** with 25% of storm observations dropped, fair values move by less than $1.
- **Announcements** disable the named stock.
- **Season isolation:** a result does not depend on which seasons ran before it.

All five pass.

## Files

- `engine.py` — the streaming fair-value engine and playbook. It keeps bounded state and is used both in
  replay and live.
- `replay.py` — the message-by-message historical replay.
- `bot_adapter.py` — the live WebSocket/REST adapter. **Not tested against the exchange.** It drives its
  clock from feed timestamps; polling `/competitions/time` would be more robust.
- `train.py` — rebuilds the frozen artifacts in `model/` (damage parameters, nowcast models, storm library,
  calibration).
- `research/` — every study script: loader, physics fit, value formulas, features, nowcast, cross-fitted
  fair values, backtests, stresses, ETF, carry, the matched-size comparison and the off-script test.
- `results/` — JSON/CSV outputs of the runs quoted above.

## What remains unproven

- Live execution is unproven: spreads, depth, how the house market makers skew quotes when we trade size,
  other teams, partial fills and cancellations. The position sizes here are large; impact could take a
  big share of the edge. Start at a fraction of size on the testnet and measure the real cost per share.
- Announcements do not exist in the archive. The rule only protects us (disable and flatten); it does not
  try to profit from them.
- Seasons 451–600 had been looked at in earlier studies, so they are not pristine. One change was made
  after the test run: the exit moved from 540 s to 550 s. The streaming parity check showed that the earlier
  timing relied on a 4-second house lag. The change was made for robustness; the test mean moved from 4,773
  to 4,778.
- The live season is the most recent one, just after season 1. The result is stable across the archive
  (out-of-fold or test means: seasons 1–100 $4,328; 101–200 $4,739; 201–300 $4,030; 301–400 $4,087;
  401–500 $4,465; 501–600 $4,740). Its correlation with the season number is 0.02. Still, that the live
  simulator matches the archive is an assumption.

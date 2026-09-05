# AI Risk Manager — Track 02

A fraud scorer whose threshold is chosen in rupees rather than F1, calibrated so
the probabilities mean what they say, evaluated on a temporal entity-disjoint
split against a rule baseline, with the false-positive cost of blocking a real
customer priced in — including the churn nobody counts, and the labels we stop
receiving the moment we start blocking.

## Setup

```bash
cd track2
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Data: IEEE-CIS `train_transaction.csv` / `train_identity.csv` symlinked into
`data/raw/`. Only the labelled `train_*` files are used — Kaggle's
`test_transaction.csv` has no `isFraud` and is not part of this pipeline; our
own temporal/entity-disjoint train/calib/test split is built from
`train_transaction.csv` alone.

## Run order

```bash
python -m src.data.split            # Phase 1 — split + entity-disjointness report
python -m eval.test_leakage         # Phase 2 — as_of leakage test (must pass)
python -m src.features.build_matrices  # Phase 2 — build train/calib/test feature matrices
python -m eval.test_policy          # Phase 5 — deterministic policy assertions
python -m src.model.train           # Phase 4 — LightGBM + isotonic calibration
python -m eval.baselines            # Phase 3 — rule baselines vs cost function
python -m src.policy.run_phase5     # Phase 5 — sweep, sensitivity, case samples
python -m src.audit.seed            # Phase 6 — seed audit chain from real decisions
python -m src.audit.chain           # Phase 6 — tamper-detection demo
python -m eval.test_analyst         # Phase 8 — refusal + grounding checkpoint
streamlit run app/app.py            # Phase 7 — dashboard
```

`eval/run_all.py` runs the leakage/policy/analyst tests together.

## Repository layout

See `TRACK02_BUILD_PROMPT.md` for the full spec this was built against.

```
src/data/       entity-disjoint temporal split (Phase 1)
src/features/   as_of-safe feature pipeline + leakage-tested batch builder (Phase 2)
src/model/      LightGBM, isotonic calibration, attributions (Phase 4)
src/policy/     cost model, net-rupee function, threshold sweep — NO ML, NO LLM (Phases 3, 5, 10)
src/audit/      hash-chained decision log (Phase 6)
src/monitor/    spike-detection heuristic (Phase 7)
src/serving/    customer/analyst view boundary (Phase 7)
src/analyst/    extractive LLM narrative layer + refusal path (Phase 8)
src/synthetic/  attack injector — demo only, never in reported metrics (Phase 9)
eval/           leakage test, policy tests, baselines, analyst test
app/            Streamlit dashboard (Phase 7)
```

## Known deviations from the brief

- **Python 3.10, not 3.11** — 3.11 isn't installed on this machine and 3.10 is
  a negligible difference for this stack.
- **Entity-disjointness key**: raw `card1..card6` is a BIN/network/type
  signature shared by thousands of physical cards, not a unique-card id.
  Enforcing disjointness on it collapsed calibration/test to a few hundred
  rows — under the positives-in-the-thousands floor the brief itself calls
  for. `card_entity` is instead `card1 + addr1 + D1n` (D1n = an approximate
  account-open day), the proxy the IEEE-CIS/Kaggle community converged on.
  See `src/data/entities.py` for the measured comparison.
- **`DISPUTE_FEE` (Rs 400)** is an estimate at the midpoint of the Rs 200-600
  dispute-fee range Razorpay's own docs cite
  (razorpay.com/docs/payments/disputes/), not a confirmed negotiated rate —
  flagged in `src/policy/costs.py`.
- **Rule baseline 3** (`failed_attempts_10min >= 5`) needed a decline/attempt
  counter IEEE-CIS doesn't carry (no acquirer response codes in this
  dataset). Substituted with `velocity_10min` (all attempts, successful or
  not, in a 10-minute window) — the nearest available signal, documented in
  `eval/baselines.py` rather than silently faked.

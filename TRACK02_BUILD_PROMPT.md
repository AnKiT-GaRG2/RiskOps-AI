# Build Prompt — AI Risk Manager (Track 02)

Paste this whole file into Claude Code (or your coding agent of choice) as the opening message. Work through it phase by phase; do not skip ahead.

---

## Your role

You are building a hackathon project for Razorpay Track 02 (AI Risk Manager). The brief's bar is: **"Honest metrics including false-positive cost. Strictly defense-only: anything offense-capable is disqualified."**

That bar is unusual — it rewards a team for reporting a *worse* number than the team next to them. Almost every submission will show AUC or F1 on a random split and call it done. We are building the version that reports what a false positive actually costs, chooses its thresholds in rupees, and admits what it cannot measure.

Build in the phase order given. After each phase there is a checkpoint. Stop, run it, confirm it passes before continuing.

---

## Stack (fixed — do not substitute)

- Python 3.11
- LightGBM — gradient boosting on tabular data
- scikit-learn — `IsotonicRegression` for calibration, metrics
- pandas + numpy
- Streamlit — dashboard (charts are the deliverable; do not build a JS frontend)
- plotly or matplotlib
- SQLite — decision log and audit chain

Do not add PyTorch, TensorFlow, or any deep learning library. Gradient boosting wins on tabular fraud data and choosing it deliberately is a signal of judgment.

---

## Seven non-negotiables

These are the project. If a phase seems to require breaking one, you have misread the phase — stop and flag it.

1. **Temporal split, never random.** Train on the earlier time window, test on the later. A random split leaks the future through shared cards and devices.

2. **Entity-disjoint holdout.** Whole cards and devices go to one side of the split or the other, never both.

3. **Every behavioural feature computed with a strict `as_of` cutoff.** No feature may use data from at or after the transaction's own timestamp. This bug never throws an error and inflates results dramatically.

4. **Thresholds chosen by sweeping net rupees, never F1.** F1 weights false positives and false negatives equally, which is a claim about the business that is essentially never true.

5. **Calibration before any expected-value calculation.** A raw model score is a ranking, not a probability. The policy layer does EV math and needs `P(fraud)` to mean what it says.

6. **The LLM never scores and never decides.** It writes prose for an internal analyst from extracted evidence, after the decision is made.

7. **Feature attributions never reach a customer surface.** They are an evasion roadmap and the track disqualifies offense-capable work.

---

## Repository layout

```
/data
  /raw               IEEE-CIS download (gitignored)
  /processed         split artifacts, feature matrices
/src
  /features          feature pipeline — as_of enforcement lives here
  /model             LightGBM training, isotonic calibration
  /policy            thresholds, cost model, sweep — NO ML, NO LLM
  /audit             hash-chained decision log
  /analyst           LLM explanation layer + refusal path
  /synthetic         attack injector, demo only
/eval
  test_leakage.py    the as_of regression test
  test_policy.py     deterministic policy assertions
  baselines.py       three hand-written rules
/app                 Streamlit dashboard
/notebooks           exploration only, nothing imports from here
```

---

# PHASE 1 — Data and splits (target: 3h)

**Download IEEE-CIS Fraud Detection from Kaggle.** Its `isFraud` label is chargeback-reported fraud, which is the actual problem. Roughly 590k transactions, low-single-digit base rate, ~430 features across transaction and identity tables. Verify the current shape before building around it.

Do not generate synthetic data for the modelling half. If you hand-code fraud patterns and then engineer features that detect those patterns, the model reverse-engineers your generator and reports near-perfect precision. The tell a judge will find: three hand-written rules perform almost as well, because the rules *are* the data.

**Three-way split, in this order:**

1. Sort by `TransactionDT`
2. Cut at ~60% / ~15% / ~25% → train / calibration / test
3. Then enforce entity-disjointness: any card or device appearing in more than one slice gets assigned to the earliest slice only, and its later rows are dropped

The calibration slice must be untouched by training. Fitting isotonic regression on training data produces overconfident probabilities and quietly invalidates every downstream rupee figure.

Log and display: rows per slice, positive count per slice, base rate per slice, date boundaries, rows dropped to entity-disjointness.

**Sample size floor, worth internalizing:** at a ~2% base rate a 200-row test set holds 3–4 positives, and precision on 3 positives has an error bar wide enough to swallow any claim. You need positives in the thousands. A 25% test slice on this dataset gives you roughly that.

**Checkpoint:** print the split table. Confirm zero card or device overlap across slices. Confirm test dates are strictly later than train dates.

---

# PHASE 2 — Feature pipeline (target: 4h)

**Static features:** amount, hour-of-day, day-of-week, card BIN attributes, product code, email domain, device and browser fields, address match indicators.

**Behavioural features:** velocity over 1h/24h/7d, distinct cards per device, distinct devices per card, amount z-score against the customer's own history, time since previous transaction, historical decline rate.

**Enforce the cutoff structurally, not by care.** Write exactly one entry point:

```python
def compute_features(txn, history_df, as_of: datetime) -> dict:
    # history_df MUST be filtered to as_of before any aggregation
```

Every call site passes `txn.timestamp` as `as_of`. No other function may compute a behavioural aggregate.

**Then write the leakage test.** This is the most important test in the project:

```python
# eval/test_leakage.py
# Compute features for a known transaction twice:
#   (a) with the full dataset available
#   (b) with every row at or after txn.timestamp deleted
# Assert the two outputs are identical.
```

If they differ, a feature is reading the future. Fix it before proceeding — every metric downstream is meaningless until this passes.

**Checkpoint:** leakage test green. Feature matrix built for all three slices.

---

# PHASE 3 — Cost model, baselines, evaluation harness (target: 3h)

Build the thing that measures before the thing being measured, or you will tune against a moving target.

### Cost model

Every parameter is a named constant in one config file, sourced or clearly flagged as an estimate:

```python
# Replace every figure with your own sourced number and cite it
AVG_FRAUD_AMOUNT     = 4500
DISPUTE_FEE          = ____   # Razorpay's current published pricing
FN_COST              = AVG_FRAUD_AMOUNT + DISPUTE_FEE + HANDLING

AVG_LEGIT_MARGIN     = 240    # 12% of a ₹2,000 order
SUPPORT_CONTACT      = 150
P_CHURN_WRONG_BLOCK  = 0.05
CUSTOMER_LTV         = 8000
FP_COST              = AVG_LEGIT_MARGIN + SUPPORT_CONTACT + (P_CHURN_WRONG_BLOCK * CUSTOMER_LTV)

P_ABANDON_STEPUP     = 0.09
STEPUP_COST          = P_ABANDON_STEPUP * (AVG_LEGIT_MARGIN + P_CHURN_WRONG_BLOCK * CUSTOMER_LTV)
REVIEW_COST          = 60     # ~4 analyst-minutes fully loaded
RETRY_RATE           = 0.40   # blocked fraudsters come back on another card
```

Three things this captures that nobody else's will:

- **Churn dominates FP cost.** A wrongly blocked customer is not a fee-sized loss. Pricing it at a few hundred rupees is roughly an order of magnitude low.
- **Step-up is not free.** It is a partial decline — legitimate customers abandon OTP and 3DS at meaningful rates. Price the middle band at zero and the policy will over-use it.
- **Prevented ≠ detected.** Discount prevented loss by `RETRY_RATE` and state the assumption. Smaller headline number, far more credible.

The FN:FP ratio that falls out will be somewhere near 7:1. That asymmetry is why the optimal threshold is not 0.5, and it is the finding the whole demo is built around.

### Net-rupee function

```python
def net_rupees(y_true, scores, t_low, t_high, costs) -> dict:
    # returns net, and the breakdown: fn_loss, fp_loss,
    # stepup_loss, review_loss, prevented (post-retry-discount),
    # plus counts and review_rate
```

### Rule baselines

```
velocity_1h > 5 × customer_median  AND  distinct_cards_per_device >= 3
amount_zscore > 4
failed_attempts_10min >= 5
```

Evaluate on the same test set through the same cost function. Record precision, recall, net rupees.

Report the baseline on the same slide as the model. If LightGBM does not beat it by a clear margin, you have learned something important before the judges do.

**Checkpoint:** baselines produce a net-rupee figure. Cost breakdown prints cleanly.

---

# PHASE 4 — Model and calibration (target: 4h)

**LightGBM.** Handle imbalance with `scale_pos_weight`, not SMOTE — synthetic minority oversampling on entity-linked fraud data fabricates transactions that never existed and interacts badly with the temporal split.

**Report PR-AUC as the primary ranking metric.** At a low base rate, ROC-AUC is flattered by the enormous true-negative mass. Report ROC-AUC as well since people expect it, but lead with PR-AUC and explain why in one sentence.

**Calibration:**

1. Fit `IsotonicRegression` on the calibration slice — never on train
2. Produce a **reliability diagram**: predicted probability (x) vs observed frequency (y), 10 bins, diagonal reference
3. Report Brier score and expected calibration error, before and after

The line to say in the demo: *if the model says 0.7 and those transactions turn out fraudulent 40% of the time, every downstream rupee calculation is wrong regardless of how good the AUC looks.*

**Checkpoint:** reliability diagram tracks the diagonal. ECE materially lower post-calibration.

---

# PHASE 5 — Policy and threshold selection (target: 3h)

Three bands, deterministic, zero ML and zero LLM in this module:

```
score < T_LOW               → ALLOW
T_LOW <= score < T_HIGH     → STEP-UP  (OTP / 3DS)
score >= T_HIGH             → HOLD for review
```

The middle band is an **abstention mechanism** — frame it that way. It lets the system decline to decide rather than guess under uncertainty, and it is more sophisticated than "medium risk."

**Sweep both thresholds against net rupees.** Grid over `T_HIGH`, and for each, grid over `T_LOW`. Produce:

- Net rupees vs `T_HIGH`, optimum marked
- **0.5 marked on the same axis, with the gap labelled in rupees**
- The F1-optimal threshold marked too, with its rupee cost relative to the optimum

That chart is your centrepiece slide.

**Sensitivity analysis.** Re-run threshold selection with each cost parameter perturbed ±30%. Plot the resulting band around the optimum.

If the optimum barely moves, **that is the finding** — the recommendation survives being wrong about the inputs. If it moves a lot, you have identified which parameter the merchant needs to measure properly. Either way it converts your threshold from an assertion into a result.

**Deterministic policy tests** in `eval/test_policy.py`: band boundaries exactly at `T_LOW` and `T_HIGH`, scores of 0.0 and 1.0, empty input, all-fraud input, all-legit input, cost function reconciles against hand-computed values on a 10-row fixture.

**Checkpoint:** sweep chart renders. Optimum is not 0.5. Sensitivity band computed. Policy tests green.

---

# PHASE 6 — Audit chain (target: 2h)

Append-only SQLite table, each row carrying `prev_hash` — SHA-256 of the previous entry — so tampering is detectable rather than merely discouraged.

Per decision: timestamp, transaction_id, **model_version**, raw score, calibrated score, band, `T_LOW` and `T_HIGH` in effect, top-3 attributions, action, outcome once known, `prev_hash`, `entry_hash`.

Two things this buys beyond compliance:

- **Threshold changes are logged as their own event type**, so "why did the block rate jump on Tuesday" is answerable without archaeology.
- **Model version on every row**, so after a retrain old decisions remain attributable to the model that made them.

Add a `verify_chain()` function and a dev-only tamper endpoint so you can break it live on stage.

**Checkpoint:** score a batch, verify the chain, tamper one row, watch verification fail.

---

# PHASE 7 — Dashboard (target: 4h)

Streamlit. Screens in this order:

1. **Cost matrix** — displayed first so every later number has units
2. **Data provenance** — split table, date boundaries, base rates, what is real vs synthetic
3. **Baseline vs model** — precision, recall, PR-AUC, net rupees, side by side
4. **Reliability diagram** — before and after calibration
5. **Net-rupee sweep** — optimum, 0.5, F1-optimum, sensitivity band
6. **Live monitor** — spike view (see below)
7. **Case detail** — analyst view and customer view side by side
8. **Audit trail** — chain with verify button

### Spike monitor — label it honestly

Rolling windows over scorer output, not a separate model:

```
mean_score_5min vs 7-day baseline
hold_rate_5min vs baseline
distinct_devices_per_card in window
```

Alert on N standard deviations from baseline. **Label this on screen as a monitoring heuristic.** Do not report precision and recall for spike detection — you will have a handful of spike events and the metrics would be unfalsifiable. If asked, say exactly that; declining to report a meaningless metric is itself a demonstration of the track's bar.

### Dual view — enforce at the API boundary

**Internal analyst view:** full attributions, feature values, band, thresholds, model version.

**Customer-facing view:** `"This transaction needs additional verification."` Nothing else. No signals, no counts, no thresholds.

Enforce the split in the data layer, not the template, so the detailed payload cannot reach a customer surface even by mistake. Show both side by side in the demo and say why.

**Checkpoint:** all eight screens render. Customer view provably cannot access attribution data.

---

# PHASE 8 — LLM analyst layer (target: 2h)

The model scores. The policy decides. The LLM explains, to an analyst, after the fact.

Input is extracted evidence only — never the raw transaction, never the thresholds:

```json
{
  "risk_score": 0.91,
  "top_features": [
    { "name": "distinct_cards_per_device", "value": 7, "contribution": 0.31 },
    { "name": "velocity_1h_ratio", "value": 6.2, "contribution": 0.24 },
    { "name": "amount_zscore", "value": 4.1, "contribution": 0.18 }
  ],
  "decision": "HOLD"
}
```

**Extractive only.** Every numeric claim in the generated narrative must trace to a value in that payload. Validate the output against the input and reject any number not present.

**Missing evidence means refusal.** If required features are null, the generator declines and the case goes to a human bare, rather than getting a fabricated rationale. A confabulated justification for a customer-affecting decision is worse than no justification.

Demo the refusal.

**Checkpoint:** feed a payload with nulls, confirm refusal. Feed a normal one, confirm every number in the output appears in the input.

---

# PHASE 9 — Attack injector (target: 2h)

Synthetic only, demo only, **never used for reported metrics.**

Generate a burst: shared device across many cards, velocity spike, amount anomalies, geo mismatch. Inject into the live monitor mid-demo and watch the spike view react.

Label it on screen as synthetic. Volunteering the boundary reads as rigour; being caught not volunteering it reads as the opposite.

---

# PHASE 10 — Selective labels (target: 1h)

One slide, and it is the one that separates you from the rest of the track.

Every transaction you hold produces **no outcome label**. You never learn whether it would have been fraud. Training data is censored by your own past decisions, and the model progressively goes blind in exactly the region it blocks — while measured precision *improves*, because you are only scoring the population you chose to let through.

Mitigation: a **randomized exploration slice**.

```
exploration_rate    2% of would-be-holds allowed through
expected_cost       0.02 × hold_volume × FN_COST
labels_recovered    ~N per month in the blind region
```

Compute the actual numbers for your test-set volumes. Note that the alternative is a model whose measured performance improves while its real performance decays.

---

## Definition of done

- [ ] Leakage test green — features identical with and without future data
- [ ] Zero card/device overlap across splits
- [ ] Test window strictly later than train window
- [ ] Rule baseline reported alongside model
- [ ] PR-AUC reported as primary, ROC-AUC secondary
- [ ] Reliability diagram tracks diagonal post-isotonic
- [ ] Optimal threshold ≠ 0.5, gap quantified in rupees
- [ ] Sensitivity band computed at ±30%
- [ ] Review rate reported as % and as analyst-hours per 1,000
- [ ] Prevented loss discounted by retry rate, assumption stated
- [ ] Customer view cannot reach attribution data
- [ ] Audit chain verifies; tampering breaks it
- [ ] Analyst layer refuses on missing evidence
- [ ] Selective-labels slide with real numbers

---

## Six-minute demo

| Beat | Time |
|---|---|
| Cost matrix | 0:00–0:40 |
| Data provenance + split discipline | 0:40–1:10 |
| Rule baseline vs model | 1:10–1:50 |
| PR-AUC, precision, recall, reliability diagram | 1:50–2:40 |
| **Net-rupee sweep, optimum vs 0.5, sensitivity band** | 2:40–3:40 |
| Live attack injection | 3:40–4:20 |
| Analyst view vs customer view | 4:20–5:00 |
| Evidence-missing refusal | 5:00–5:20 |
| **Selective labels + exploration budget** | 5:20–6:00 |

Close on one figure with both sides of the ledger: **net rupees saved per 1,000 transactions at the chosen threshold, and the analyst-hours it costs.**

---

## Do not build

Real-time streaming infrastructure — batch scoring is fine and nobody will ask. Deep learning. A model zoo. Isolation Forest (you have labels; unsupervised anomaly detection answers a different question and is strictly worse). SHAP dashboards aimed at customers. Multi-network chargeback rules. Graph-based ring detection — genuinely interesting, and a different project.

---

## If you fall behind

Cut in this order: attack injector → spike monitor → LLM analyst layer → dashboard polish.

**Never cut:** the temporal split, the leakage test, the calibration curve, the cost model, or the net-rupee sweep. Those five are the submission; everything else is presentation around them.

---

**Pitch:** *A fraud scorer whose threshold is chosen in rupees rather than F1, calibrated so the probabilities mean what they say, evaluated on a temporal entity-disjoint split against a rule baseline, with the false-positive cost of blocking a real customer priced in — including the churn nobody counts, and the labels we stop receiving the moment we start blocking.*

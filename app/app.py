"""Phase 7 — Streamlit dashboard. Run: streamlit run app/app.py

Screen order matters: cost matrix first, so every later rupee figure
already has units attached.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audit.chain import connect, verify_chain, tamper_row, DEFAULT_DB_PATH
from src.data.split import PROCESSED_DIR
from src.model.train import MODELS_DIR, MODEL_VERSION
from src.monitor.spike import bucket_scored_transactions, detect_spikes
from src.policy.costs import DEFAULT_COSTS
from src.analyst.llm_analyst import generate_analyst_note
from src.serving.views import customer_view, analyst_view
from src.synthetic.attack_injector import generate_attack_burst

RESULTS_DIR = MODELS_DIR / "phase5"

st.set_page_config(page_title="AI Risk Manager — Track 02", layout="wide")


@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_parquet(path):
    return pd.read_parquet(path)


@st.cache_data
def load_scored_and_features():
    scored = load_parquet(PROCESSED_DIR / "test_scored.parquet")
    features = load_parquet(PROCESSED_DIR / "test_features.parquet")
    return scored, features


def band_of(score, t_low, t_high):
    if score >= t_high:
        return "HOLD"
    if score >= t_low:
        return "STEPUP"
    return "ALLOW"


# ---------------------------------------------------------------------
SCREENS = [
    "1. Cost matrix",
    "2. Data provenance",
    "3. Baseline vs model",
    "4. Reliability diagram",
    "5. Net-rupee sweep",
    "6. Live monitor",
    "7. Case detail",
    "8. Audit trail",
    "9. Selective labels",
]

screen = st.sidebar.radio("Screen", SCREENS)
st.sidebar.markdown("---")
st.sidebar.caption("Razorpay Track 02 — AI Risk Manager")


# ---------------------------------------------------------------------
if screen == "1. Cost matrix":
    st.title("Cost matrix")
    st.caption("Every rupee figure on every later screen traces back to these numbers.")
    c = DEFAULT_COSTS

    col1, col2, col3 = st.columns(3)
    col1.metric("False Negative cost (fraud allowed)", f"Rs. {c.fn_cost:,.0f}")
    col2.metric("False Positive cost (legit blocked)", f"Rs. {c.fp_cost:,.0f}")
    col3.metric("FN : FP ratio", f"{c.fn_fp_ratio:.1f} : 1")

    st.markdown("### False Negative — a fraud we let through")
    st.table(pd.DataFrame([
        {"component": "Avg fraud amount", "value": c.avg_fraud_amount, "source": "given"},
        {"component": "Dispute fee", "value": c.dispute_fee, "source": "estimate — Razorpay docs cite Rs 200-600/dispute"},
        {"component": "Handling", "value": c.handling, "source": "estimate — ~20 analyst-min fully loaded"},
        {"component": "TOTAL FN_COST", "value": c.fn_cost, "source": ""},
    ]))

    st.markdown("### False Positive — a legitimate customer wrongly blocked")
    st.table(pd.DataFrame([
        {"component": "Avg legit margin", "value": c.avg_legit_margin, "source": "given — 12% of Rs 2,000 order"},
        {"component": "Support contact", "value": c.support_contact, "source": "given"},
        {"component": "Churn cost (P_churn x LTV)", "value": c.p_churn_wrong_block * c.customer_ltv,
         "source": f"given — {c.p_churn_wrong_block:.0%} x Rs {c.customer_ltv:,.0f} LTV"},
        {"component": "TOTAL FP_COST", "value": c.fp_cost, "source": ""},
    ]))
    st.info("Churn dominates FP cost — it's nearly double the margin itself. Pricing a wrong block "
            "at just the lost fee is roughly an order of magnitude low.")

    st.markdown("### Step-up and review")
    st.table(pd.DataFrame([
        {"component": "Step-up cost (expected, per stepped-up txn)", "value": round(c.stepup_cost, 2),
         "source": f"given — {c.p_abandon_stepup:.0%} abandon x (margin + churn)"},
        {"component": "Review cost", "value": c.review_cost, "source": "given — ~4 analyst-min fully loaded"},
        {"component": "Retry rate (prevented-loss discount)", "value": c.retry_rate,
         "source": "given — ASSUMPTION, not measured on this dataset"},
    ]))
    st.warning("Prevented != detected. A blocked fraudster typically retries on another card, so "
               f"prevented loss is discounted by the {c.retry_rate:.0%} retry rate — a smaller headline "
               "number, but a credible one.")


# ---------------------------------------------------------------------
elif screen == "2. Data provenance":
    st.title("Data provenance")
    report_path = PROCESSED_DIR / "split_report.json"
    if not report_path.exists():
        st.error("Run `python -m src.data.split` first.")
    else:
        report = load_json(report_path)
        st.markdown("### Temporal, entity-disjoint split")
        df = pd.DataFrame(report["slices"])
        df["base_rate"] = df["base_rate"].map(lambda x: f"{x:.4%}")
        st.table(df)

        st.markdown("### Entity-disjointness enforcement")
        dr = report["drop_report"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows dropped (total)", f"{dr['rows_dropped_total']:,}")
        col2.metric("...by card_entity conflict", f"{dr['rows_dropped_by_entity']['card_entity']:,}")
        col3.metric("...by device_entity conflict", f"{dr['rows_dropped_by_entity']['device_entity']:,}")
        st.caption(
            "card_entity = card1 + addr1 + account-open-day anchor (D1n). Raw card1-6 alone is a "
            "BIN/network/type signature shared by thousands of cards, not a unique-card id — using it "
            "verbatim collapsed calibration/test to a few hundred rows each. See src/data/entities.py."
        )
        st.markdown("### What is real vs synthetic")
        st.success("Everything on this screen and screens 3-5, 7-8 is real IEEE-CIS transaction data. "
                    "Only the Live Monitor screen's injected burst (screen 6) is synthetic, and it is "
                    "labelled on screen and never used for a reported metric.")


# ---------------------------------------------------------------------
elif screen == "3. Baseline vs model":
    st.title("Baseline rules vs LightGBM")
    baselines_path = RESULTS_DIR / "baselines.json"
    summary_path = RESULTS_DIR / "sweep_summary.json"
    metrics_path = MODELS_DIR / f"{MODEL_VERSION}_metrics.json"
    if not (baselines_path.exists() and summary_path.exists() and metrics_path.exists()):
        st.error("Run `python -m src.model.train` then `python -m src.policy.run_phase5` first.")
    else:
        baselines = load_json(baselines_path)
        summary = load_json(summary_path)
        metrics = load_json(metrics_path)

        rows = []
        for name, m in baselines.items():
            if name == "_reference_stats":
                continue
            rows.append({"policy": name, "precision": m["precision"], "recall": m["recall"],
                         "net_rupees": m["net"], "review_rate": m.get("flagged_rate")})
        rows.append({
            "policy": "LightGBM (net-rupee optimum)", "precision": summary["optimum"]["precision"],
            "recall": summary["optimum"]["recall"], "net_rupees": summary["optimum"]["net"],
            "review_rate": summary["optimum"]["review_rate"],
        })
        comp = pd.DataFrame(rows).sort_values("net_rupees", ascending=False)
        st.dataframe(comp.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                          "net_rupees": "Rs. {:,.0f}", "review_rate": "{:.2%}"}))

        col1, col2 = st.columns(2)
        col1.metric("PR-AUC (primary)", f"{metrics['pr_auc_test']:.4f}")
        col2.metric("ROC-AUC (secondary)", f"{metrics['roc_auc_test']:.4f}")
        st.caption("PR-AUC leads: at a "
                   f"{metrics['base_rate_test']:.2%} base rate, ROC-AUC is flattered by the huge "
                   "true-negative mass and overstates how useful the model actually is.")

        best_rule_net = max(m["net"] for n, m in baselines.items() if n != "_reference_stats")
        model_net = summary["optimum"]["net"]
        if model_net > best_rule_net:
            st.success(f"Model beats the best hand-written rule by Rs. {model_net - best_rule_net:,.0f} "
                       "net on the test slice.")
        else:
            st.warning("The model does NOT clearly beat the best rule baseline — reporting this "
                       "honestly is the point of the track.")


# ---------------------------------------------------------------------
elif screen == "4. Reliability diagram":
    st.title("Calibration — reliability diagram")
    before_path = MODELS_DIR / f"{MODEL_VERSION}_reliability_before.json"
    after_path = MODELS_DIR / f"{MODEL_VERSION}_reliability_after.json"
    metrics_path = MODELS_DIR / f"{MODEL_VERSION}_metrics.json"
    if not (before_path.exists() and after_path.exists()):
        st.error("Run `python -m src.model.train` first.")
    else:
        before = pd.DataFrame(load_json(before_path))
        after = pd.DataFrame(load_json(after_path))
        metrics = load_json(metrics_path)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                  line=dict(dash="dash", color="gray")))
        fig.add_trace(go.Scatter(x=before["mean_predicted"], y=before["observed_frequency"],
                                  mode="lines+markers", name="Before (raw score)"))
        fig.add_trace(go.Scatter(x=after["mean_predicted"], y=after["observed_frequency"],
                                  mode="lines+markers", name="After (isotonic, fit on calib only)"))
        fig.update_layout(xaxis_title="Predicted probability", yaxis_title="Observed fraud frequency",
                           height=500)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Brier score", f"{metrics['brier_after']:.4f}", delta=f"{metrics['brier_after']-metrics['brier_before']:.4f}",
                    delta_color="inverse")
        col2.metric("Expected calibration error", f"{metrics['ece_after']:.4f}",
                    delta=f"{metrics['ece_after']-metrics['ece_before']:.4f}", delta_color="inverse")
        st.info("If the model says 0.7 and those transactions turn out fraudulent 40% of the time, "
                "every downstream rupee calculation is wrong regardless of how good the AUC looks.")


# ---------------------------------------------------------------------
elif screen == "5. Net-rupee sweep":
    st.title("Net-rupee threshold sweep")
    sweep_path, summary_path, sens_path = (RESULTS_DIR / f for f in
                                            ["sweep.json", "sweep_summary.json", "sensitivity.json"])
    if not sweep_path.exists():
        st.error("Run `python -m src.policy.run_phase5` first.")
    else:
        sweep_df = pd.DataFrame(load_json(sweep_path))
        summary = load_json(summary_path)
        sens_df = pd.DataFrame(load_json(sens_path))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sweep_df["t_high"], y=sweep_df["net"], mode="lines", name="Best net(T_HIGH)"))
        fig.add_trace(go.Scatter(x=[summary["optimum"]["t_high"]], y=[summary["optimum"]["net"]],
                                  mode="markers", marker=dict(size=14, color="green", symbol="star"),
                                  name=f"Optimum (T_HIGH={summary['optimum']['t_high']:.2f})"))
        fig.add_trace(go.Scatter(x=[0.5], y=[summary["naive_0.5"]["net"]],
                                  mode="markers", marker=dict(size=12, color="red"), name="Naive 0.5 cutoff"))
        fig.add_trace(go.Scatter(x=[summary["f1_optimal"]["t"]], y=[summary["f1_optimal"]["net"]],
                                  mode="markers", marker=dict(size=12, color="orange"), name="F1-optimal cutoff"))
        fig.update_layout(xaxis_title="T_HIGH", yaxis_title="Net rupees (test slice)", height=500)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Gap vs 0.5 cutoff", f"Rs. {summary['gap_vs_0.5_rupees']:,.0f}")
        col2.metric("Gap vs F1-optimal cutoff", f"Rs. {summary['gap_vs_f1_rupees']:,.0f}")
        col3.metric("Review rate", f"{summary['optimum']['review_rate']:.2%}",
                    help=f"{summary['optimum']['analyst_hours_per_1000_txns']:.1f} analyst-hours per 1,000 transactions")
        st.caption(f"Chosen policy: ALLOW below {summary['optimum']['t_low']:.2f}, STEP-UP up to "
                   f"{summary['optimum']['t_high']:.2f}, HOLD above. "
                   f"{summary['optimum']['analyst_hours_per_1000_txns']:.1f} analyst-hours per 1,000 transactions.")
        st.success(f"**Both sides of the ledger, per 1,000 transactions:** "
                   f"Rs. {summary['optimum']['net_per_1000_txns']:,.0f} net, costing "
                   f"{summary['optimum']['analyst_hours_per_1000_txns']:.1f} analyst-hours.")
        if summary["optimum"]["net"] < 0:
            st.info(
                f"Net at the optimum is still negative in absolute terms — the model reduces fraud "
                f"loss, it doesn't eliminate it (recall at this operating point is "
                f"{summary['optimum']['recall']:.1%}). Read it against doing nothing "
                f"(t=1.0, allow everyone): that costs Rs. {summary['do_nothing']['net']:,.0f}. "
                f"The chosen policy is Rs. {summary['improvement_vs_do_nothing_rupees']:,.0f} better "
                f"than doing nothing, which is the honest number to lead with."
            )

        st.markdown("### Sensitivity — thresholds re-optimized at each cost input +/-30%")
        fig2 = px.scatter(sens_df, x="param", y="opt_t_high", color="direction",
                           title="Where the optimal T_HIGH lands under each perturbation")
        fig2.add_hline(y=summary["optimum"]["t_high"], line_dash="dash",
                        annotation_text="unperturbed optimum")
        st.plotly_chart(fig2, use_container_width=True)
        band_width = sens_df["opt_t_high"].max() - sens_df["opt_t_high"].min()
        if band_width < 0.10:
            st.success(f"Optimal T_HIGH moves only {band_width:.3f} across every +/-30% perturbation — "
                       "the recommendation survives being wrong about the cost inputs.")
        else:
            st.warning(f"Optimal T_HIGH swings {band_width:.3f} across perturbations — see which "
                       "parameter dominates in the table below; that's what the merchant needs to "
                       "measure properly.")
        st.dataframe(sens_df)


# ---------------------------------------------------------------------
elif screen == "6. Live monitor":
    st.title("Live monitor — spike view")
    st.warning("MONITORING HEURISTIC, not a model. Precision/recall are not reported here: a demo "
               "produces a handful of spike events, and a metric on a handful of events is unfalsifiable.")

    scored, features = load_scored_and_features()
    summary = load_json(RESULTS_DIR / "sweep_summary.json") if (RESULTS_DIR / "sweep_summary.json").exists() else None
    t_low, t_high = (summary["optimum"]["t_low"], summary["optimum"]["t_high"]) if summary else (0.3, 0.7)

    merged = scored.merge(features[["TransactionID", "distinct_cards_per_device"]], on="TransactionID")
    merged["band"] = merged["calibrated_score"].apply(lambda s: band_of(s, t_low, t_high))

    if "injected_burst" not in st.session_state:
        st.session_state.injected_burst = None

    if st.button("Inject synthetic attack burst"):
        start_dt = int(merged["TransactionDT"].max()) + 60
        burst = generate_attack_burst(start_dt=start_dt, n_cards=15)
        burst_scored = pd.DataFrame({
            "TransactionID": burst["TransactionID"], "isFraud": burst["isFraud"],
            "TransactionDT": burst["TransactionDT"],
            "raw_score": 0.95, "calibrated_score": 0.95,
            # the shared device's fan-out count grows with each new card it transacts on
            "distinct_cards_per_device": range(1, len(burst) + 1),
            "band": "HOLD",
        })
        st.session_state.injected_burst = burst_scored
        st.success(f"Injected {len(burst)} synthetic transactions from one shared device across "
                   "distinct cards, starting at TransactionDT={}.".format(start_dt))

    display_df = merged
    if st.session_state.injected_burst is not None:
        display_df = pd.concat([merged, st.session_state.injected_burst], ignore_index=True)
        st.info("SYNTHETIC BURST ACTIVE — the spike below includes injected demo data, labelled here "
                "and excluded from every reported metric on screens 3-5.")

    bucketed = bucket_scored_transactions(display_df)
    alerts = detect_spikes(bucketed)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bucketed["bucket_start_dt"], y=bucketed["hold_rate"], name="hold_rate (5min)"))
    fig.add_trace(go.Scatter(x=bucketed["bucket_start_dt"], y=bucketed["mean_score"], name="mean_score (5min)"))
    fig.update_layout(height=400, xaxis_title="TransactionDT", yaxis_title="value")
    st.plotly_chart(fig, use_container_width=True)

    if alerts:
        st.error(f"{len(alerts)} spike alert(s) fired (>3 std dev from 7-day rolling baseline):")
        st.dataframe(pd.DataFrame([vars(a) for a in alerts]).tail(20))
    else:
        st.success("No spikes above threshold in the current window.")


# ---------------------------------------------------------------------
elif screen == "7. Case detail":
    st.title("Case detail — analyst view vs customer view")
    cases_path = RESULTS_DIR / "case_samples.json"
    summary_path = RESULTS_DIR / "sweep_summary.json"
    if not cases_path.exists():
        st.error("Run `python -m src.policy.run_phase5` first.")
    else:
        cases = load_json(cases_path)
        summary = load_json(summary_path)
        txn_id = st.selectbox("Transaction", [c["TransactionID"] for c in cases])
        case = next(c for c in cases if c["TransactionID"] == txn_id)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Internal analyst view")
            av = analyst_view({
                "transaction_id": case["TransactionID"], "model_version": MODEL_VERSION,
                "raw_score": case["raw_score"], "calibrated_score": case["calibrated_score"],
                "band": case["band"], "t_low": summary["optimum"]["t_low"], "t_high": summary["optimum"]["t_high"],
                "top_attributions": case["top_attributions"],
            })
            st.json(vars(av))

            st.markdown("#### LLM analyst note (extractive, post-decision)")
            note = generate_analyst_note({
                "risk_score": case["calibrated_score"], "top_features": case["top_attributions"],
                "decision": case["band"], "transaction_id": case["TransactionID"],
            })
            if note["refused"]:
                st.error(f"REFUSED: {note['reason']}")
            else:
                st.text(note["narrative"])
                st.caption(f"source={note['source']}  llm_rejected={note.get('llm_rejected', False)}")

        with col2:
            st.markdown("### Customer-facing view")
            cv = customer_view(case["band"])
            st.json(vars(cv))
            st.caption("This function's signature cannot accept a score, attribution, or threshold — "
                       "see src/serving/views.py. There is no code path from the left column to here.")


# ---------------------------------------------------------------------
elif screen == "8. Audit trail":
    st.title("Audit trail")
    if not DEFAULT_DB_PATH.exists():
        st.error("Run `python -m src.audit.seed` first.")
    else:
        conn = connect()
        rows = conn.execute(
            "SELECT id, event_type, timestamp, transaction_id, model_version, band, t_low, t_high, "
            "action, outcome, entry_hash FROM decisions ORDER BY id DESC LIMIT 100"
        ).fetchall()
        cols = ["id", "event_type", "timestamp", "transaction_id", "model_version", "band",
                "t_low", "t_high", "action", "outcome", "entry_hash"]
        st.dataframe(pd.DataFrame(rows, columns=cols))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Verify chain"):
                report = verify_chain(conn)
                if report["valid"]:
                    st.success(f"Chain valid — {report['rows_checked']} rows checked.")
                else:
                    st.error(f"CHAIN BROKEN at row id(s): {report['broken_row_ids']}")
        with col2:
            tamper_id = st.number_input("Row id to tamper (dev only)", min_value=1, step=1, value=2)
            if st.button("Tamper this row"):
                tamper_row(conn, row_id=int(tamper_id), action="TAMPERED")
                st.warning(f"Row {tamper_id} tampered. Click Verify chain to see it fail.")
        conn.close()


# ---------------------------------------------------------------------
elif screen == "9. Selective labels":
    st.title("Selective labels")
    st.markdown(
        "Every HOLD produces **no outcome label** — we never learn whether it would have been "
        "fraud. Training data is censored by the policy's own past decisions: the model "
        "progressively goes blind in exactly the region it blocks, while measured precision on "
        "the shrinking let-through population can *improve* even as real performance decays."
    )

    sl_path = RESULTS_DIR / "selective_labels.json"
    if not sl_path.exists():
        st.error("Run `python -m src.policy.run_phase5` first.")
    else:
        budget = load_json(sl_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("HOLD volume (test slice)", f"{budget['hold_volume']:,}")
        col2.metric("Fraud rate observed in HOLD", f"{budget['fraud_rate_in_holds']:.1%}")
        col3.metric("Exploration rate", f"{budget['exploration_rate']:.0%}")

        st.markdown("### Mitigation: a randomized exploration slice")
        st.write(
            f"Letting **{budget['exploration_rate']:.0%}** of would-be-holds through anyway, purely "
            "to keep recovering ground truth in the blind region, costs:"
        )
        col1, col2 = st.columns(2)
        col1.metric("Expected cost (test-slice window)", f"Rs. {budget['expected_cost']:,.0f}")
        col2.metric("Labels recovered (test-slice window)", f"{budget['labels_recovered_per_period']:.0f}")

        st.info(
            "The alternative isn't free — it's just invisible: a model whose measured performance "
            "improves while its real performance quietly decays in the region it never sees labels "
            "for again."
        )

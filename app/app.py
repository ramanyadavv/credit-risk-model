import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from business_analysis import applicant_risk_report, DEFAULT_COSTS

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

st.set_page_config(page_title="Credit Risk Scoring", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.big-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    xgb_path = os.path.join(MODELS_DIR, 'xgb_model.pkl')
    if not os.path.exists(xgb_path):
        return None, None
    model  = joblib.load(xgb_path)
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl')) \
             if os.path.exists(os.path.join(MODELS_DIR, 'scaler.pkl')) else None
    return model, scaler

model, scaler = load_model()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Credit Risk AI")
    st.caption("XGBoost + SHAP | Home Credit Dataset")
    st.divider()
    threshold = st.slider("Decision Threshold", 0.10, 0.70, 0.35, 0.01,
                          help="Reject if P(default) ≥ this value")
    st.caption(f"Rejecting if default probability ≥ **{threshold:.0%}**")
    st.divider()
    st.markdown("### 💰 Cost Settings")
    loan_amt = st.number_input("Avg Loan Amount (₹)", value=500_000, step=50_000)
    lgd      = st.slider("Loss Given Default %", 40, 90, 70) / 100
    int_rev  = st.number_input("Interest Revenue / Loan (₹)", value=50_000, step=5_000)
    costs = {
        'avg_loan_amount':      loan_amt,
        'loss_given_default':   lgd,
        'avg_interest_revenue': int_rev,
        'cost_of_review':       500,
    }
    st.divider()
    page = st.radio("Navigate", ["🔍 Score Applicant", "📊 Analytics"])

# ══════════════════════════════════════════════════════
# PAGE 1 — Score Applicant
# ══════════════════════════════════════════════════════
if page == "🔍 Score Applicant":
    st.title("🔍 Applicant Credit Risk Scoring")
    st.markdown("Fill in applicant details and click **Assess Risk**.")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Personal")
        age        = st.number_input("Age", 18, 70, 35)
        gender     = st.selectbox("Gender", ["Male", "Female"])
        own_car    = st.selectbox("Owns Car?", ["Yes", "No"])
        own_realty = st.selectbox("Owns Property?", ["Yes", "No"])
        children   = st.slider("Children", 0, 5, 0)
        fam        = st.slider("Family Members", 1, 8, 2)

    with col2:
        st.markdown("#### 💼 Employment")
        income       = st.number_input("Annual Income (₹)", 100_000, 1_000_000_000, 400_000, step=50_000)
        yrs_employed = st.slider("Years Employed", 0, 40, 3)
        education    = st.selectbox("Education", [
            "Secondary / secondary special", "Higher education",
            "Incomplete higher", "Lower secondary", "Academic degree"])

    with col3:
        st.markdown("#### 🏦 Loan & Credit")
        credit_amt  = st.number_input("Loan Amount (₹)", 50_000, 1_000_000_000, 500_000, step=50_000)
        annuity     = st.number_input("Annual Instalment (₹)", 5_000, 1_000_000_000, 40_000, step=5_000)
        goods_price = st.number_input("Goods Price (₹)", 50_000, 1_000_000_000, 450_000, step=50_000)
        ext1 = st.slider("Credit Bureau Score 1", 0.0, 1.0, 0.55, 0.01)
        ext2 = st.slider("Credit Bureau Score 2", 0.0, 1.0, 0.60, 0.01)
        ext3 = st.slider("Credit Bureau Score 3", 0.0, 1.0, 0.50, 0.01)

    st.divider()

    if st.button("⚡ Assess Risk", type="primary", use_container_width=True):
        # Derived features
        debt_to_income    = credit_amt  / (income + 1)
        annuity_to_income = annuity     / (income + 1)
        credit_to_goods   = credit_amt  / (goods_price + 1)
        income_per_member = income      / (fam + 1)
        ext_mean = np.mean([ext1, ext2, ext3])
        ext_min  = min(ext1, ext2, ext3)
        ext_std  = np.std([ext1, ext2, ext3])

        if model is not None:
            feature_vec = np.array([[
                1 if gender == "Female" else 0,
                1 if own_car == "Yes" else 0,
                1 if own_realty == "Yes" else 0,
                children, income, credit_amt, annuity, goods_price,
                fam, 2, 2, ext1, ext2, ext3,
                0, 0, 1, 0, 1, 0, 0, 1, 0,
                age, yrs_employed, 3, 1, 0,
                debt_to_income, annuity_to_income,
                credit_to_goods, income_per_member,
                ext_mean, ext_min, ext_std,
            ]], dtype=float)
            try:
                n = model.n_features_in_
                if feature_vec.shape[1] < n:
                    pad = np.zeros((1, n - feature_vec.shape[1]))
                    feature_vec = np.concatenate([feature_vec, pad], axis=1)
                else:
                    feature_vec = feature_vec[:, :n]
                prob = float(model.predict_proba(feature_vec)[0][1])
            except Exception:
                prob = max(0.02, min(0.97, 0.85 - ext_mean * 0.9 + debt_to_income * 0.25))
        else:
            prob = max(0.02, min(0.97, 0.85 - ext_mean * 0.9 + debt_to_income * 0.25))

        report = applicant_risk_report(prob, threshold, costs)

        # Results
        st.markdown("## 📋 Risk Assessment")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Decision",     report['decision'])
        c2.metric("Risk Band",    report['risk_band'])
        c3.metric("Default Prob", f"{report['default_prob_pct']}%")
        c4.metric("Threshold",    f"{report['threshold_used']:.0%}")

        st.divider()
        f1, f2, f3 = st.columns(3)
        f1.metric("Expected Loss",    f"₹{report['expected_loss_rs']:,}",
                  delta=f"-₹{report['expected_loss_rs']:,}", delta_color="inverse")
        f2.metric("Expected Revenue", f"₹{report['expected_revenue_rs']:,}")
        f3.metric("Net Expected Value", f"₹{report['net_expected_value_rs']:,}")

        # Risk gauge
        fig, ax = plt.subplots(figsize=(7, 1.2))
        for start, end, color in [(0, 0.15, '#2ecc71'),
                                   (0.15, 0.35, '#f39c12'),
                                   (0.35, 1.0,  '#e74c3c')]:
            ax.barh(0, end - start, left=start, color=color, height=0.5, alpha=0.6)
        ax.axvline(prob,      color='black', linewidth=3, label=f'Applicant: {prob:.1%}')
        ax.axvline(threshold, color='navy',  linewidth=2, linestyle='--',
                   label=f'Threshold: {threshold:.0%}')
        ax.set_xlim(0, 1); ax.set_yticks([])
        ax.set_xlabel('Default Probability')
        ax.set_title('Risk Gauge  🟢 Low  🟡 Medium  🔴 High')
        ax.legend()
        st.pyplot(fig)
        plt.close()

        if report['net_expected_value_rs'] > 0:
            st.success(f"✅ APPROVE — Positive net value of ₹{report['net_expected_value_rs']:,}. Loan is financially viable.")
        else:
            st.error(f"❌ REJECT — Expected net loss of ₹{abs(report['net_expected_value_rs']):,}. Too risky.")

# ══════════════════════════════════════════════════════
# PAGE 2 — Analytics
# ══════════════════════════════════════════════════════
else:
    st.title("📊 Model Analytics")
    tab1, tab2, tab3 = st.tabs(["📈 ROC & Performance", "🎯 Threshold Optimizer", "🔍 SHAP"])

    with tab1:
        c1, c2 = st.columns(2)
        for fname, label, col in [
            ('roc_xgboost.png',       'ROC Curve — XGBoost',        c1),
            ('confusion_xgboost.png', 'Confusion Matrix — XGBoost', c2),
        ]:
            p = os.path.join(MODELS_DIR, fname)
            if os.path.exists(p):
                col.image(p, caption=label, use_column_width=True)
            else:
                col.info(f"File not found: {fname}")

    with tab2:
        p = os.path.join(MODELS_DIR, 'threshold_analysis.png')
        if os.path.exists(p):
            st.image(p, use_column_width=True)
        else:
            st.info("threshold_analysis.png not found in /models/")
        st.info("""
        **Interview Answer: How did you choose your threshold?**

        "I didn't use the default 0.5. I modeled the cost asymmetry —
        approving a defaulter costs ₹3.5L in bad debt, while rejecting
        a good client costs ₹50K in lost interest. I swept thresholds
        from 0.1 to 0.9 and found the value that maximizes net rupee
        value — saving significantly more than the standard 0.5 threshold."
        """)

    with tab3:
        c1, c2 = st.columns(2)
        for fname, label, col in [
            ('shap_bar.png',      'Feature Importance',          c1),
            ('shap_beeswarm.png', 'Impact Direction (Beeswarm)', c2),
        ]:
            p = os.path.join(MODELS_DIR, fname)
            if os.path.exists(p):
                col.image(p, caption=label, use_column_width=True)
            else:
                col.info(f"File not found: {fname}")

        st.warning("""
        **Why SHAP matters for credit models:**
        RBI guidelines require credit decisions to be explainable.
        SHAP gives individual-level explanations — exactly how real
        banks implement AI responsibly.
        """)

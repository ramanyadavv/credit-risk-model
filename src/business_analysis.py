import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

DEFAULT_COSTS = {
    'avg_loan_amount': 500_000,
    'loss_given_default': 0.70,
    'avg_interest_revenue': 50_000,
    'cost_of_review': 500,
}

def compute_expected_loss(y_true, y_prob, threshold, costs=DEFAULT_COSTS):
    y_pred = (y_prob >= threshold).astype(int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    loss_bad = FN * costs['avg_loan_amount'] * costs['loss_given_default']
    loss_rej = FP * costs['avg_interest_revenue']
    revenue  = TN * costs['avg_interest_revenue']
    net      = revenue - loss_bad - loss_rej
    return {
        'threshold': round(threshold, 2),
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'approval_rate': round((TN + FN) / len(y_true) * 100, 1),
        'loss_bad_loans': round(loss_bad / 1e7, 2),
        'loss_rejected': round(loss_rej / 1e7, 2),
        'revenue': round(revenue / 1e7, 2),
        'net_value_crores': round(net / 1e7, 2),
    }

def threshold_analysis(y_true, y_prob, costs=DEFAULT_COSTS):
    thresholds = np.arange(0.10, 0.91, 0.02)
    results = [compute_expected_loss(y_true, y_prob, t, costs) for t in thresholds]
    df = pd.DataFrame(results)
    best = df.loc[df['net_value_crores'].idxmax()]
    std_rows = df[df['threshold'].between(0.49, 0.51)]
    print(f"\n{'='*60}")
    print(f"  THRESHOLD OPTIMIZATION")
    print(f"{'='*60}")
    if len(std_rows):
        print(f"  Standard (0.50) → ₹{std_rows['net_value_crores'].values[0]:.1f} Cr")
    print(f"  Optimal  ({best['threshold']:.2f}) → ₹{best['net_value_crores']:.1f} Cr")
    print(f"  Approval Rate: {best['approval_rate']}%")
    print(f"{'='*60}\n")
    _plot(df, float(best['threshold']))
    return df, float(best['threshold'])

def _plot(df, best_threshold):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(df['threshold'], df['net_value_crores'], color='steelblue', linewidth=2.5)
    ax1.axvline(best_threshold, color='red', linestyle='--', label=f'Optimal={best_threshold:.2f}')
    ax1.axvline(0.50, color='gray', linestyle=':', label='Standard=0.50')
    ax1.set_xlabel('Threshold'); ax1.set_ylabel('₹ Crores')
    ax1.set_title('Net Value vs Threshold'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.stackplot(df['threshold'], df['loss_bad_loans'], df['loss_rejected'],
                  labels=['Bad Loans', 'Rejected Good'], colors=['#e74c3c','#f39c12'], alpha=0.7)
    ax2.plot(df['threshold'], df['revenue'], color='#27ae60', linewidth=2.5, label='Revenue')
    ax2.axvline(best_threshold, color='black', linestyle='--')
    ax2.set_xlabel('Threshold'); ax2.set_ylabel('₹ Crores')
    ax2.set_title('Revenue vs Loss'); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'threshold_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

def applicant_risk_report(prob, threshold, costs=DEFAULT_COSTS):
    exp_loss = prob * costs['avg_loan_amount'] * costs['loss_given_default']
    exp_rev  = (1 - prob) * costs['avg_interest_revenue']
    net      = exp_rev - exp_loss
    return {
        'decision': 'APPROVE ✅' if prob < threshold else 'REJECT ❌',
        'risk_band': ('Low Risk 🟢' if prob < 0.15 else
                      'Medium Risk 🟡' if prob < 0.35 else 'High Risk 🔴'),
        'default_prob_pct': round(prob * 100, 1),
        'expected_loss_rs': round(exp_loss),
        'expected_revenue_rs': round(exp_rev),
        'net_expected_value_rs': round(net),
        'threshold_used': round(threshold, 2),
    }

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
    roc_curve, classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    print(f"[INFO] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train, random_state=42):
    print(f"[INFO] Before SMOTE: {y_train.value_counts().to_dict()}")
    sm = SMOTE(random_state=random_state, sampling_strategy=0.3)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[INFO] After  SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res

def scale(X_train, X_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    return X_train_sc, X_test_sc, scaler

def train_logistic(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, class_weight='balanced',
                            C=0.1, random_state=42)
    lr.fit(X_train, y_train)
    return lr

def train_xgboost(X_train, y_train, scale_pos_weight=None):
    if scale_pos_weight is None:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric='auc', random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return xgb

def evaluate(model, X_test, y_test, model_name, feature_names=None):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)
    print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
    print(f"  ROC-AUC : {roc_auc:.4f}\n  PR-AUC  : {pr_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    name = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(MODELS_DIR, f'confusion_{name}.png'), dpi=150)
    plt.close()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f'roc_{name}.png'), dpi=150)
    plt.close()
    return {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'y_prob': y_prob}

def explain_with_shap(model, X_test, feature_names, n_samples=300):
    print("[INFO] Computing SHAP values...")
    X_sample = pd.DataFrame(X_test, columns=feature_names).sample(
        n=min(n_samples, len(X_test)), random_state=42)
    # Fix: convert all columns to float to avoid object dtype error
    X_sample = X_sample.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type='bar',
                      show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'shap_bar.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'shap_beeswarm.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] SHAP plots saved")
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    return shap_df

def run_training(X, y):
    feature_names = list(X.columns)
    # Fix: convert entire dataframe to float before training
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    X_train_sc, X_test_sc, scaler = scale(X_train_res, X_test)
    print("\n[TRAINING] Logistic Regression...")
    lr_model  = train_logistic(X_train_sc, y_train_res)
    print("[TRAINING] XGBoost...")
    xgb_model = train_xgboost(X_train_res, y_train_res)
    lr_results  = evaluate(lr_model,  X_test_sc,     y_test, 'Logistic Regression')
    xgb_results = evaluate(xgb_model, X_test.values, y_test, 'XGBoost', feature_names)
    joblib.dump(xgb_model,     os.path.join(MODELS_DIR, 'xgb_model.pkl'))
    joblib.dump(lr_model,      os.path.join(MODELS_DIR, 'lr_model.pkl'))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.pkl'))
    print("\n[INFO] Models saved to /models/")
    shap_df = explain_with_shap(xgb_model, X_test.values, feature_names)
    print("\n[TOP 10 FEATURES BY SHAP]")
    print(shap_df.head(10).to_string(index=False))
    return (xgb_model, lr_model, scaler, X_test, y_test,
            xgb_results, lr_results, feature_names)

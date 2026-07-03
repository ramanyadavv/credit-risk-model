import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess        import full_pipeline
from train             import run_training
from business_analysis import threshold_analysis

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'application_train.csv')

def main():
    print("\n" + "="*60)
    print("   CREDIT RISK MODEL — FULL PIPELINE")
    print("="*60)

    # Step 1: Preprocess
    print("\n[STEP 1] Preprocessing...")
    X, y, features = full_pipeline(DATA_PATH)

    # Step 2: Train
    print("\n[STEP 2] Training...")
    (xgb_model, lr_model, scaler,
     X_test, y_test,
     xgb_results, lr_results,
     feature_names) = run_training(X, y)

    # Step 3: Business analysis
    print("\n[STEP 3] Threshold optimization...")
    df_thresh, optimal_threshold = threshold_analysis(
        y_test.values,
        xgb_results['y_prob']
    )

    print("\n✅ DONE!")
    print(f"   Optimal threshold : {optimal_threshold:.2f}")
    print(f"   Models saved to   : /models/")
    print(f"   Plots saved to    : /models/*.png")
    print(f"\n   Launch dashboard  : streamlit run app/app.py\n")

if __name__ == '__main__':
    main()
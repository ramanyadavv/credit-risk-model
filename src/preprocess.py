import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

SELECTED_FEATURES = [
    'TARGET', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'NAME_FAMILY_STATUS',
    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE',
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
    'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE',
    'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
    'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
    'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'AMT_REQ_CREDIT_BUREAU_YEAR',
]

def load_raw(filepath):
    df = pd.read_csv(filepath)
    available = [c for c in SELECTED_FEATURES if c in df.columns]
    print(f"[INFO] Loaded {len(df):,} rows | Using {len(available)} features")
    return df[available].copy()

def clean(df):
    df = df.copy()
    df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365).round(1)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, 0)
    df['YEARS_EMPLOYED'] = (-df['DAYS_EMPLOYED'] / 365).clip(lower=0).round(1)
    df['DAYS_ID_PUBLISH'] = (-df['DAYS_ID_PUBLISH']).abs()
    df['DAYS_LAST_PHONE_CHANGE'] = (-df['DAYS_LAST_PHONE_CHANGE']).abs()
    df['DAYS_REGISTRATION'] = (-df['DAYS_REGISTRATION']).abs()
    df.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
                     'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE'],
            errors='ignore', inplace=True)
    return df

def engineer_features(df):
    df = df.copy()
    df['DEBT_TO_INCOME'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_TO_INCOME'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['CREDIT_TO_GOODS'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)
    df['INCOME_PER_MEMBER'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    ext = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    available_ext = [c for c in ext if c in df.columns]
    df['EXT_SOURCE_MEAN'] = df[available_ext].mean(axis=1)
    df['EXT_SOURCE_MIN'] = df[available_ext].min(axis=1)
    df['EXT_SOURCE_STD'] = df[available_ext].std(axis=1).fillna(0)
    return df

def encode_categoricals(df):
    df = df.copy()
    for col in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    multi_cols = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                  'NAME_FAMILY_STATUS', 'OCCUPATION_TYPE']
    available = [c for c in multi_cols if c in df.columns]
    df = pd.get_dummies(df, columns=available, drop_first=True)
    return df

def handle_missing(df):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    return df

def full_pipeline(filepath):
    df = load_raw(filepath)
    df = clean(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    df = handle_missing(df)
    y = df['TARGET']
    X = df.drop(columns=['TARGET'])
    print(f"[INFO] Final shape: {X.shape} | Default rate: {y.mean():.2%}")
    return X, y, list(X.columns)

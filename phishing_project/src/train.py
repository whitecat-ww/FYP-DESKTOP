# src/train.py
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from feature_extractor import extract_features, features_to_vector, FEATURE_ORDER
from tqdm import tqdm
import os
import time

# 定义路径
BASE_DIR = os.path.dirname(__file__)
NEW_DATA_PATH = os.path.join(BASE_DIR, "../data/phishing_dataset.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "../data/history_features.csv")
MODEL_OUT = os.path.join(BASE_DIR, "../models/phishing_rf.pkl") # 保持后缀为pkl方便app调用

def load_new_data(path):
    """加载新上传的原始数据集"""
    if not os.path.exists(path):
        print(f"Warning: {path} not found. No new data to learn.")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        raise ValueError("CSV must contain 'label' column with 1=phishing,0=legit")
    df = df[['url', 'label']].dropna()
    return df

def extract_features_from_df(df):
    """对DataFrame中的URL提取特征"""
    if df.empty:
        return pd.DataFrame()
        
    rows = []
    times = []
    print(f"Extracting features for {len(df)} new URLs...")
    
    for _, r in tqdm(df.iterrows(), total=len(df)):
        url = r['url']
        t0 = time.time()
        try:
            feats = extract_features(url)
            vec = features_to_vector(feats)
            rows.append(vec)
            t1 = time.time()
            times.append(t1 - t0)
        except Exception as e:
            print(f"Error extracting {url}: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    feat_df = pd.DataFrame(rows, columns=FEATURE_ORDER)
    feat_df['label'] = df['label'].values
    # 记录原始URL，方便去重
    feat_df['url_source'] = df['url'].values 
    
    if times:
        print("Average extraction time: {:.2f}s".format(sum(times)/len(times)))
    
    return feat_df

def impute_missing(df):
    """
    升级版缺失值处理：
    - 保留 -1 (WHOIS失败) 和 -999 (SSL失败) 作为特殊特征 (Failure as Feature)
    - 只对真正的 NaN 或其他异常值进行中位数插补
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in ['label', 'url_source']]
    
    # 定义特定的“失败特征码”，这些我们不想被覆盖
    failure_codes = [-1, -999]

    for col in feature_cols:
        # 1. 先把真正的 NaN 填成 0 (或者该列的中位数)
        # 计算中位数时，要排除掉 failure_codes，以免拉低中位数
        valid_values = df[~df[col].isin(failure_codes)][col].dropna()
        if not valid_values.empty:
            med = valid_values.median()
        else:
            med = 0
            
        # 2. 只填补真正的 NaN，保留 -1 和 -999
        df[col] = df[col].fillna(med)
        
        # 关键修改：删除了原来的 replace 语句，不再把 -1/-999 替换成中位数
        
    return df

def merge_with_history(new_feat_df):
    """将新特征合并到历史库中，并去重"""
    
    # 1. 加载历史数据
    if os.path.exists(HISTORY_PATH):
        print("Loading history features...")
        try:
            history_df = pd.read_csv(HISTORY_PATH)
            # 检查特征列是否一致（防止旧history与新代码不兼容）
            if not all(col in history_df.columns for col in FEATURE_ORDER):
                print("Feature mismatch detected. Backing up and resetting history.")
                os.rename(HISTORY_PATH, HISTORY_PATH + ".bak")
                history_df = pd.DataFrame(columns=FEATURE_ORDER + ['label', 'url_source'])
        except Exception:
            history_df = pd.DataFrame(columns=FEATURE_ORDER + ['label', 'url_source'])
    else:
        print("No history found. Creating new history file.")
        history_df = pd.DataFrame(columns=FEATURE_ORDER + ['label', 'url_source'])

    # 2. 合并
    if not new_feat_df.empty:
        # 确保列对齐
        for col in history_df.columns:
            if col not in new_feat_df.columns:
                new_feat_df[col] = 0 # 理论上不会发生
        new_feat_df = new_feat_df[history_df.columns]
        combined_df = pd.concat([history_df, new_feat_df], ignore_index=True)
    else:
        combined_df = history_df

    if combined_df.empty:
        raise ValueError("No data available for training.")

    # 3. 去重 (保留最新的 label)
    if 'url_source' in combined_df.columns:
        before_len = len(combined_df)
        combined_df.drop_duplicates(subset=['url_source'], keep='last', inplace=True)
        print(f"Merged data. Total samples: {len(combined_df)} (Removed {before_len - len(combined_df)} duplicates)")
    
    # 4. 保存
    combined_df.to_csv(HISTORY_PATH, index=False)
    print(f"History updated and saved to {HISTORY_PATH}")
    
    return combined_df

def main():
    # 1. 加载并处理新数据
    new_raw_df = load_new_data(NEW_DATA_PATH)
    new_feat_df = extract_features_from_df(new_raw_df)
    
    # 2. 合并历史
    full_dataset = merge_with_history(new_feat_df)
    
    # 3. 清洗
    full_dataset = impute_missing(full_dataset)
    
    # 4. 准备训练集
    # 【关键修改】：把容易导致作弊的 https 特征加到丢弃列表里！
    drop_cols = ['label', 'url_source', 'is_https', 'https_token']
    X = full_dataset.drop(columns=[c for c in drop_cols if c in full_dataset.columns])
    y = full_dataset['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training XGBoost Classifier...")
    # 使用 XGBoost
    clf = XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1, 
        random_state=42, 
        n_jobs=-1,
        eval_metric='logloss'
    )
    clf.fit(X_train, y_train)

    # 5. 评估
    preds = clf.predict(X_test)
    print("Accuracy on Test Set:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # 6. 保存
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump({'model': clf, 'feature_order': FEATURE_ORDER}, MODEL_OUT)
    print("Saved XGBoost model to", MODEL_OUT)

if __name__ == "__main__":
    main()
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# 定义路径
BASE_DIR = os.path.dirname(__file__)
BACKUP_PATH = os.path.join(BASE_DIR, "../data/temp_features_backup.csv")
MODEL_OUT = os.path.join(BASE_DIR, "../models/phishing_rf.pkl")

def main():
    if not os.path.exists(BACKUP_PATH):
        print(f"❌ 找不到备份文件: {BACKUP_PATH}，请确认文件路径是否正确。")
        return

    print("📥 正在加载备份数据...")
    df = pd.read_csv(BACKUP_PATH)
    
    # 分离特征和标签，同时强制剔除可能导致作弊的特征
    drop_cols = ['label_encoded', 'is_https', 'https_token']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['label_encoded']
    
    print(f"📊 成功读取数据: {len(X)} 条, 当前参与训练的特征数: {len(X.columns)}")
    
    print("✂️ 正在分割训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("🚀 启动 GPU 极速训练 (RTX 4070 Super)...")
    clf = XGBClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        n_jobs=-1,
        device="cuda",      # 使用显卡加速
        tree_method="hist"  
    )
    clf.fit(X_train, y_train)
    
    print("\n--- 🎯 模型评估报告 ---")
    preds = clf.predict(X_test)
    print("准确率 (Accuracy):", accuracy_score(y_test, preds))
    print("\n详细报告:\n", classification_report(y_test, preds))
    
    # 保存模型
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    # 注意：我们要把最终保留下来的特征列表存进去，方便 app.py 调用
    joblib.dump({'model': clf, 'feature_order': list(X.columns)}, MODEL_OUT)
    print(f"\n✅ 太棒了！模型已成功生成并保存到: {MODEL_OUT}")

if __name__ == "__main__":
    main()

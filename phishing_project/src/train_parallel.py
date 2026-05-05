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
import requests # [新增] 用于捕获底层网络超时错误
import concurrent.futures # [新增] 用于控制线程级别的超时
from concurrent.futures import ThreadPoolExecutor

# --- 配置 ---
# 线程数量：根据你的网速调整，通常 20-50 比较合适
MAX_WORKERS = 50

# 定义路径
BASE_DIR = os.path.dirname(__file__)
NEW_DATA_PATH = os.path.join(BASE_DIR, "../data/phishing_dataset.csv")
MODEL_OUT = os.path.join(BASE_DIR, "../models/phishing_rf.pkl")

def process_single_url(data):
    """
    [升级版] 单个 URL 的处理函数，增加了错误类型分类
    """
    url, label = data
    try:
        # 调用特征提取逻辑
        feats = extract_features(url)
        vec = features_to_vector(feats)
        return vec, label, "success"
        
    except requests.exceptions.Timeout:
        # 明确捕获到底层的网络超时
        return None, label, "timeout"
        
    except Exception as e:
        # 其他各种报错（比如解析 HTML 失败、网站彻底无法访问等）
        return None, label, "error"

def load_data_parallel(path):
    print(f"Loading raw data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    
    df = pd.read_csv(path)
    
    # 查找标签列
    target_col = 'status' if 'status' in df.columns else 'label'
    if 'url' not in df.columns or not target_col:
        raise ValueError("CSV must have 'url' and 'label' columns")

    urls = df['url'].values
    labels = df[target_col].values
    
    # 准备数据对
    data_pairs = list(zip(urls, labels))
    total = len(data_pairs)
    
    print(f"🚀 启动工业级多线程提取 (线程数: {MAX_WORKERS})...")
    print("🛡️ 已开启【实时存盘】功能，再也不怕程序卡死，随时可以 Ctrl+C 强退！")
    
    processed_rows = []
    processed_labels = []
    
    # [新增] 详细的状态统计字典
    stats = {"success": 0, "timeout": 0, "error": 0, "thread_killed": 0}
    
    start_time = time.time()
    
    # === 新增：定义一个紧急避险的临时文件路径 ===
    temp_csv_path = os.path.join(BASE_DIR, "../data/temp_features_backup.csv")
    
    # --- 核心：多线程并行执行 (带防假死 + 自动存档) ---
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_url, pair): pair for pair in data_pairs}
        
        # 使用 tqdm 显示进度条
        for future in tqdm(concurrent.futures.as_completed(futures), total=total, unit="url"):
            try:
                # 【终极防线】如果某个线程超过 15 秒还没返回结果，主线程直接判定它死亡！
                result = future.result(timeout=15) 
                
                if result is not None:
                    vec, label, status = result
                    if status == "success":
                        processed_rows.append(vec)
                        processed_labels.append(label)
                        stats["success"] += 1
                        
                        # === 关键护城河：每凑够 100 条，立刻存进硬盘！ ===
                        if stats["success"] > 0 and stats["success"] % 100 == 0:
                            temp_df = pd.DataFrame(processed_rows, columns=FEATURE_ORDER)
                            temp_df['label_encoded'] = processed_labels
                            temp_df.to_csv(temp_csv_path, index=False)
                        # ===================================================
                        
                    else:
                        stats[status] += 1
                        
            except concurrent.futures.TimeoutError:
                # 线程彻底失联 (超过 15 秒没反应)，强行越过
                stats["thread_killed"] += 1
            except Exception:
                # 线程执行过程中发生了崩溃
                stats["error"] += 1
                
                
    end_time = time.time()
    duration = end_time - start_time
    
    # --- 打印超级详细的体检报告 ---
    print(f"\n✅ 特征提取阶段结束！")
    print(f"⏱️ 总耗时: {duration:.2f} 秒 ({total / duration:.2f} URL/s)")
    print(f"📊 提取成功: {stats['success']} 个")
    print(f"⚠️ 网络超时: {stats['timeout']} 个 (网站响应太慢)")
    print(f"❌ 提取报错: {stats['error']} 个 (网站已死或格式错误)")
    print(f"💀 线程假死: {stats['thread_killed']} 个 (强制斩断卡死线程)")
    
    # 如果一条都没成功
    if len(processed_rows) == 0:
        return pd.DataFrame(), pd.Series()

    # 转换为 DataFrame
    X = pd.DataFrame(processed_rows, columns=FEATURE_ORDER)
    
    # 处理标签
    y_raw = pd.Series(processed_labels)
    if y_raw.dtype == object:
        y = y_raw.apply(lambda x: 1 if str(x).lower().strip() == 'phishing' else 0)
    else:
        y = y_raw.astype(int)
        
    return X, y

def main():
    # 1. 并行加载数据
    try:
        X, y = load_data_parallel(NEW_DATA_PATH)
    except Exception as e:
        print(f"Error: {e}")
        return

    if len(X) == 0:
        print("没有提取到有效数据，程序结束。请检查网络或 URL 列表。")
        return

    # === 【关键修改：强制剔除作弊特征】 ===
    # 防止多线程版本训练出的模型也发生“特征泄漏”
    drop_cols = ['is_https', 'https_token']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    print(f"\n⚠️ 已强制剔除作弊特征！当前参与训练的特征数量: {len(X.columns)}")
    # ======================================

    # 2. 分割数据
    print(f"Splitting {len(X)} successful samples...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 3. GPU 训练 (这步是毫秒级的)
    print("🚀 Training XGBoost with RTX 4070 Super...")
    
    clf = XGBClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        n_jobs=-1,
        device="cuda",      # 使用 GPU
        tree_method="hist"  # 极速模式
    )
    
    clf.fit(X_train, y_train)
    print("✅ Model trained!")

    # 4. 评估
    print("\n--- 模型评估报告 ---")
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # 5. 保存
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    # 注意：保存的时候，我们要把真正用于训练的特征列名保存进去
    final_features = list(X.columns)
    joblib.dump({'model': clf, 'feature_order': final_features}, MODEL_OUT)
    print(f"\n💾 Model safely saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
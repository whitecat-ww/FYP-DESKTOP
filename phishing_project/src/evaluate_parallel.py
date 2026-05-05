# src/evaluate_parallel.py
import pandas as pd
import joblib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extractor import extract_features, features_to_vector, FEATURE_ORDER

# --- 配置 ---
MAX_WORKERS = 50  # 线程数，可根据你的网络状况调整

BASE_DIR = os.path.dirname(__file__)
TEST_CSV = os.path.join(BASE_DIR, "../data/unseen_test_dataset.csv") 
MODEL_PATH = os.path.join(BASE_DIR, "../models/phishing_rf.pkl")
CM_IMAGE_PATH = os.path.join(BASE_DIR, "../models/real_world_confusion_matrix.png")

def process_url(data):
    """提取单个 URL 的特征"""
    url, label = data
    try:
        feats = extract_features(url)
        vec = features_to_vector(feats)
        return vec, label
    except Exception as e:
        # 如果提取失败（例如目标网站已彻底死链），返回 None
        return None

def main():
    print("=== 开始真实世界数据 (Unseen Real-world Data) 实测 ===")
    
    # 1. 检查文件是否存在
    if not os.path.exists(TEST_CSV):
        print(f"❌ 找不到测试集文件: {TEST_CSV}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型文件: {MODEL_PATH}。请先运行 train.py 或 train_parallel.py")
        return

    # 2. 加载模型
    print("加载已训练的模型...")
    m = joblib.load(MODEL_PATH)
    clf = m['model']
    # 确保特征顺序一致
    expected_features = m['feature_order'] 

    # 3. 读取测试数据
    df = pd.read_csv(TEST_CSV)
    urls = df['url'].values
    labels = df['label'].values
    data_pairs = list(zip(urls, labels))
    total = len(data_pairs)

    print(f"🚀 开始多线程特征提取 (共 {total} 个 URL, 线程数: {MAX_WORKERS})...")
    
    processed_rows = []
    processed_labels = []
    
    start_time = time.time()
    
    # 4. 并行特征提取
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_url, pair): pair for pair in data_pairs}
        
        for future in tqdm(as_completed(futures), total=total, desc="Extracting"):
            result = future.result()
            if result is not None:
                vec, label = result
                processed_rows.append(vec)
                processed_labels.append(label)

    duration = time.time() - start_time
    print(f"\n✅ 特征提取完成！成功提取 {len(processed_rows)}/{total} 个 URL。耗时: {duration:.2f} 秒。")

    # 5. 模型预测
    X_test = pd.DataFrame(processed_rows, columns=expected_features)
    y_true = processed_labels
    
    print("\n🧠 模型正在进行预测...")
    y_pred = clf.predict(X_test)

    # 6. 计算与打印评估指标
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*40)
    print(f"🏆 真实世界未见数据准确率 (Accuracy): {acc * 100:.2f}%")
    print("="*40)
    print("\n详细分类报告 (Classification Report):")
    print(classification_report(y_true, y_pred, target_names=["Legitimate (0)", "Phishing (1)"]))

    # 7. 生成并保存混淆矩阵图 (Figure 4.5)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'], 
                yticklabels=['Legitimate', 'Phishing'],
                annot_kws={"size": 14}) # 字体放大一点方便论文截图
    plt.xlabel('Predicted Label (模型预测)', fontsize=12)
    plt.ylabel('True Label (真实情况)', fontsize=12)
    plt.title('Figure 4.5: Confusion Matrix on Real-world Unseen Data', fontsize=14, pad=15)
    
    plt.tight_layout()
    plt.savefig(CM_IMAGE_PATH, dpi=300) # 保存为高清图片
    print(f"\n📊 混淆矩阵图已生成并保存至: {CM_IMAGE_PATH}")
    print("你可以直接将此图片插入到你的 FYP Chapter 4 中。")

if __name__ == "__main__":
    main()
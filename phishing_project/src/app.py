# src/app.py
from flask import Flask, render_template, request
import joblib
from feature_extractor import extract_features, features_to_vector, FEATURE_ORDER
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "../templates"), static_folder=os.path.join(os.path.dirname(__file__), "../static"))

# 确保路径指向 .pkl 文件
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/phishing_rf.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Train the model first by running: python src/train.py")

m = joblib.load(MODEL_PATH)
clf = m['model']
feat_order = m['feature_order']

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    details = {}
    if request.method == "POST":
        url = request.form.get("url")
        if url:
            try:
                # 1. 提取特征
                feats = extract_features(url)
                vec = features_to_vector(feats)
                
                # 2. 获取预测概率
                # predict_proba 返回 [[prob_legit, prob_phish]]
                probs = clf.predict_proba([vec])[0]
                prob_phish = probs[1]  # 获取属于 Class 1 (Phishing) 的概率

                # 3. 判定逻辑
                if prob_phish > 0.5:
                    label = "PHISHING"
                else:
                    label = "LEGITIMATE"

                # 4. 结果打包
                # 无论结果如何，我们都返回 "prob_phish" 作为风险值
                result = {
                    'label': label,
                    'probability': float(prob_phish)
                }
                
                # 5. 提取特征详情用于展示
                details = {k: feats.get(k) for k in feat_order}
            except Exception as e:
                print(f"Error processing URL: {e}")
                result = {'label': "ERROR", 'probability': 0.0}

    return render_template("index.html", result=result, details=details)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
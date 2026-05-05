import joblib
import pandas as pd

# 加载你刚才训练好的模型
model_dict = joblib.load('../models/phishing_rf.pkl')
model = model_dict['model']
features = model_dict['feature_order']

# 获取特征重要性
importances = model.feature_importances_
df = pd.DataFrame({'Feature': features, 'Importance': importances})
df = df.sort_values(by='Importance', ascending=False)

print("🏆 模型最看重的前 15 个特征 (规则)：")
print(df.head(15))
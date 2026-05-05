# src/evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from feature_extractor import extract_features, features_to_vector, FEATURE_ORDER
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/phishing_rf.pkl")
TEST_CSV = os.path.join(os.path.dirname(__file__), "../data/test_dataset.csv")  # optional

if not os.path.exists(TEST_CSV):
    print("Place a test CSV at data/test_dataset.csv with columns: url,label")
else:
    data = pd.read_csv(TEST_CSV)
    m = joblib.load(MODEL_PATH)
    clf = m['model']
    feat_order = m['feature_order']

    rows = []
    for _, r in tqdm(data.iterrows(), total=len(data)):
        feats = extract_features(r['url'])
        vec = features_to_vector(feats)
        rows.append(vec)

    X = pd.DataFrame(rows, columns=feat_order)
    y_true = data['label'].values
    y_pred = clf.predict(X)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(5,4))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(os.path.dirname(__file__), "../models/confusion_matrix.png"))
    print("Saved confusion matrix to models/confusion_matrix.png")

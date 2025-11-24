# src/eval.py
import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from src.features import extract_features




def df_from_urls(url_series):
feats = [extract_features(u) for u in url_series]
return pd.DataFrame(feats)




def main():
model_path = os.path.join('models', 'rf_phish_pipeline.pkl')
if not os.path.exists(model_path):
raise FileNotFoundError('Model not found. Train first with src/train.py')


df = pd.read_csv(os.path.join('data', 'phishing_urls.csv'))
X = df_from_urls(df['url'])
y = df['label'].astype(int)


pipeline = joblib.load(model_path)
preds = pipeline.predict(X)
probs = pipeline.predict_proba(X)[:,1]


print('Classification report:')
print(classification_report(y, preds))
try:
print('AUC:', roc_auc_score(y, probs))
except Exception:
pass
print('Confusion matrix:')
print(confusion_matrix(y, preds))


if __name__ == '__main__':
main()

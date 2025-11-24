# src/train.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from src.features import extract_features




def df_from_urls(url_series):
feats = [extract_features(u) for u in url_series]
return pd.DataFrame(feats)




def main():
DATA_CSV = os.path.join('data', 'phishing_urls.csv')
if not os.path.exists(DATA_CSV):
raise FileNotFoundError(f"Dataset not found at {DATA_CSV}. Put a CSV with columns `url,label` there.")


df = pd.read_csv(DATA_CSV)
if 'url' not in df.columns or 'label' not in df.columns:
raise ValueError("CSV must contain 'url' and 'label' columns (label: 1=phish,0=legit)")


print(f"Loaded {len(df)} rows from {DATA_CSV}")


X = df_from_urls(df['url'])
y = df['label'].astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


pipeline = Pipeline([
('scaler', StandardScaler()),
('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])


print('Training model...')
pipeline.fit(X_train, y_train)


preds = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)[:, 1]


print('\nClassification report:')
print(classification_report(y_test, preds))
try:
auc = roc_auc_score(y_test, probs)
print('AUC:', auc)
except Exception:
print('AUC could not be computed')


os.makedirs('models', exist_ok=True)
model_path = os.path.join('models', 'rf_phish_pipeline.pkl')
joblib.dump(pipeline, model_path)
print(f'Saved model to {model_path}')




if __name__ == '__main__':
main()

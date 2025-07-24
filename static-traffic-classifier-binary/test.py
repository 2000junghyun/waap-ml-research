# 데이터 불러오기
import pandas as pd
df = pd.read_csv('data/train_data.csv')
df.head()

# 컬럼명 공백 제거
df.columns = df.columns.str.strip()
print(df.columns)

# EDA (데이터 탐색)
print("[+] df.shape")
print(df.shape)

print("\n[+] df.info()")
print(df.info())

print("\n[+] df.describe()")
print(df.describe())

print("\n[+] df.isnull().sum()")
print(df.isnull().sum())

df['Label'].value_counts()

df['Label_binary'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
df['Label_binary'].value_counts()

# Train/Test 분할
# X_train, X_test: inf/-inf→NaN→적당히 대체(0, 평균, 중앙값 등)
# y_train, y_test: NaN, inf 있는 행은 삭제(해당 샘플은 사용불가)

import numpy as np
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Label', 'Label_binary'])  # 입력값만 남김
y = df['Label_binary']  # 정답(타겟)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

for df in [X_train, X_test]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

for X, y in [(X_train, y_train), (X_test, y_test)]:
    mask = y.notnull() & ~np.isinf(y)
    X = X[mask]
    y = y[mask]

# 모델 학습 및 평가 (RandomForest 사용)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 예측값 구하기
y_pred = clf.predict(X_test)

# 개별 지표 계산
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print("\nConfusion Matrix:\n", cm)

# 더 자세한 리포트(클래스별 지표)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4))
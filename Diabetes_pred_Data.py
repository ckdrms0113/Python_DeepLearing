import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터셋 로드 및 결측치 확인
data_url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(data_url)
print("결측치 확인:")
print(df.isnull().sum())

# 2. 데이터 전처리 (MinMaxScaler 적용)
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. 데이터 분할 (학습 70%, 테스트 30%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. 로지스틱 회귀 모델 학습 및 평가
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# 5. SVM 모델 학습 및 평가
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# 6. 결정 트리 모델 학습 및 평가
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

# 7. CNN 모델 학습 및 평가[심화]
# CNN 모델을 위해 데이터를 3차원으로 변환
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)
y_train_cnn = to_categorical(y_train)
y_test_cnn = to_categorical(y_test)

cnn = Sequential([
    Conv1D(32, 2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(2, activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = cnn.fit(X_train_cnn, y_train_cnn, epochs=5, batch_size=16, validation_data=(X_test_cnn, y_test_cnn), verbose=1)

y_pred_cnn = cnn.predict(X_test_cnn)
y_pred_cnn_classes = np.argmax(y_pred_cnn, axis=1)

# 8. 평가지표 출력 - 차트 시각화[심화]
def print_metrics(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{model_name} 성능 지표:")
    print(f"정확도: {accuracy}")
    print(f"정밀도: {precision}")
    print(f"재현율: {recall}")
    print(f"F1 스코어: {f1}")
    print()
    return accuracy, precision, recall, f1

log_reg_metrics = print_metrics(y_test, y_pred_log_reg, "로지스틱 회귀")
svm_metrics = print_metrics(y_test, y_pred_svm, "SVM")
tree_metrics = print_metrics(y_test, y_pred_tree, "결정 트리")
cnn_metrics = print_metrics(y_test, y_pred_cnn_classes, "CNN")

# CNN 학습 과정 시각화[[심화]
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('CNN Model Accuracy')
plt.show()

# 모델 비교 차트 시각화[심화]
models = ['Logistic Regression', 'SVM', 'Decision Tree', 'CNN']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
log_reg_metrics = list(log_reg_metrics)
svm_metrics = list(svm_metrics)
tree_metrics = list(tree_metrics)
cnn_metrics = list(cnn_metrics)

data = [log_reg_metrics, svm_metrics, tree_metrics, cnn_metrics]
data = np.array(data).T

x = np.arange(len(metrics))
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5*width, data[:, 0], width, label='Logistic Regression')
rects2 = ax.bar(x - 0.5*width, data[:, 1], width, label='SVM')
rects3 = ax.bar(x + 0.5*width, data[:, 2], width, label='Decision Tree')
rects4 = ax.bar(x + 1.5*width, data[:, 3], width, label='CNN')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.show()

# 결정 트리 중요 특성 시각화
importances = tree.feature_importances_
features = df.columns[:-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title('Feature Importances in Decision Tree')
plt.show()

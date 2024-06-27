import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, LSTM

# 데이터 로드 및 전처리
max_features = 10000  # 사용할 단어의 최대 개수
maxlen = 200  # 시퀀스의 최대 길이

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# 모델 구성
model = Sequential([
    Embedding(max_features, 128, input_length=maxlen),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 요약 정보 출력
model.summary()

# 모델 학습
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# 모델 평가
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# 예측 결과 확인
import numpy as np
predictions = model.predict(X_test[:5])
print("Predictions:", predictions.flatten().round())
print("Actual labels:", y_test[:5])

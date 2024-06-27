import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Opencv를 이용하여 이미지 로드 및 전처리
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Opencv는 BGR 형식으로 이미지를 읽으므로 RGB로 변환
    img = cv2.resize(img, (150, 150))  # 이미지 크기를 150x150 픽셀로 조정
    img = img / 255.0  # 이미지를 0에서 1 사이의 값으로 정규화
    return img

# 이미지 경로
image_path = 'test.jpg'
# Opencv를 사용하여 이미지 로드 및 전처리
image = load_and_preprocess_image(image_path)
# 이미지를 배치 형태로 변환
image = np.expand_dims(image, axis=0)

# CNN 모델 구성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 요약 정보 출력
model.summary()

# 예제 데이터 생성
X_example = np.random.rand(100, 150, 150, 3)  # 랜덤 이미지 데이터 생성
y_example = np.random.randint(0, 2, size=100)  # 랜덤 레이블 데이터 생성

# 모델 학습 (예제 데이터 사용)
model.fit(X_example, y_example, epochs=5, batch_size=32)

# 이미지 분류 예측
predictions = model.predict(image)
print("Predictions:", predictions)

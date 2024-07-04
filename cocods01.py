# 필요한 라이브러리 설치
!pip install pycocotools
!pip install tensorflow

import os
import urllib.request
import zipfile
from pycocotools.coco import COCO
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# 데이터셋 다운로드 경로 설정
dataDir = '/content/CocoSet'
if not os.path.exists(dataDir):
    os.makedirs(dataDir)

# COCO 2017 Train 이미지 다운로드
train_images_url = 'http://images.cocodataset.org/zips/train2017.zip'
train_images_path = os.path.join(dataDir, 'train2017.zip')
if not os.path.exists(train_images_path):
    urllib.request.urlretrieve(train_images_url, train_images_path)

# COCO 2017 Train 주석 다운로드
annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
annotations_path = os.path.join(dataDir, 'annotations_trainval2017.zip')
if not os.path.exists(annotations_path):
    urllib.request.urlretrieve(annotations_url, annotations_path)

# 압축 해제
with zipfile.ZipFile(train_images_path, 'r') as zip_ref:
    zip_ref.extractall(dataDir)

with zipfile.ZipFile(annotations_path, 'r') as zip_ref:
    zip_ref.extractall(dataDir)

# 주석 파일 경로 설정
annFile = f'{dataDir}/annotations/instances_train2017.json'

# COCO API 초기화
coco = COCO(annFile)

# 카테고리와 이미지 ID 가져오기
catIds = coco.getCatIds(catNms=['person', 'dog', 'cat'])
imgIds = coco.getImgIds(catIds=catIds)

# 이미지 전처리 함수
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(np.array(image), target_size)
    image = image / 255.0
    return image

# 데이터 준비
images = []
labels = []

for imgId in imgIds[:100]:  # 예제에서는 100개의 이미지만 사용
    img_info = coco.loadImgs(imgId)[0]
    img_path = os.path.join(dataDir, 'train2017', img_info['file_name'])
    img = Image.open(img_path).convert('RGB')
    images.append(preprocess_image(img))

    annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    label = np.zeros(len(catIds))
    for ann in anns:
        cat_index = catIds.index(ann['category_id'])
        label[cat_index] = 1
    labels.append(label)

# 넘파이 배열로 변환
X = np.array(images)
y = np.array(labels)

# 데이터셋 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 구성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(catIds), activation='sigmoid')  # 다중 라벨 분류를 위해 sigmoid 사용
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 모델 학습
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# 모델 평가 및 추론
def predict_image(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(np.array([preprocessed_image]))
    predicted_class = np.argmax(prediction)
    return coco.loadCats(catIds[predicted_class])[0]['name']

# 예제 이미지 추론
test_img_info = coco.loadImgs(imgIds[10])[0]  # 10번째 이미지를 사용하도록 수정
test_img_path = os.path.join(dataDir, 'train2017', test_img_info['file_name'])
test_img = Image.open(test_img_path).convert('RGB')

predicted_label = predict_image(test_img)
print(f'Predicted class: {predicted_label}')

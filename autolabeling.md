# YOLOv8 기반 머리 감지 및 바운딩 박스 시각화

## 개요
이 프로젝트는 **YOLOv8**을 활용하여 이미지에서 사람의 머리를 감지하고, 해당 영역을 시각적으로 표시한 후 저장하는 Python 스크립트입니다. 감지된 바운딩 박스 정보는 별도의 텍스트 파일로 저장됩니다.

## 환경 설정

### 요구 사항
이 코드를 실행하기 위해서는 다음과 같은 패키지가 필요합니다:

- Python 3.x
- OpenCV (`cv2`)
- `ultralytics` 라이브러리 (YOLOv8 사용)
- 기타 필수 라이브러리 (`os`, `time`)

### 설치 방법
아래 명령어를 실행하여 필요한 패키지를 설치할 수 있습니다:

```bash
pip install ultralytics opencv-python
```

## 실행 방법

### 1. 사전 준비
- `best.pt` 모델 가중치 파일을 다운로드하여 `models/weights/` 폴더에 저장합니다.
- 감지할 이미지 파일을 `images/` 폴더에 배치합니다.

### 2. 코드 실행
아래 명령어를 실행하면 YOLOv8을 사용하여 이미지 내 머리를 감지하고, 결과를 저장합니다:

```bash
python script.py
```

## 코드 설명

### 1. 모델 로드 및 폴더 생성
```python
weight_path = r"D:\Downloads\Head-Detection-Yolov8-main\Head-Detection-Yolov8-main\models\weights\best.pt"
model = YOLO(weight_path)
```
- YOLOv8 모델을 지정된 경로에서 불러옵니다.
- 감지 결과를 저장할 `label_image` 및 `information` 폴더를 생성합니다.

### 2. 이미지 처리 및 객체 감지
```python
for image_name in os.listdir(image_folder):
    image = cv2.imread(image_path)
    results = model(image)
```
- `images/` 폴더 내의 모든 이미지를 읽어 YOLOv8을 사용해 분석합니다.

### 3. 감지된 바운딩 박스 정보 저장
```python
with open(info_file_path, "w") as info_file:
    for result in results:
        boxes = result.boxes.xyxy  # 바운딩 박스 좌표
        confidences = result.boxes.conf  # 신뢰도 점수
```
- 감지된 객체의 바운딩 박스 좌표 및 신뢰도를 `information/` 폴더의 텍스트 파일로 저장합니다.

### 4. 감지된 영역 시각화 및 저장
```python
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite(labeled_image_path, image)
```
- 감지된 영역에 녹색 박스를 그린 후, `label_image/` 폴더에 저장합니다.

## 결과물
코드 실행 후 다음과 같은 파일이 생성됩니다:

- `label_image/`: 감지된 영역이 표시된 이미지 저장
- `information/`: 감지된 바운딩 박스 좌표를 담은 `.txt` 파일

## 실행 시간
전체 프로세스의 실행 시간은 다음과 같이 측정됩니다:
```python
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds")
```

## 라이선스
이 프로젝트는 공개된 YOLOv8 모델을 사용하며, 해당 라이선스를 따릅니다.


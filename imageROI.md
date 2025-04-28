# YOLOv5를 활용한 이미지 내 사람 감지 및 분류

## 개요
이 프로젝트는 YOLOv5를 활용하여 이미지에서 사람을 감지하고, 감지된 이미지와 감지되지 않은 이미지를 각각 다른 디렉토리에 분류하는 코드입니다.

## 요구 사항
- Python 3.x
- torch
- ultralytics/yolov5
- OpenCV
- Pillow
- NumPy

## 설치 방법
아래 명령어를 실행하여 필요한 라이브러리를 설치합니다.
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python pillow numpy
```

## 코드 설명
### 1. 디렉토리 설정
사용자가 지정한 이미지 폴더에서 사람 감지 여부에 따라 결과를 분류할 폴더를 설정합니다.
```python
image_folder = r'E:\대구 테크비즈센터\2025-01-02\image8'  # 원본 이미지 디렉토리
person_detected_folder = r'E:\대구 테크비즈센터\2025-01-02\image8\Soure01'  # 사람이 감지된 이미지 디렉토리
no_person_folder = r'E:\대구 테크비즈센터\2025-01-02\image8\Dummy'  # 사람이 감지되지 않은 이미지 디렉토리
```

### 2. YOLOv5 모델 로드
YOLOv5 모델을 로드하여 객체 감지를 수행합니다.
```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
```

### 3. ROI 설정
우측 하단 25%를 제외한 영역을 ROI(Region of Interest)로 설정하여 분석합니다.
```python
def get_roi(image):
    width, height = image.size
    right_x = int(width * 0.75)
    return 0, right_x
```

### 4. 이미지 처리 및 분류
이미지를 하나씩 읽어 ROI 영역에서 YOLOv5를 통해 사람을 감지하고, 감지 여부에 따라 폴더로 이동합니다.
```python
for image_name in os.listdir(image_folder):
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        roi_x1, roi_x2 = get_roi(image)
        image_array = np.array(image)
        roi_image = image_array[:, roi_x1:roi_x2]
        results = model(roi_image)
        boxes = results.xyxy[0]
        person_detected = any(int(box[5]) == 0 and box[4] >= 0.4 for box in boxes)
        target_folder = person_detected_folder if person_detected else no_person_folder
        shutil.move(image_path, os.path.join(target_folder, image_name))
        print(f"{'[감지O]' if person_detected else '[감지X]'} {image_name} → {target_folder}")
```

## 실행 방법
Python 스크립트를 실행하면 지정된 폴더에서 이미지를 자동으로 분류합니다.
```sh
python detect_person.py
```

## 결과
- 사람이 감지된 이미지는 `Soure01` 폴더로 이동
- 사람이 감지되지 않은 이미지는 `Dummy` 폴더로 이동

## 참고 사항
- YOLOv5의 성능을 최적화하려면 모델의 가중치를 사용자 데이터셋으로 재학습하는 것이 좋습니다.
- ROI 영역을 조정하여 특정 부분만 분석하도록 수정할 수 있습니다.


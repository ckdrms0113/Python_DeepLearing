import os
import shutil
import torch
from PIL import Image
import numpy as np
import cv2

# 디렉토리 설정 (사용자가 지정)
image_folder = r'E:\대구 테크비즈센터\2025-01-02\image8'  # 원본 이미지 디렉토리
person_detected_folder = r'E:\대구 테크비즈센터\2025-01-02\image8\Soure01'  # 사람이 감지된 이미지를 저장할 디렉토리
no_person_folder = r'E:\대구 테크비즈센터\2025-01-02\image8\Dummy'  # 사람이 감지되지 않은 이미지를 저장할 디렉토리

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# YOLO 클래스 ID와 임계값 설정
person_class_id = 0  # 'person' 클래스 ID
confidence_threshold = 0.4  # 신뢰도 임계값

# ROI 범위 설정 함수 (우측 하단을 제외한 영역)
# ROI 범위 설정 (우측 하단의 x값만 제외한 영역)
def get_roi(image):
    width, height = image.size
    
    # 우측 25%를 제외한 x 범위
    right_x = int(width * 0.75)  # 우측 25%를 제외
    return 0, right_x  # ROI 범위 반환 (x_min, x_max)
##########
# 이미지 처리
for image_name in os.listdir(image_folder):
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        # ROI 범위 얻기
        roi_x1, roi_x2 = get_roi(image)
        
        # ROI 영역만 적용 (y값 전체 사용)
        image_array = np.array(image)
        roi_image = image_array[:, roi_x1:roi_x2]  # y값 전체, x값만 자름

        # ROI 경계선 그리기 (x축만 표시)
        cv2.rectangle(image_array, (roi_x1, 0), (roi_x2, image_array.shape[0]), (150, 255, 90), 3)

        # ROI 경계가 그려진 이미지를 확인 (디버그용)
        #roi_image_with_border = Image.fromarray(image_array)
        #roi_image_with_border.show()

        # YOLO 모델을 사용하여 객체 감지 (ROI 영역만 적용)
        results = model(roi_image)

        # 바운딩 박스 정보 추출
        boxes = results.xyxy[0]  # x1, y1, x2, y2, confidence, class

        # 감지된 객체에 대한 결과 출력
        person_detected = False
        for box in boxes:
            if int(box[5]) == person_class_id and box[4] >= confidence_threshold:  # 'person' 클래스 ID가 0, 신뢰도 임계값 0.4
                person_detected = True
                break
        
        # 감지 여부에 따라 다른 디렉토리로 이동
        if person_detected:
            shutil.move(image_path, os.path.join(person_detected_folder, image_name))
            print(f"[감지O] {image_name} → {person_detected_folder}")
        else:
            shutil.move(image_path, os.path.join(no_person_folder, image_name))
            print(f"[감지X] {image_name} → {no_person_folder}")

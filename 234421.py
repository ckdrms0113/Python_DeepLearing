import os
import shutil
import torch
from PIL import Image
import numpy as np
import cv2

# 디렉토리 설정 (사용자가 지정)
image_folder = r'D:\Desktop\images'  # 원본 이미지 디렉토리
person_detected_folder = r'D:\Desktop\Yes'  # 사람이 감지된 이미지를 저장할 디렉토리
no_person_folder = r'D:\Desktop\NO'  # 사람이 감지되지 않은 이미지를 저장할 디렉토리

# ROI 조절 파라미터 (비율 입력, 0.0 ~ 1.0)
roi_top = 0.3   # 상단 25% 제외
roi_bottom = 0.0 # 하단 제외 없음
roi_left = 0.0   # 좌측 제외 없음
roi_right = 0.0 # 우측 25% 제외

#----------------------------------------------------------------#
#----------------------------------------------------------------#
#----------------------------------------------------------------#
#개발자 제외 아래 코드 변경 금지

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# YOLO 클래스 ID와 임계값 설정
person_class_id = 0  # 'person' 클래스 ID
confidence_threshold = 0.4  # 신뢰도 임계값

# ROI 범위 설정 함수 (사용자 지정 비율 적용)
def get_roi(image):
    width, height = image.size
    
    # ROI 계산
    top_y = int(height * roi_top)
    bottom_y = height - int(height * roi_bottom)
    left_x = int(width * roi_left)
    right_x = width - int(width * roi_right)
    
    return left_x, right_x, top_y, bottom_y  # ROI 범위 반환

# 이미지 처리
for image_name in os.listdir(image_folder):
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        # ROI 범위 얻기
        roi_x1, roi_x2, roi_y1, roi_y2 = get_roi(image)
        
        # ROI 영역만 적용
        image_array = np.array(image)
        roi_image = image_array[roi_y1:roi_y2, roi_x1:roi_x2]  # 잘라낸 영역 적용

        # ROI 경계선 그리기
        cv2.rectangle(image_array, (roi_x1, roi_y1), (roi_x2, roi_y2), (150, 255, 90), 3)

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

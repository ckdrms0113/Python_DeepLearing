from ultralytics import YOLO
import os
import cv2
import time
import urllib.request
from urllib.parse import urlparse

# 시작 시간 기록   ##### 작업 시간 측정 용도
#start_time = time.time()

################# URL 기반 폴더 위치 및 생성 폴더 이름 ##########################
# 모델 경로
weight_path = r"C:\\Users\\HOME\\Desktop\\Head-Detection-Yolov8-main\\models\\weights\\best.pt"

# Load a model
model = YOLO(weight_path)  # load a pretrained model (recommended for training)

# 입력 이미지 URL 폴더
image_folder_url = "https://example.com/images/"

# 로컬 저장 폴더 (URL에서 이미지를 다운로드하여 저장할 위치)
local_image_folder = "downloaded_images"
os.makedirs(local_image_folder, exist_ok=True)

# 라벨링된 이미지 저장 URL 폴더 (로컬 저장 경로와 동일하게 사용)
label_image_folder = "label_image"
os.makedirs(label_image_folder, exist_ok=True)

# 바운딩 박스 정보 저장 URL 폴더 (로컬 저장 경로와 동일하게 사용)
information_folder = "information"
os.makedirs(information_folder, exist_ok=True)

################ 모델 박스 크기 설정 및 탐지 기준#######################################
### 최소 박스 탐지 크기 
##### 현재 가로 30px 세로 30px 이상 크기 박스만 탐지 가능
box_width = 10
box_height = 10

###### 탐지되는 신뢰도 기준   
#### 0부터 1까지 값
box_conf = 0.2

#####################################################################################

def download_image(image_url, save_folder):
    """이미지 URL을 다운로드하여 로컬에 저장"""
    parsed_url = urlparse(image_url)
    image_name = os.path.basename(parsed_url.path)
    local_path = os.path.join(save_folder, image_name)
    urllib.request.urlretrieve(image_url, local_path)
    return local_path

# URL에서 이미지 파일 이름 가져오기
image_urls = [
    f"{image_folder_url}image1.jpg",
    f"{image_folder_url}image2.jpg",
    f"{image_folder_url}image3.jpg"
]

# 이미지 처리
for image_url in image_urls:
    try:
        # 이미지 다운로드
        image_path = download_image(image_url, local_image_folder)

        # 이미지 읽기
        image = cv2.imread(image_path)

        # 모델 추론
        results = model(image)

        # 결과 파일 경로 설정
        image_name = os.path.basename(image_path)
        labeled_image_path = os.path.join(label_image_folder, image_name)
        info_file_path = os.path.join(information_folder, f"{os.path.splitext(image_name)[0]}.txt")

        # 바운딩 박스 정보 저장
        with open(info_file_path, "w") as info_file:
            for result in results:
                boxes = result.boxes.xyxy  # 바운딩 박스 좌표 (x1, y1, x2, y2)
                confidences = result.boxes.conf  # 신뢰도 점수

                for box, confidence in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    confidence = float(confidence)

                    # 바운딩 박스 크기 필터링
                    if (x2 - x1 > box_width) and (y2 - y1 > box_height) and confidence >= box_conf:
                        # 바운딩 박스 정보 파일에 저장
                        info_file.write(f"{x1} {y1} {x2} {y2}\n")

                        # 이미지에 바운딩 박스 그리기
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 라벨링된 이미지 저장
        cv2.imwrite(labeled_image_path, image)
        print(f"Processed and saved: {labeled_image_path}")
    except Exception as e:
        print(f"Error processing {image_url}: {e}")

# 총 실행 시간 계산 및 출력
#end_time = time.time()
#execution_time = end_time - start_time
#print(f"All images have been processed and labeled.")
#print(f"Total execution time: {execution_time:.2f} seconds")

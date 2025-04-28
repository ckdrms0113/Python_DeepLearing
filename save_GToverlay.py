import os
import cv2
import numpy as np
from glob import glob

# ✅ GT 라벨 및 이미지 경로 설정
image_dir = r"D:\Desktop\ddd\GwangAn2.0\images\train"  # 원본 이미지 폴더
label_dir = r"D:\Desktop\ddd\1234\labels\train"  # GT 라벨 폴더
output_dir = r"D:\Desktop\ddd\imgs"  # GT 바운딩 박스 오버레이된 이미지 저장 폴더

# 폴더 생성 (존재하지 않으면 생성)
os.makedirs(output_dir, exist_ok=True)

# 평가할 이미지 리스트 불러오기
image_paths = glob(os.path.join(image_dir, "*.jpg"))

# 🔹 GT 바운딩 박스를 이미지에 오버레이하는 함수
def draw_gt_boxes(image, gt_boxes):
    for box in gt_boxes:
        x1, y1, x2, y2 = box[:4]  # 바운딩 박스 좌표 (픽셀 단위)
        class_id = int(box[4])  # 클래스 ID

        # GT 박스는 빨간색
        color = (0, 0, 255)  # 빨간색
        label = f"GT {class_id}"

        # 바운딩 박스 및 라벨 추가
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

# ✅ 모든 이미지 처리
for image_path in image_paths:
    # 이미지 파일명 가져오기
    image_name = os.path.basename(image_path)
    label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))  # GT 라벨 파일 경로
    output_path = os.path.join(output_dir, image_name)  # 저장 경로

    # 이미지 로드
    image = cv2.imread(image_path)
    height, width = image.shape[:2]  # 원본 이미지 크기 가져오기

    # ✅ GT 파일이 존재하는지 확인 후 로드
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]  # 빈 줄 제거

        for line in lines:
            values = line.split()
            if len(values) != 5:
                print(f"🚨 Warning: Invalid label format in {label_path} → {line}")
                continue  # 잘못된 데이터 무시

            class_id, x_center, y_center, w, h = map(float, values)

            # YOLO 포맷 → 픽셀 좌표 변환
            x1 = round((x_center - w / 2) * width)
            y1 = round((y_center - h / 2) * height)
            x2 = round((x_center + w / 2) * width)
            y2 = round((y_center + h / 2) * height)

            # GT 바운딩 박스 추가
            gt_boxes.append([x1, y1, x2, y2, class_id])

    else:
        print(f"🚨 Warning: No GT file found for {image_path}")
        continue  # GT 파일 없으면 해당 이미지 스킵

    # ✅ GT 바운딩 박스 오버레이
    image_with_gt = draw_gt_boxes(image.copy(), gt_boxes)

    # ✅ 결과 이미지 저장
    cv2.imwrite(output_path, image_with_gt)
    print(f"✅ 저장 완료: {output_path}")

print("\n🚀 모든 이미지에 대한 GT 바운딩 박스 오버레이 완료!")

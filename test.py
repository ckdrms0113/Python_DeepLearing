import cv2
import os

def draw_yolo_boxes(image_path, label_path, class_names=None):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # 잘못된 라벨 무시

        class_id, x_center, y_center, box_width, box_height = map(float, parts)

        # YOLO 좌표를 픽셀 좌표로 변환
        x1 = int((x_center - box_width / 2) * w)
        y1 = int((y_center - box_height / 2) * h)
        x2 = int((x_center + box_width / 2) * w)
        y2 = int((y_center + box_height / 2) * h)

        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 클래스 이름 표시
        if class_names:
            label = class_names[int(class_id)]
        else:
            label = f"{int(class_id)}"

        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 이미지 보여주기
    cv2.imshow("YOLO Label Overlay", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 예시 사용
image_path = "202411_GwangAnSubway_81-4-5_frame_14186.jpg"
label_path = "202411_GwangAnSubway_81-4-5_frame_14186.txt"
draw_yolo_boxes(image_path, label_path)

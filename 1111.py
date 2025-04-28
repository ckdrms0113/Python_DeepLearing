import os
import glob
import time
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def visualize_yolo_labels(image_dir, label_dir):
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    label_paths = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob.glob(os.path.join(label_dir, "*.txt"))
    }

    paired_files = [
        (img_path, label_paths.get(os.path.splitext(os.path.basename(img_path))[0]))
        for img_path in image_paths
        if os.path.splitext(os.path.basename(img_path))[0] in label_paths
    ]

    if not paired_files:
        print("⚠️ 유효한 이미지-라벨 쌍이 없습니다.")
        return

    for img_path, txt_path in paired_files:
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size

        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, box_width, box_height = map(float, parts)
                x1 = (x_center - box_width / 2) * width
                y1 = (y_center - box_height / 2) * height
                x2 = (x_center + box_width / 2) * width
                y2 = (y_center + box_height / 2) * height
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"{os.path.basename(img_path)} - 5초 후 다음 이미지로 이동")
        plt.show(block=False)
        plt.pause(1)
        plt.close()

# ============================
# 사용 시 아래 경로를 수정하세요
# ============================
if __name__ == "__main__":
    image_folder = r"D:\Desktop\ddd\810405test01\images\train"
    label_folder = r"D:\Desktop\ddd\810405test04\labels\train"
    visualize_yolo_labels(image_folder, label_folder)

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2

def create_annotation_xml(all_annotations, output_path):
    annotations = ET.Element("annotations")

    for idx, (image_id, image_name, image_path, bounding_boxes) in enumerate(all_annotations):
        # 이미지 크기 가져오기
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            continue
        im_height, im_width = image.shape[:2]

        image_element = ET.SubElement(annotations, "image", {
            "id": str(image_id),
            "name": image_name,
            "width": str(im_width),
            "height": str(im_height)
        })

        for box in bounding_boxes:
            label, x1, y1, x2, y2 = box

            ET.SubElement(image_element, "box", {
                "label": "male" if label == "0" else "Person",
                "source": "manual",
                "occluded": "0",
                "xtl": f"{x1:.2f}",
                "ytl": f"{y1:.2f}",
                "xbr": f"{x2:.2f}",
                "ybr": f"{y2:.2f}",
                "z_order": "0"
            })

    xml_str = minidom.parseString(ET.tostring(annotations)).toprettyxml(indent="  ")
    with open(output_path, "w") as f:
        f.write(xml_str)

def parse_txt_file(txt_file):
    bounding_boxes = []
    with open(txt_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            label = parts[0]
            x1 = float(parts[1])
            y1 = float(parts[2])
            x2 = float(parts[3])
            y2 = float(parts[4])
            bounding_boxes.append((label, x1, y1, x2, y2))
    return bounding_boxes

def main():
    label_directory = r"C:\Users\HOME\Desktop\job_900_dataset_2025_03_31_05_47_56_cvat for images 1.1\imformation"
    image_folder = r"C:\Users\HOME\Desktop\Head-Detection-Yolov8-main\81-1"
    output_file = "annotations.xml"

    all_annotations = []

    for txt_file in os.listdir(label_directory):
        if txt_file.endswith(".txt"):
            image_id = txt_file.split(".")[0]
            image_name = f"{image_id}.jpg"
            txt_path = os.path.join(label_directory, txt_file)
            image_path = os.path.join(image_folder, image_name)

            if not os.path.exists(image_path):
                print(f"⚠️ 이미지 없음: {image_path}")
                continue

            bounding_boxes = parse_txt_file(txt_path)
            all_annotations.append((image_id, image_name, image_path, bounding_boxes))

    create_annotation_xml(all_annotations, output_file)
    print(f"✅ 어노테이션 XML 저장 완료: {output_file}")

if __name__ == "__main__":
    main()

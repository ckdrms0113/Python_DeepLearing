import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

# 이미지 크기 설정
im_width = 928
im_height = 480

def create_annotation_xml(all_annotations, width, height, output_path):
    # 루트 엘리먼트 생성
    annotations = ET.Element("annotations")

    # 이미지 정보 및 바운딩 박스 추가
    for idx, (image_id, image_name, bounding_boxes) in enumerate(all_annotations):
        # 이미지 ID에서 숫자 추출
        numeric_id = image_id.split('_')[-1] if '_' in image_id else image_id

        # 이미지 엘리먼트 생성
        image_element = ET.SubElement(annotations, "image", {
            "id": str(numeric_id),
            "name": image_name,
            "width": str(width),
            "height": str(height)
        })

        # 바운딩 박스 추가
        for box in bounding_boxes:
            label, x1, y1, x2, y2 = box

            # 박스 엘리먼트 생성
            ET.SubElement(image_element, "box", {
                "label": "Person" if label == "0" else "bicycle",  # "0" -> Person, 그 외 -> bicycle
                "source": "manual",
                "occluded": "0",
                "xtl": f"{x1:.2f}",
                "ytl": f"{y1:.2f}",
                "xbr": f"{x2:.2f}",
                "ybr": f"{y2:.2f}",
                "z_order": "0"
            })

    # XML을 예쁘게 출력
    xml_str = minidom.parseString(ET.tostring(annotations)).toprettyxml(indent="  ")
    with open(output_path, "w") as f:
        f.write(xml_str)

def parse_txt_file(txt_file, width, height):
    bounding_boxes = []
    with open(txt_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            label = "0"  # 모든 레이블을 "0"으로 가정
            x1 = float(parts[0])
            y1 = float(parts[1])
            x2 = float(parts[2])
            y2 = float(parts[3])
            bounding_boxes.append((label, x1, y1, x2, y2))
    return bounding_boxes

def main():
    # 설정
    label_directory = r"C:\Users\HOME\Desktop\Head-Detection-Yolov8-main\information"
    image_width = im_width
    image_height = im_height
    output_file = "annotations.xml"  # 출력 파일 이름 변경

    # 각 텍스트 파일 처리
    all_annotations = []
    for txt_file in os.listdir(label_directory):
        if txt_file.endswith(".txt"):
            image_id = txt_file.split(".")[0]
            image_name = f"{image_id}.jpg"
            txt_path = os.path.join(label_directory, txt_file)
            bounding_boxes = parse_txt_file(txt_path, image_width, image_height)
            all_annotations.append((image_id, image_name, bounding_boxes))

    # 모든 어노테이션을 하나의 XML로 결합
    create_annotation_xml(all_annotations, image_width, image_height, output_file)

if __name__ == "__main__":
    main()

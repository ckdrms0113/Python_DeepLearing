import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString

# 이미지 크기 (필요 시 이미지별 처리로 확장 가능)
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

def create_pascal_voc_xml(image_filename, annotations, output_dir):
    annotation = ET.Element('annotation')

    ET.SubElement(annotation, 'folder').text = os.path.basename(output_dir)
    ET.SubElement(annotation, 'filename').text = image_filename
    ET.SubElement(annotation, 'path').text = os.path.join(output_dir, image_filename)

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(IMAGE_WIDTH)
    ET.SubElement(size, 'height').text = str(IMAGE_HEIGHT)
    ET.SubElement(size, 'depth').text = '3'

    ET.SubElement(annotation, 'segmented').text = '0'

    for ann in annotations:
        class_name, xmin, ymin, xmax, ymax = ann

        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = class_name
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(xmin))
        ET.SubElement(bndbox, 'ymin').text = str(int(ymin))
        ET.SubElement(bndbox, 'xmax').text = str(int(xmax))
        ET.SubElement(bndbox, 'ymax').text = str(int(ymax))

    xml_str = ET.tostring(annotation, encoding='utf-8')
    pretty_xml = parseString(xml_str).toprettyxml(indent="  ")

    xml_path = os.path.join(output_dir, os.path.splitext(image_filename)[0] + '.xml')
    with open(xml_path, 'w') as f:
        f.write(pretty_xml)

def convert_custom_txts_to_xml(txt_dir, output_dir, image_ext='.jpg'):
    os.makedirs(output_dir, exist_ok=True)

    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.txt'):
            annotations = []
            txt_path = os.path.join(txt_dir, txt_file)

            with open(txt_path, 'r') as f:
                lines = f.read().strip().splitlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 6:
                    class_name = parts[0]
                    xmin = float(parts[1])
                    ymin = float(parts[2])
                    xmax = float(parts[3])
                    ymax = float(parts[4])
                    # confidence = float(parts[5]) → 사용 안 함
                    annotations.append((class_name, xmin, ymin, xmax, ymax))

            image_filename = os.path.splitext(txt_file)[0] + image_ext
            create_pascal_voc_xml(image_filename, annotations, output_dir)

# 🔽 여기에 명확하게 디렉토리 지정 🔽
if __name__ == "__main__":
    # 입력 txt 파일이 있는 디렉토리 경로
    txt_input_dir = r'D:\Desktop\930\930\anntations'

    # 결과 XML을 저장할 디렉토리 경로
    xml_output_dir = r'D:\Desktop\930\930\xml'

    # 이미지 확장자 (.jpg, .png 등)
    image_extension = '.jpg'

    # 변환 실행
    convert_custom_txts_to_xml(txt_input_dir, xml_output_dir, image_extension)

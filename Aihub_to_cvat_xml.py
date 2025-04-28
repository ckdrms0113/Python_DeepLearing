import xml.etree.ElementTree as ET
import os

# 통합된 XML 파일 경로
input_annotation_file = r"D:\Desktop\xmls\annotations.xml"

# 출력 디렉토리 설정 (개별 XML 파일 저장용)
output_dir = r"D:\Desktop\xmls"
os.makedirs(output_dir, exist_ok=True)

# XML 파싱
tree = ET.parse(input_annotation_file)
root = tree.getroot()

# 이미지별로 분리
for image in root.findall("image"):
    image_name = image.get("name")
    annotation = ET.Element("annotation")
    
    # 이미지 정보 추가
    folder = ET.SubElement(annotation, "folder")
    folder.text = "images"
    
    filename = ET.SubElement(annotation, "filename")
    filename.text = image_name
    
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = image.get("width")
    
    height = ET.SubElement(size, "height")
    height.text = image.get("height")
    
    depth = ET.SubElement(size, "depth")
    depth.text = "3"  # RGB 기본값
    
    # Object 정보 추가
    for box in image.findall("box"):
        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = box.get("label")
        
        # 'pose' 태그 추가
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"  # pose 태그 추가 (선택 사항)
        
        # 'bndbox'는 좌표를 포함하는 부분
        bndbox = ET.SubElement(obj, "bndbox")
        
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = box.get("xtl")
        
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = box.get("ytl")
        
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = box.get("xbr")
        
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = box.get("ybr")
        
        # Occlusion 추가 (Pascal VOC에는 옵션으로 포함됨)
        occluded = ET.SubElement(obj, "occluded")
        occluded.text = box.get("occluded", "0")  # 기본값은 "0"
        
        # Difficult 항목 추가 (대부분 "0" 또는 "1" 값)
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"  # 기본값은 "0"
    
    # 개별 XML 파일 저장
    output_file = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.xml")
    tree = ET.ElementTree(annotation)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

print(f"개별 XML 파일이 {output_dir} 디렉토리에 저장되었습니다.")

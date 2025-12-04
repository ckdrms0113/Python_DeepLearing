import os

# 바탕화면 경로
desktop_path = os.path.join(os.path.expanduser("~"), "바탕 화면")

# 삭제할 이미지 확장자 목록
image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff']

# 바탕화면에서 'GwangAn'이 포함된 이미지 파일 삭제
for filename in os.listdir(desktop_path):
    # 확장자 확인
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        if 'GwangAn' in filename:
            file_path = os.path.join(desktop_path, filename)
            try:
                os.remove(file_path)
                print(f"삭제됨: {file_path}")
            except Exception as e:
                print(f"삭제 실패: {file_path} - {e}")

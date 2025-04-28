import os
import glob
import random
import tkinter as tk
from tkinter import filedialog, messagebox
import shutil

class PreprocessingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("전처리 + 셔플러 통합 툴")
        self.root.geometry("700x780")
        self.root.config(bg="#f4f6f9")

        self.image_folder_path = ""
        self.label_folder_path = ""

        self.setup_ui()

    def setup_ui(self):
        frame = tk.Frame(self.root, bg="#f4f6f9")
        frame.pack(pady=20)

        tk.Label(self.root, text="📌 사용법:\n1. 이미지 & 라벨 폴더 선택\n2. 정리 작업 실행 (경고창 표시)\n3. 셔플 실행 시 파일명 변경 및 매핑 저장",
                 bg="#f4f6f9", font=("Arial", 12), justify="left").pack(pady=10)

        # 이미지 폴더 선택
        tk.Button(frame, text="이미지 폴더 선택", command=self.select_image_directory,
                  width=25, height=2, bg="#4CAF50", fg="white", font=("Arial", 11, "bold")).grid(row=0, column=0, padx=10)
        self.image_dir_entry = tk.Entry(frame, font=("Arial", 11), width=40, bd=2)
        self.image_dir_entry.grid(row=0, column=1, padx=10)

        # 라벨 폴더 선택
        tk.Button(frame, text="라벨 TXT 폴더 선택", command=self.select_label_directory,
                  width=25, height=2, bg="#4CAF50", fg="white", font=("Arial", 11, "bold")).grid(row=1, column=0, padx=10, pady=10)
        self.label_dir_entry = tk.Entry(frame, font=("Arial", 11), width=40, bd=2)
        self.label_dir_entry.grid(row=1, column=1, padx=10)

        # 정리 버튼
        tk.Button(self.root, text="정리 작업 실행", command=self.confirm_cleanup,
                  width=30, height=2, bg="#2196F3", fg="white", font=("Arial", 12, "bold")).pack(pady=10)

        # 셔플 설정
        tk.Label(self.root, text="새로운 이름 규칙 (예: GwangAn->GwangAn000000):", bg="#f4f6f9", font=("Arial", 11)).pack(pady=5)
        self.base_name_entry = tk.Entry(self.root, font=("Arial", 11), width=30, bd=2)
        self.base_name_entry.pack()

        tk.Label(self.root, text="셔플 횟수 (기본값 1):", bg="#f4f6f9", font=("Arial", 11)).pack(pady=5)
        self.shuffle_count_entry = tk.Entry(self.root, font=("Arial", 11), width=10, bd=2)
        self.shuffle_count_entry.pack()

        tk.Button(self.root, text="셔플 실행", command=self.execute_shuffle,
                  width=30, height=2, bg="#FF5722", fg="white", font=("Arial", 12, "bold")).pack(pady=10)

        self.status_label = tk.Label(self.root, text="상태: 대기 중", bg="#f4f6f9", font=("Arial", 12), fg="gray")
        self.status_label.pack(pady=10)

    def select_image_directory(self):
        self.image_folder_path = filedialog.askdirectory(title="이미지 폴더 선택")
        self.image_dir_entry.delete(0, tk.END)
        self.image_dir_entry.insert(0, self.image_folder_path)

    def select_label_directory(self):
        self.label_folder_path = filedialog.askdirectory(title="라벨 폴더 선택")
        self.label_dir_entry.delete(0, tk.END)
        self.label_dir_entry.insert(0, self.label_folder_path)

    def confirm_cleanup(self):
        if not self.image_folder_path or not self.label_folder_path:
            messagebox.showwarning("경고", "이미지와 라벨 폴더를 모두 선택하세요.")
            return

        img_exts = ['.jpg', '.png']
        img_files = [f for ext in img_exts for f in glob.glob(os.path.join(self.image_folder_path, f'*{ext}'))]
        label_files = glob.glob(os.path.join(self.label_folder_path, '*.txt'))

        label_names = {os.path.splitext(os.path.basename(txt))[0] for txt in label_files}
        delete_images = [img for img in img_files if os.path.splitext(os.path.basename(img))[0] not in label_names]

        delete_labels = []
        for txt in label_files:
            with open(txt, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines or any(len(line.strip().split()) < 5 for line in lines):
                    delete_labels.append(txt)

        delete_count = len(delete_images) + len(delete_labels)

        if delete_count == 0:
            messagebox.showinfo("정리 결과", "삭제할 항목이 없습니다.")
            return

        proceed = messagebox.askyesno("삭제 확인", f"총 {delete_count}개 파일이 삭제됩니다. 진행할까요?")
        if proceed:
            self.execute_cleanup(delete_images, delete_labels)

    def execute_cleanup(self, delete_images, delete_labels):
        deleted = []
        for img in delete_images:
            try:
                os.remove(img)
                deleted.append(img)
            except: pass

        for txt in delete_labels:
            try:
                os.remove(txt)
                deleted.append(txt)
                base = os.path.splitext(os.path.basename(txt))[0]
                for ext in ['.jpg', '.png']:
                    img_path = os.path.join(self.image_folder_path, base + ext)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        deleted.append(img_path)
            except: pass

        log_path = os.path.join(self.image_folder_path, "deleted_files_log.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            for item in deleted:
                f.write(item + '\n')

        self.status_label.config(text=f"삭제 완료: 총 {len(deleted)}개", fg="green")

    def execute_shuffle(self):
        base_name = self.base_name_entry.get().strip()
        try:
            count = int(self.shuffle_count_entry.get())
        except:
            count = 1

        if not base_name or not self.image_folder_path or not self.label_folder_path:
            self.status_label.config(text="❗ 경로 및 이름 규칙을 확인하세요.", fg="red")
            return

        images = sorted([f for f in os.listdir(self.image_folder_path) if f.lower().endswith(('.jpg', '.png'))])
        labels = {os.path.splitext(f)[0] for f in os.listdir(self.label_folder_path) if f.endswith('.txt')}

        valid_pairs = [f for f in images if os.path.splitext(f)[0] in labels]
        for _ in range(count):
            random.shuffle(valid_pairs)

        mapping = []
        for idx, old_img in enumerate(valid_pairs):
            new_base = f"{base_name}_{str(idx).zfill(6)}"
            old_img_path = os.path.join(self.image_folder_path, old_img)
            old_txt_path = os.path.join(self.label_folder_path, os.path.splitext(old_img)[0] + '.txt')
            new_img_path = os.path.join(self.image_folder_path, new_base + '.jpg')
            new_txt_path = os.path.join(self.label_folder_path, new_base + '.txt')

            shutil.move(old_img_path, new_img_path)
            shutil.move(old_txt_path, new_txt_path)
            mapping.append(f"{old_img} -> {new_base}.jpg")

        mapping_path = os.path.join(self.image_folder_path, "shuffle_mapping.txt")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            for line in mapping:
                f.write(line + '\n')

        self.status_label.config(text=f"셔플 완료! 총 {len(mapping)}개 변경됨", fg="blue")

# 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = PreprocessingTool(root)
    root.mainloop()

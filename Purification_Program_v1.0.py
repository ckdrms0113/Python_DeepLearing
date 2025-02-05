import os
import glob
import random
import tkinter as tk
from tkinter import filedialog

# Functions for selecting directories
def select_image_directory():
    global image_folder_path
    image_folder_path = filedialog.askdirectory(title="이미지 폴더 선택")
    if image_folder_path:
        image_dir_label.config(text=f"선택됨: {image_folder_path}", fg="green")
        image_dir_entry.delete(0, tk.END)
        image_dir_entry.insert(0, image_folder_path)  # Display path in the entry field
    else:
        image_dir_label.config(text="폴더가 선택되지 않았습니다.", fg="red")

def select_label_directory():
    global txt_folder_path
    txt_folder_path = filedialog.askdirectory(title="라벨링 폴더 선택")
    if txt_folder_path:
        label_dir_label.config(text=f"선택됨: {txt_folder_path}", fg="green")
        label_dir_entry.delete(0, tk.END)
        label_dir_entry.insert(0, txt_folder_path)  # Display path in the entry field
    else:
        label_dir_label.config(text="폴더가 선택되지 않았습니다.", fg="red")

def execute_cleanup():
    if not image_folder_path or not txt_folder_path:
        status_label.config(text="폴더를 모두 선택한 후 실행하세요.", fg="red")
        return

    # Image file extensions
    image_ext = ['.jpg', '.png']

    # Get TXT file names (without extensions)
    txt_files = glob.glob(os.path.join(txt_folder_path, '*.txt'))
    txt_file_names = {os.path.splitext(os.path.basename(txt))[0] for txt in txt_files}

    # Get image files
    image_files = [img for ext in image_ext for img in glob.glob(os.path.join(image_folder_path, f'*{ext}'))]

    # Initialize a counter for deleted images
    deleted_image_count = 0

    # Process image files
    for image_file in image_files:
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        if image_name not in txt_file_names:
            try:
                os.remove(image_file)
                deleted_image_count += 1  # Increment the deleted image count
                print(f'{image_file} 이미지 파일이 삭제되었습니다.')
            except Exception as e:
                print(f'{image_file} 삭제 중 오류 발생: {e}')

    # Process TXT files
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        should_delete = not lines or any(
            not line.strip() or not line.split()[0].isdigit() or len(line.split()) < 5 for line in lines
        )

        if should_delete:
            try:
                os.remove(txt_file)
                print(f'{txt_file} 텍스트 파일이 삭제되었습니다.')

                same_name = os.path.splitext(txt_file)[0]
                for ext in image_ext:
                    image_file = os.path.join(image_folder_path, os.path.basename(same_name) + ext)
                    if os.path.exists(image_file):
                        os.remove(image_file)
                        deleted_image_count += 1  # Increment the deleted image count
                        print(f'{image_file} 이미지 파일이 삭제되었습니다.')
            except Exception as e:
                print(f'{txt_file} 또는 {image_file} 삭제 중 오류 발생: {e}')

    # Display the result
    status_label.config(text=f"정리 작업 완료! 삭제된 이미지 파일: {deleted_image_count}개", fg="green")

# Function for saving image names and labels to a text file
def save_image_names_to_txt(image_directory, label_directory, txt_filename):
    image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
    label_files = [f for f in os.listdir(label_directory) if f.endswith('.txt')]
    
    with open(txt_filename, 'w') as file:
        for image in image_files:
            label_name = image.split('.')[0] + '.txt'
            if label_name in label_files:
                file.write(image + "\n")

# Function for shuffling and renaming image files
def shuffle_image_names(txt_filename, shuffle_count):
    with open(txt_filename, 'r') as file:
        image_names = file.readlines()
    
    image_mapping = []
    used_numbers = set()
    
    for image in image_names:
        image = image.strip()
        while True:
            random_number = random.randint(1000, 9999)
            if random_number not in used_numbers:
                used_numbers.add(random_number)
                break
        
        image_mapping.append((image, random_number))
    
    # Repeat shuffle process if shuffle_count is greater than 1
    for _ in range(shuffle_count - 1):
        random.shuffle(image_mapping)
    
    return image_mapping

def rename_images(image_mapping, image_directory, label_directory, new_txt_filename, base_name):
    new_mapping = []
    
    for idx, (image, _) in enumerate(image_mapping):
        new_name = f"{base_name}_{str(idx).zfill(6)}.jpg"
        
        old_image_path = os.path.join(image_directory, image)
        new_image_path = os.path.join(image_directory, new_name)
        
        if os.path.exists(old_image_path):
            os.rename(old_image_path, new_image_path)
        
        old_label_path = os.path.join(label_directory, image.split('.')[0] + '.txt')
        new_label_path = os.path.join(label_directory, new_name.split('.')[0] + '.txt')
        
        if os.path.exists(old_label_path):
            os.rename(old_label_path, new_label_path)
        
        new_mapping.append((image, new_name))
    
    with open(new_txt_filename, 'w') as file:
        for old_name, new_name in new_mapping:
            file.write(f"{old_name} -> {new_name}\n")

# Function to execute the rename and shuffle
def execute_rename_shuffle():
    image_directory = image_dir_entry.get()  # 텍스트 박스에서 폴더 경로를 가져옵니다.
    label_directory = label_dir_entry.get()
    txt_filename = os.path.join(image_directory, 'image_names.txt')
    new_txt_filename = os.path.join(image_directory, 'updated_image_names.txt')
    
    base_name = base_name_entry.get()  # 텍스트 박스에서 새로운 이름 규칙을 가져옵니다.
    
    # Get shuffle count from user input, default to 1 if not provided
    shuffle_count_str = shuffle_count_entry.get()
    try:
        shuffle_count = int(shuffle_count_str) if shuffle_count_str else 1
        if shuffle_count < 1:
            status_label.config(text="셔플 횟수는 1 이상이어야 합니다.", fg="red")
            return
    except ValueError:
        status_label.config(text="유효한 셔플 횟수를 입력하세요.", fg="red")
        return

    if not base_name:
        status_label.config(text="새로운 이름 규칙을 입력하세요.", fg="red")
        return

    save_image_names_to_txt(image_directory, label_directory, txt_filename)
    image_mapping = shuffle_image_names(txt_filename, shuffle_count)
    rename_images(image_mapping, image_directory, label_directory, new_txt_filename, base_name)

    # 완료 알림
    status_label.config(text="셔플 완료! 이미지 이름이 변경되었습니다.", fg="green")

# Initialize the main window
root = tk.Tk()
root.title("셔플러")
root.geometry("650x700")
root.config(bg="#f4f6f9")

# Global variables
image_folder_path = ""
txt_folder_path = ""

# Frame for buttons and labels
frame = tk.Frame(root, bg="#f4f6f9")
frame.pack(pady=20)

# Help Text for Instructions
help_label = tk.Label(root, text="사용법: \n1. '이미지 폴더'와 '라벨링 폴더'를 선택하세요.\n2. '정리 작업 실행' 버튼을 통해 빈 이미지를 제거합니다.\n3. '셔플러' 버튼을 통해 셔플링합니다.(최소 5회반복)", 
                      bg="#f4f6f9", font=("Arial", 12), justify="left")
help_label.pack(pady=10)

# Buttons and Labels for Image Directory
image_dir_button = tk.Button(frame, text="이미지 폴더 선택", command=select_image_directory, width=30, height=2, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
image_dir_button.grid(row=0, column=0, padx=20, pady=10)

image_dir_entry = tk.Entry(frame, font=("Arial", 12), width=30, bd=2)
image_dir_entry.grid(row=0, column=1, padx=20)

image_dir_label = tk.Label(frame, text="폴더가 선택되지 않았습니다.", bg="#f4f6f9", font=("Arial", 12))
image_dir_label.grid(row=0, column=2, padx=20)

# Buttons and Labels for Label Directory
label_dir_button = tk.Button(frame, text="라벨링 TXT 폴더 선택", command=select_label_directory, width=30, height=2, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
label_dir_button.grid(row=1, column=0, padx=20, pady=10)

label_dir_entry = tk.Entry(frame, font=("Arial", 12), width=30, bd=2)
label_dir_entry.grid(row=1, column=1, padx=20)

label_dir_label = tk.Label(frame, text="폴더가 선택되지 않았습니다.", bg="#f4f6f9", font=("Arial", 12))
label_dir_label.grid(row=1, column=2, padx=20)

# Input for base name
base_name_label = tk.Label(root, text="새로운 이름 규칙:", bg="#f4f6f9", font=("Arial", 12))
base_name_label.pack(pady=10)

base_name_entry = tk.Entry(root, font=("Arial", 12), width=40, bd=2)
base_name_entry.pack(pady=10)

# Input for shuffle count
shuffle_count_label = tk.Label(root, text="셔플 횟수 (기본 1회):", bg="#f4f6f9", font=("Arial", 12))
shuffle_count_label.pack(pady=5)

shuffle_count_entry = tk.Entry(root, font=("Arial", 12), width=10, bd=2)
shuffle_count_entry.pack(pady=5)

# Execute Buttons
execute_button = tk.Button(root, text="정리 작업 실행", command=execute_cleanup, width=30, height=2, bg="#2196F3", fg="white", font=("Arial", 12, "bold"))
execute_button.pack(pady=10)

rename_shuffle_button = tk.Button(root, text="셔플러 실행", command=execute_rename_shuffle, width=30, height=2, bg="#FF5722", fg="white", font=("Arial", 12, "bold"))
rename_shuffle_button.pack(pady=10)

# Status label
status_label = tk.Label(root, text="상태: 대기 중", bg="#f4f6f9", font=("Arial", 12), fg="gray")
status_label.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()
import os
import glob
import random
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

# ---------------------------- 스타일 설정 ----------------------------
BG_COLOR = "#f0f0f0"
PRIMARY_COLOR = "#2c3e50"
SECONDARY_COLOR = "#3498db"
ACCENT_COLOR = "#e74c3c"
TEXT_COLOR = "#2c3e50"
FONT = ("맑은 고딕", 10)

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = ttk.Label(self.tooltip, text=self.text, background="#ffffe0", 
                         relief="solid", borderwidth=1, font=("맑은 고딕", 9))
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

# ---------------------------- 핵심 기능 (변경 없음) ----------------------------
# (기존의 select_image_directory, select_label_directory, execute_cleanup, 
# save_image_names_to_txt, shuffle_image_names, rename_images, 
# execute_rename_shuffle 함수들 동일하게 유지)

# ---------------------------- GUI 업그레이드 ----------------------------
class ShufflerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("고급 파일 셔플러")
        self.geometry("800x750")
        self.configure(bg=BG_COLOR)
        
        self.image_folder_path = ""
        self.txt_folder_path = ""
        self.already_shuffle = False
        self.execute_rename_shuffle_count = 0
        
        self.create_widgets()
        self.style = ttk.Style()
        self.setup_style()
        
    def setup_style(self):
        self.style.theme_use("clam")
        self.style.configure("TFrame", background=BG_COLOR)
        self.style.configure("TButton", font=FONT, padding=6)
        self.style.configure("Primary.TButton", background=SECONDARY_COLOR, foreground="white")
        self.style.configure("Secondary.TButton", background=ACCENT_COLOR, foreground="white")
        self.style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=FONT)
        self.style.configure("TEntry", font=FONT, padding=5)
        self.style.configure("Header.TLabel", font=("맑은 고딕", 12, "bold"))
        self.style.configure("Status.TLabel", font=("맑은 고딕", 9))
        self.style.configure("Red.TLabel", foreground="red")
        self.style.configure("Green.TLabel", foreground="green")
        
    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Directory Selection Section
        dir_frame = ttk.LabelFrame(main_frame, text="폴더 선택", padding=(10, 5))
        dir_frame.pack(fill=tk.X, pady=5)
        
        self.image_dir_entry = self.create_dir_row(dir_frame, "이미지 폴더:", 0, self.select_image_directory)
        self.label_dir_entry = self.create_dir_row(dir_frame, "라벨 폴더:", 1, self.select_label_directory)
        
        # Configuration Section
        config_frame = ttk.LabelFrame(main_frame, text="설정", padding=(10, 5))
        config_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(config_frame, text="새 이름 규칙:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.base_name_entry = ttk.Entry(config_frame, width=30)
        self.base_name_entry.grid(row=0, column=1, padx=5, pady=2)
        ToolTip(self.base_name_entry, "예시: mydata → mydata_000001.jpg 형식으로 생성")
        
        ttk.Label(config_frame, text="셔플 횟수:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.shuffle_count_entry = ttk.Entry(config_frame, width=10)
        self.shuffle_count_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        self.shuffle_count_entry.insert(0, "1")
        
        # Progress Section
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, pady=10)
        
        # Action Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.clean_btn = ttk.Button(btn_frame, text="파일 정리 실행", style="Primary.TButton", 
                                  command=self.threaded_execute_cleanup)
        self.clean_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        self.shuffle_btn = ttk.Button(btn_frame, text="셔플 실행", style="Secondary.TButton", 
                                    command=self.threaded_execute_rename_shuffle)
        self.shuffle_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        # Log Console
        log_frame = ttk.LabelFrame(main_frame, text="실행 로그", padding=(10, 5))
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 9), 
                                                 padx=10, pady=10, height=10)
        self.log_area.pack(fill=tk.BOTH, expand=True)
        
    def create_dir_row(self, parent, label_text, row, command):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky=tk.EW, pady=2)
        
        label = ttk.Label(frame, text=label_text, width=10)
        label.pack(side=tk.LEFT)
        
        entry = ttk.Entry(frame, width=40)
        entry.pack(side=tk.LEFT, padx=5)
        
        btn = ttk.Button(frame, text="찾아보기", style="Primary.TButton", command=command)
        btn.pack(side=tk.LEFT)
        return entry
    
    def log_message(self, message, color=None):
        self.log_area.configure(state='normal')
        tag = str(random.randint(1,100000))  # Unique tag for color
        if color:
            self.log_area.tag_configure(tag, foreground=color)
            self.log_area.insert(tk.END, message + "\n", tag)
        else:
            self.log_area.insert(tk.END, message + "\n")
        self.log_area.configure(state='disabled')
        self.log_area.see(tk.END)
    
    def threaded_execute_cleanup(self):
        self.progress["value"] = 0
        thread = threading.Thread(target=self.execute_cleanup, daemon=True)
        thread.start()
    
    def threaded_execute_rename_shuffle(self):
        self.progress["value"] = 0
        thread = threading.Thread(target=self.execute_rename_shuffle, daemon=True)
        thread.start()
    
    # ---------------------------- 기존 기능 메서드화 ----------------------------
    # (기존 함수들을 클래스 메서드로 변환)
    # ... [기존 함수 코드를 여기에 클래스 메서드로 재구성] ...
    
    def select_image_directory(self):
        path = filedialog.askdirectory(title="이미지 폴더 선택")
        if path:
            self.image_folder_path = path
            self.image_dir_entry.delete(0, tk.END)
            self.image_dir_entry.insert(0, path)
            self.log_message(f"이미지 폴더 설정됨: {path}", "green")
    
    def select_label_directory(self):
        path = filedialog.askdirectory(title="라벨 폴더 선택")
        if path:
            self.txt_folder_path = path
            self.label_dir_entry.delete(0, tk.END)
            self.label_dir_entry.insert(0, path)
            self.log_message(f"라벨 폴더 설정됨: {path}", "green")

if __name__ == "__main__":
    app = ShufflerApp()
    app.mainloop()
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Modify filter")
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.img_path = ""

        # Left frame
        self.left_frame = tk.Frame(root, width=self.screen_width/2, bg="red")
        self.left_frame.pack(side='left', padx=5, fill=tk.BOTH)

        ## original image
        self.ori_img_canvas = tk.Canvas(self.left_frame, width=self.screen_width/2, bg="green")
        print(self.screen_width/2)
        self.ori_img_canvas.pack(fill=tk.BOTH, expand=True);

        self.trans_img_canvas = tk.Canvas(self.left_frame, width=self.screen_width/2, bg="yellow")
        self.trans_img_canvas.pack(fill=tk.BOTH, expand=True);

        # Right frame
        self.right_frame = tk.Frame(root, width=self.screen_width/2, height=(self.screen_height/10)*6, bg='white')
        self.right_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        self.right_frame.config(highlightthickness=1, highlightbackground='gray')

        # Transformation img panel
        font_panel = ('Comic Sans MS', 10)
        width_scale = 5
        # Log panel
        self.log_panel = tk.LabelFrame(self.right_frame, text='Biến đổi log', bg='#81ecec', font=font_panel)
        self.log_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.cweight_label = tk.Label(self.log_panel, text='Hệ số c', font=font_panel, bg='#81ecec')
        self.cweight_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.log_scale = tk.Scale(self.log_panel, orient=tk.HORIZONTAL, resolution=0.1, width=width_scale, length=500, command=self.log_trans)
        self.log_scale.pack(side=tk.LEFT, anchor='nw')

        # Button frame
        self.button_frame = tk.Frame(root, width=self.screen_width / 2, bg='white')
        self.button_frame.pack(fill=tk.BOTH, padx=5, pady=10)
        self.button_frame.config(highlightthickness=1, highlightbackground='gray')
        # Add button
        self.gap = 16
        font_button = ('Comic Sans MS', 10)
        bg_button = "#badc58"
        # choose img
        self.choose_folder_button = tk.Button(self.button_frame, text="Chọn ảnh", width=15, command=self.choose_image,
                                              font=font_button, bg=bg_button)
        self.choose_folder_button.pack(side=tk.LEFT, pady=10, padx=self.gap)
        # update
        self.update_button = tk.Button(self.button_frame, text="Cập nhật", width=15, command='', font=font_button,
                                       bg=bg_button)
        self.update_button.pack(side=tk.LEFT, pady=10, padx=self.gap)
        # save
        self.save_button = tk.Button(self.button_frame, text="Lưu ra file", width=15, command='', font=font_button,
                                     bg=bg_button)
        self.save_button.pack(side=tk.LEFT, pady=10, padx=self.gap)
        # close
        self.close_button = tk.Button(self.button_frame, text="Đóng ảnh", width=15, command='', font=font_button,
                                      bg=bg_button)
        self.close_button.pack(side=tk.LEFT, pady=10, padx=self.gap)

    def log_trans(self, value):
        c = float(value)
        img = np.array(self.img_rgb, dtype=float)
        log_img = c * np.log(img + 1)
        log_img = np.array(log_img, dtype=np.uint8)
        self.display_transformed_img(log_img)

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            self.img_path = file_path
            print(f"Hình ảnh đã chọn: {file_path}")
            self.display_img(file_path)

    def display_img(self, image_path):
        for widget in self.ori_img_canvas.winfo_children():
            widget.destroy()

        img_large = cv2.imread(image_path)
        img_large = cv2.cvtColor(img_large, cv2.COLOR_BGR2RGB)
        self.img_rgb = img_large
        width_img = self.ori_img_canvas.winfo_width()
        height_img = self.ori_img_canvas.winfo_height()
        img_large = cv2.resize(img_large, (width_img, height_img))
        image_tk_large = ImageTk.PhotoImage(Image.fromarray(img_large))

        self.ori_img_canvas.create_image(0, 0, anchor=tk.NW, image=image_tk_large)
        self.ori_img_canvas.image = image_tk_large

        self.trans_img_canvas.create_image(0, 0, anchor=tk.NW, image=image_tk_large)
        self.trans_img_canvas.image = image_tk_large

    def display_transformed_img(self, img_rgb):
        width_img = self.ori_img_canvas.winfo_width()
        height_img = self.ori_img_canvas.winfo_height()
        img = cv2.resize(img_rgb, (width_img, height_img))
        img_tk = ImageTk.PhotoImage(Image.fromarray(img))
        self.trans_img_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.trans_img_canvas.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('1500x700')
    app = App(root)
    root.mainloop()
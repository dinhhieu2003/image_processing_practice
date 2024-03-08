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
        self.ori_img_canvas.pack(fill=tk.BOTH, expand=True)

        self.trans_img_canvas = tk.Canvas(self.left_frame, width=self.screen_width/2, bg="yellow")
        self.trans_img_canvas.pack(fill=tk.BOTH, expand=True)

        # Right frame
        self.right_frame = tk.Frame(root, width=self.screen_width/2, height=(self.screen_height/10)*6, bg='white')
        self.right_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        self.right_frame.config(highlightthickness=1, highlightbackground='gray')

        # Transformation img panel
        font_panel = ('Comic Sans MS', 10)
        width_scale = 5

        # Erosion
        self.erosion_panel = tk.LabelFrame(self.right_frame, text='Erosion', bg='#81ecec',
                                                        font=font_panel)
        self.erosion_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.erosion_label = tk.Label(self.erosion_panel, text='Size', font=font_panel,
                                               bg='#81ecec')
        self.erosion_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.erosion_size_scale = tk.Scale(self.erosion_panel, orient=tk.HORIZONTAL, resolution=1.0,
                                               width=width_scale, length=500, command=self.erosion_trans,
                                               from_=1.0)
        self.erosion_size_scale.pack(side=tk.LEFT, anchor='nw')

        # Dilation
        self.dilation_panel = tk.LabelFrame(self.right_frame, text='dilation', bg='#a29bfe',
                                           font=font_panel)
        self.dilation_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.dilation_label = tk.Label(self.dilation_panel, text='Size', font=font_panel,
                                      bg='#a29bfe')
        self.dilation_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.dilation_size_scale = tk.Scale(self.dilation_panel, orient=tk.HORIZONTAL, resolution=1.0,
                                           width=width_scale, length=500, command=self.dilation_trans,
                                           from_=1.0)
        self.dilation_size_scale.pack(side=tk.LEFT, anchor='nw')

        # Button frame
        self.button_frame = tk.Frame(root, width=self.screen_width/2, bg='white')
        self.button_frame.pack(fill=tk.BOTH, padx=5, pady=10)
        self.button_frame.config(highlightthickness=1, highlightbackground='gray')
        # Add button
        self.gap = 16
        font_button = ('Comic Sans MS', 10)
        bg_button = "#badc58"
        # choose img
        self.choose_folder_button = tk.Button(self.button_frame, text="Chọn ảnh", width=15, command=self.choose_image, font=font_button, bg=bg_button)
        self.choose_folder_button.pack(side=tk.LEFT, pady=10, padx=self.gap)
        # update
        self.update_button = tk.Button(self.button_frame, text="Cập nhật", width=15, command=self.update_image, font=font_button ,bg=bg_button)
        self.update_button.pack(side=tk.LEFT, pady=10, padx=self.gap)
        # save
        self.save_button = tk.Button(self.button_frame, text="Lưu ra file", width=15, command=self.save_image, font=font_button, bg=bg_button)
        self.save_button.pack(side=tk.LEFT, pady=10, padx=self.gap)
        # close
        self.close_button = tk.Button(self.button_frame, text="Đóng ảnh", width=15, command='', font=font_button, bg=bg_button)
        self.close_button.pack(side=tk.LEFT, pady=10, padx=self.gap)

    def erosion_trans(self, value):
        size = int(value)
        kernel = np.ones((size, size), np.uint8)
        img_ero = cv2.erode(self.img_rgb, kernel, iterations=1)
        self.display_transformed_img(img_ero)

    def dilation_trans(self, value):
        size = int(value)
        kernel = np.ones((size, size), np.uint8)
        img_ero = cv2.dilate(self.img_rgb, kernel, iterations=1)
        self.display_transformed_img(img_ero)

    def update_image(self):
        self.img_rgb = self.transformed_img
        self.display_update()

    def save_image(self):
        image = self.transformed_img
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            # Lưu hình ảnh ra file
            cv2.imwrite(file_path, image)

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            self.img_path = file_path
            print(f"Hình ảnh đã chọn: {file_path}")
            self.display_img(file_path)

    def display_img(self, image_path):
        for widget in self.ori_img_canvas.winfo_children():
            widget.destroy()

        img_origin = cv2.imread(image_path, 0)
        threshval = 100
        n = 255
        retval, img_origin = cv2.threshold(img_origin, threshval, n,
                                    cv2.THRESH_BINARY)
        #img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
        self.img_rgb = img_origin
        width_img = self.ori_img_canvas.winfo_width()
        height_img = self.ori_img_canvas.winfo_height()
        img_origin = cv2.resize(img_origin, (width_img, height_img))
        image_tk_large = ImageTk.PhotoImage(Image.fromarray(img_origin))

        self.ori_img_canvas.create_image(0, 0, anchor=tk.NW, image=image_tk_large)
        self.ori_img_canvas.image = image_tk_large

        self.trans_img_canvas.create_image(0, 0, anchor=tk.NW, image=image_tk_large)
        self.trans_img_canvas.image = image_tk_large

    def display_update(self):
        for widget in self.ori_img_canvas.winfo_children():
            widget.destroy()

        width_img = self.ori_img_canvas.winfo_width()
        height_img = self.ori_img_canvas.winfo_height()
        img_origin = cv2.resize(self.img_rgb, (width_img, height_img))
        image_tk_large = ImageTk.PhotoImage(Image.fromarray(img_origin))

        self.ori_img_canvas.create_image(0, 0, anchor=tk.NW, image=image_tk_large)
        self.ori_img_canvas.image = image_tk_large

        self.trans_img_canvas.create_image(0, 0, anchor=tk.NW, image=image_tk_large)
        self.trans_img_canvas.image = image_tk_large

    def display_transformed_img(self, img_rgb):
        self.transformed_img = img_rgb
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
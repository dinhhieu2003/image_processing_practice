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

        # negative img
        self.negative_img_button = tk.Button(self.right_frame, text="Negative image", width=15, command=self.trans_to_neg_img, font=("Comic Sans MS", 10), bg="#badc58")
        self.negative_img_button.pack(side=tk.TOP, anchor='nw')

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

        # Piecewise panel
        self.piecewise_panel = tk.LabelFrame(self.right_frame, text='Biến đổi Piecewise-Linear', bg='#a29bfe', font=font_panel)
        self.piecewise_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        # # Hệ số cao
        self.highweight_scale_frame = tk.Frame(self.piecewise_panel, bg='#a29bfe')
        self.highweight_scale_frame.pack(pady=10, anchor='nw')

        self.highweight_label = tk.Label(self.highweight_scale_frame, text='Hệ số cao', font=font_panel, bg='#a29bfe')
        self.highweight_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.highweight_scale = tk.Scale(self.highweight_scale_frame, orient=tk.HORIZONTAL, resolution=0.1, width=width_scale, length=500)
        self.highweight_scale.pack(side=tk.LEFT, anchor='nw')
        # # Hệ số thấp
        self.lowweight_scale_frame = tk.Frame(self.piecewise_panel, bg='#a29bfe')
        self.lowweight_scale_frame.pack(anchor='nw')

        self.lowweight_label = tk.Label(self.lowweight_scale_frame, text='Hệ số thấp', font=font_panel, bg='#a29bfe')
        self.lowweight_label.pack(side=tk.LEFT, padx=10, anchor='nw')

        self.lowweight_scale = tk.Scale(self.lowweight_scale_frame, orient=tk.HORIZONTAL, resolution=0.1, width=width_scale, length=500)
        self.lowweight_scale.pack(side=tk.LEFT, anchor='nw')

        # Gamma trans
        self.gamma_panel = tk.LabelFrame(self.right_frame, text='Biến đổi Gamma', bg='#ffbe76', font=font_panel)
        self.gamma_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        # # Hệ số C
        self.cweight_gamma_scale_frame = tk.Frame(self.gamma_panel, bg='#ffbe76')
        self.cweight_gamma_scale_frame.pack(pady=10, anchor='nw')

        self.cweight_gamma_label = tk.Label(self.cweight_gamma_scale_frame, text='Hệ số C', font=font_panel, bg='#ffbe76')
        self.cweight_gamma_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.cweight_gamma_scale = tk.Scale(self.cweight_gamma_scale_frame, orient=tk.HORIZONTAL, resolution=1.0, width=width_scale, length=500, command=self.change_cweight_gamma)
        self.cweight_gamma_scale.pack(side=tk.LEFT, anchor='nw')
        # # Hệ số gamma
        self.gammaweight_scale_frame = tk.Frame(self.gamma_panel, bg='#ffbe76')
        self.gammaweight_scale_frame.pack(anchor='nw')

        self.gammaweight_label = tk.Label(self.gammaweight_scale_frame, text='Gamma', font=font_panel, bg='#ffbe76')
        self.gammaweight_label.pack(side=tk.LEFT, padx=10, anchor='nw')

        self.gammaweight_scale = tk.Scale(self.gammaweight_scale_frame, orient=tk.HORIZONTAL, resolution=0.01, width=width_scale, length=500, to=25.0, command=self.change_gammaweight)
        self.gammaweight_scale.pack(side=tk.LEFT, anchor='nw')

        # Làm trơn ảnh (Lọc trung bình)
        self.mean_filter_panel = tk.LabelFrame(self.right_frame, text='Làm trơn ảnh (lọc trung bình)', bg='#81ecec', font=font_panel)
        self.mean_filter_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.mean_filter_label = tk.Label(self.mean_filter_panel, text='Kích thước lọc', font=font_panel, bg='#81ecec')
        self.mean_filter_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.mean_filter_scale = tk.Scale(self.mean_filter_panel, orient=tk.HORIZONTAL, resolution=1.0, width=width_scale, length=500, command=self.mean_filter)
        self.mean_filter_scale.pack(side=tk.LEFT, anchor='nw')

        # Làm trơn ảnh (lọc Gauss)
        self.gaussfilter_panel = tk.LabelFrame(self.right_frame, text='Làm trơn ảnh (lọc Gauss)', bg='#a29bfe', font=font_panel)
        self.gaussfilter_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        # # kích thước lọc
        self.gaussfilter_size_scale_frame = tk.Frame(self.gaussfilter_panel, bg='#a29bfe')
        self.gaussfilter_size_scale_frame.pack(pady=10, anchor='nw')

        self.gaussfilter_size_label = tk.Label(self.gaussfilter_size_scale_frame, text='Kích thước lọc', font=font_panel, bg='#a29bfe')
        self.gaussfilter_size_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.gaussfilter_size_scale = tk.Scale(self.gaussfilter_size_scale_frame, orient=tk.HORIZONTAL, resolution=1.0, width=width_scale, length=500,from_=5, command=self.gauss_filter_change_size)
        self.gaussfilter_size_scale.pack(side=tk.LEFT, anchor='nw')
        # # Hệ số sigma
        self.gaussfilter_sigma_scale_frame = tk.Frame(self.gaussfilter_panel, bg='#a29bfe')
        self.gaussfilter_sigma_scale_frame.pack(anchor='nw')

        self.gaussfilter_sigma_label = tk.Label(self.gaussfilter_sigma_scale_frame, text='Hệ số sigma', font=font_panel, bg='#a29bfe')
        self.gaussfilter_sigma_label.pack(side=tk.LEFT, padx=10, anchor='nw')

        self.gaussfilter_sigma_scale = tk.Scale(self.gaussfilter_sigma_scale_frame, orient=tk.HORIZONTAL, resolution=0.1, width=width_scale, length=500, from_=1.5, command=self.gauss_filter_change_sigma)
        self.gaussfilter_sigma_scale.pack(side=tk.LEFT, anchor='nw')

        # Làm trơn ảnh (lọc trung vị)
        self.medfilter_panel = tk.LabelFrame(self.right_frame, text='Làm trơn ảnh (lọc trung vị)', bg='#ffbe76', font=font_panel)
        self.medfilter_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.medfilter_label = tk.Label(self.medfilter_panel, text='Kích thước lọc', font=font_panel, bg='#ffbe76')
        self.medfilter_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.medfilter_scale = tk.Scale(self.medfilter_panel, orient=tk.HORIZONTAL, resolution=1.0, width=width_scale, length=500, command=self.median_filter)
        self.medfilter_scale.pack(side=tk.LEFT, anchor='nw')

        # Cân bằng sáng dùng histogram
        self.brightness_balancing_panel = tk.LabelFrame(self.right_frame, text='Cân bằng sáng dùng histogram', bg='#81ecec', font=font_panel)
        self.brightness_balancing_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.brightness_balancing_label = tk.Label(self.brightness_balancing_panel, text='Kích thước lọc', font=font_panel, bg='#81ecec')
        self.brightness_balancing_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.brightness_balancing_scale = tk.Scale(self.brightness_balancing_panel, orient=tk.HORIZONTAL, resolution=0.1, width=width_scale, length=500)
        self.brightness_balancing_scale.pack(side=tk.LEFT, anchor='nw')

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

    def Gausskernel(self, l, sig):
        s = round((l - 1) / 2)
        ax = np.linspace(-s, s, l)
        gauss = np.exp(-np.square(ax) / (2 * (sig ** 2)))
        kernel = np.outer(gauss, gauss)
        # tính tích the outer product of two vectors.
        return kernel / np.sum(kernel)

    def gauss_filter(self, sz, sig):
        k = self.Gausskernel(l=sz, sig=sig)
        img = np.array(self.img_rgb, dtype=float)
        img_target = cv2.filter2D(src=img, kernel=k, ddepth=-1)
        img_target = np.array(img_target, dtype=np.uint8)

        #img_target = cv2.GaussianBlur(self.img_rgb, (sz,sz), sig)
        self.display_transformed_img(img_target)

    def gauss_filter_change_size(self, value):
        sz = int(value)
        sig = self.gaussfilter_sigma_scale.get()
        self.gauss_filter(sz=sz, sig=sig)

    def gauss_filter_change_sigma(self, value):
        sig = float(value)
        sz = self.gaussfilter_size_scale.get()
        self.gauss_filter(sz=sz, sig=sig)

    def median_filter(self, value):
        sz = int(value)
        if sz == 0:
            return
        # img_rgb = self.img_rgb
        # h,w, _ = img_rgb.shape
        # img_target = np.ones((h, w))
        # for i in range(0, h-sz+1):
        #     for j in range(0, w-sz+1):
        #         sA = img_rgb[i:i+sz, j:j+sz]
        #         img_target[i, j] = np.median(sA)
        # img_target = img_target[0:h-sz+1, 0:w-sz+1]

        img_target = cv2.medianBlur(self.img_rgb, sz)
        self.display_transformed_img(img_target)

    def mean_filter(self, value):
        sz = int(value)
        if sz == 0:
            return
        kernel = np.ones((sz, sz), np.float32)/(sz*sz)
        img_mean_filter = cv2.filter2D(src=self.img_rgb, kernel=kernel, ddepth=-1)
        self.display_transformed_img(img_mean_filter)

    def gamma_trans(self, c, gamma):
        image_normalized = self.img_rgb.astype(float) / 255.0
        img_gamma = c * np.power(image_normalized, gamma)
        img_gamma = (img_gamma * 255).astype(np.uint8)
        self.display_transformed_img(img_gamma)


    def change_gammaweight(self, value):
        gamma = float(value)
        c = float(self.cweight_gamma_scale.get())
        self.gamma_trans(c, gamma)

    def change_cweight_gamma(self, value):
        c = float(value)
        gamma = float(self.gammaweight_scale.get())
        self.gamma_trans(c, gamma)

    def log_trans(self, value):
        c = float(value)
        #c = 255 / (np.log(1 + np.max(self.img_rgb)))
        img = np.array(self.img_rgb, dtype=float)
        log_img = c * np.log(img + 1)
        log_img = np.array(log_img, dtype=np.uint8)
        self.display_transformed_img(log_img)

    def trans_to_neg_img(self):
        img_bgr = cv2.imread(self.img_path)
        img_neg = img_bgr
        height, width, _ = img_bgr.shape
        for i in range (0 , height - 1):
            for j in range (0, width - 1):
                pixel = img_bgr[i, j]
                pixel[0] = 255 - pixel[0]
                pixel[1] = 255 - pixel[1]
                pixel[2] = 255 - pixel[2]
                img_neg[i, j] = pixel
        img_neg = cv2.cvtColor(img_neg, cv2.COLOR_BGR2RGB)
        self.display_transformed_img(img_neg)

    def update_image(self):
        self.img_rgb = self.transformed_img
        self.display_update()

    def save_image(self):
        image = self.transformed_img
        # Hiển thị hộp thoại "Lưu" để chọn nơi lưu và đặt tên file
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        # Kiểm tra nếu người dùng đã chọn nơi lưu và đặt tên file
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

        img_origin = cv2.imread(image_path)
        img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
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
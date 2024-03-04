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

        # Lowpass - Ideal filter
        self.lowpass_ideal_filter_panel = tk.LabelFrame(self.right_frame, text='Lowpass ideal filter', bg='#81ecec', font=font_panel)
        self.lowpass_ideal_filter_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.lowpass_ideal_d0_label = tk.Label(self.lowpass_ideal_filter_panel, text='D0', font=font_panel, bg='#81ecec')
        self.lowpass_ideal_d0_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.lowpass_ideal_d0_scale = tk.Scale(self.lowpass_ideal_filter_panel, orient=tk.HORIZONTAL, resolution=0.1, width=width_scale, length=500, command=self.lowpass_ideal_filter, from_=1.0)
        self.lowpass_ideal_d0_scale.pack(side=tk.LEFT, anchor='nw')

        # Lowpass - Butterworth filter
        self.lowpass_butterworth_filter_panel = tk.LabelFrame(self.right_frame, text='Lowpass butterworth filter', bg='#a29bfe', font=font_panel)
        self.lowpass_butterworth_filter_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        # # D0
        self.lowpass_butterworth_d0_scale_frame = tk.Frame(self.lowpass_butterworth_filter_panel, bg='#a29bfe')
        self.lowpass_butterworth_d0_scale_frame.pack(pady=10, anchor='nw')

        self.lowpass_butterworth_d0_label = tk.Label(self.lowpass_butterworth_d0_scale_frame, text='D0', font=font_panel, bg='#a29bfe')
        self.lowpass_butterworth_d0_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.lowpass_butterworth_d0_scale = tk.Scale(self.lowpass_butterworth_d0_scale_frame, orient=tk.HORIZONTAL, resolution=0.1, width=width_scale, length=500, command=self.lowpass_butterworth_change_d0, from_=1.0)
        self.lowpass_butterworth_d0_scale.pack(side=tk.LEFT, anchor='nw')
        # # n
        self.lowpass_butterworth_n_scale_frame = tk.Frame(self.lowpass_butterworth_filter_panel, bg='#a29bfe')
        self.lowpass_butterworth_n_scale_frame.pack(anchor='nw')

        self.lowpass_butterworth_n_label = tk.Label(self.lowpass_butterworth_n_scale_frame, text='n', font=font_panel, bg='#a29bfe')
        self.lowpass_butterworth_n_label.pack(side=tk.LEFT, padx=10, anchor='nw')

        self.lowpass_butterworth_n_scale = tk.Scale(self.lowpass_butterworth_n_scale_frame, orient=tk.HORIZONTAL, resolution=0.1, width=width_scale, length=500, command=self.lowpass_butterworth_change_n, from_=1.0)
        self.lowpass_butterworth_n_scale.pack(side=tk.LEFT, anchor='nw')

        # Gaussian
        self.lowpass_gauss_filter_panel = tk.LabelFrame(self.right_frame, text='Lowpass gaussian filter', bg='#81ecec',
                                                        font=font_panel)
        self.lowpass_gauss_filter_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.lowpass_gauss_d0_label = tk.Label(self.lowpass_gauss_filter_panel, text='D0', font=font_panel,
                                               bg='#81ecec')
        self.lowpass_gauss_d0_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.lowpass_gauss_d0_scale = tk.Scale(self.lowpass_gauss_filter_panel, orient=tk.HORIZONTAL, resolution=0.1,
                                               width=width_scale, length=500, command=self.lowpass_gauss_filter, from_=1.0)
        self.lowpass_gauss_d0_scale.pack(side=tk.LEFT, anchor='nw')

        # Highpass - Ideal filter
        self.highpass_ideal_filter_panel = tk.LabelFrame(self.right_frame, text='Highpass ideal filter', bg='#81ecec',
                                                        font=font_panel)
        self.highpass_ideal_filter_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.highpass_ideal_d0_label = tk.Label(self.highpass_ideal_filter_panel, text='D0', font=font_panel,
                                               bg='#81ecec')
        self.highpass_ideal_d0_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.highpass_ideal_d0_scale = tk.Scale(self.highpass_ideal_filter_panel, orient=tk.HORIZONTAL, resolution=0.1,
                                               width=width_scale, length=500, command=self.highpass_ideal_filter, from_=1.0)
        self.highpass_ideal_d0_scale.pack(side=tk.LEFT, anchor='nw')

        # Highpass - Butterworth filter
        self.highpass_butterworth_filter_panel = tk.LabelFrame(self.right_frame, text='Highpass butterworth filter',
                                                              bg='#a29bfe', font=font_panel)
        self.highpass_butterworth_filter_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        # # D0
        self.highpass_butterworth_d0_scale_frame = tk.Frame(self.highpass_butterworth_filter_panel, bg='#a29bfe')
        self.highpass_butterworth_d0_scale_frame.pack(pady=10, anchor='nw')

        self.highpass_butterworth_d0_label = tk.Label(self.highpass_butterworth_d0_scale_frame, text='D0',
                                                     font=font_panel, bg='#a29bfe')
        self.highpass_butterworth_d0_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.highpass_butterworth_d0_scale = tk.Scale(self.highpass_butterworth_d0_scale_frame, orient=tk.HORIZONTAL,
                                                     resolution=0.1, width=width_scale, length=500, from_=1.0, command=self.highpass_butterworth_change_D0)
        self.highpass_butterworth_d0_scale.pack(side=tk.LEFT, anchor='nw')
        # # n
        self.highpass_butterworth_n_scale_frame = tk.Frame(self.highpass_butterworth_filter_panel, bg='#a29bfe')
        self.highpass_butterworth_n_scale_frame.pack(anchor='nw')

        self.highpass_butterworth_n_label = tk.Label(self.highpass_butterworth_n_scale_frame, text='n', font=font_panel,
                                                    bg='#a29bfe')
        self.highpass_butterworth_n_label.pack(side=tk.LEFT, padx=10, anchor='nw')

        self.highpass_butterworth_n_scale = tk.Scale(self.highpass_butterworth_n_scale_frame, orient=tk.HORIZONTAL,
                                                    resolution=0.1, width=width_scale, length=500, from_=1.0, command=self.highpass_butterworth_change_n)
        self.highpass_butterworth_n_scale.pack(side=tk.LEFT, anchor='nw')

        #Gaussian
        self.highpass_gauss_filter_panel = tk.LabelFrame(self.right_frame, text='Highpass gaussian filter', bg='#81ecec',
                                                        font=font_panel)
        self.highpass_gauss_filter_panel.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.highpass_gauss_d0_label = tk.Label(self.highpass_gauss_filter_panel, text='D0', font=font_panel,
                                               bg='#81ecec')
        self.highpass_gauss_d0_label.pack(side=tk.LEFT, anchor='nw', padx=10)

        self.highpass_gauss_d0_scale = tk.Scale(self.highpass_gauss_filter_panel, orient=tk.HORIZONTAL, resolution=0.1,
                                               width=width_scale, length=500, command=self.highpass_gauss_filter, from_=1.0)
        self.highpass_gauss_d0_scale.pack(side=tk.LEFT, anchor='nw')

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

    def highpass_butterworth_filter(self, D0, n):
        img = self.img_rgb
        F = np.fft.fft2(img)
        F = np.fft.fftshift(F)
        M, N = img.shape
        u = np.arange(0, M) - M / 2
        v = np.arange(0, N) - N / 2
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = 1 / np.power(1 + (D0 / D), (2 * n))
        G = H * F
        G = np.fft.ifftshift(G)
        img_out = np.real(np.fft.ifft2(G))
        self.display_transformed_img(img_out)

    def highpass_butterworth_change_D0(self, value):
        D0 = float(value)
        n = float(self.highpass_butterworth_n_scale.get())
        self.highpass_butterworth_filter(D0, n)

    def highpass_butterworth_change_n(self, value):
        n = float(value)
        D0 = float(self.highpass_butterworth_d0_scale.get())

    def highpass_gauss_filter(self, value):
        D0 = float(value)
        img = self.img_rgb
        F = np.fft.fft2(img)
        F = np.fft.fftshift(F)
        M, N = img.shape
        u = np.arange(0, M) - M / 2
        v = np.arange(0, N) - N / 2
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = 1 - np.exp((-1 * np.square(D)) / (2 * D0 ** 2))
        G = H * F
        G = np.fft.ifftshift(G)
        img_out = np.real(np.fft.ifft2(G))
        self.display_transformed_img(img_out)

    def highpass_ideal_filter(self, value):
        D0 = float(value)
        img = self.img_rgb
        F = np.fft.fft2(img)
        F = np.fft.fftshift(F)
        M, N = img.shape
        u = np.arange(0, M) - M / 2
        v = np.arange(0, N) - N / 2
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = np.array(D > D0, 'float')
        G = H * F
        G = np.fft.ifftshift(G)
        img_out = np.real(np.fft.ifft2(G))
        self.display_transformed_img(img_out)

    def lowpass_gauss_filter(self, value):
        img = self.img_rgb
        F = np.fft.fft2(img)
        F = np.fft.fftshift(F)
        M, N = img.shape
        D0 = float(value)
        u = np.arange(0, M) - M / 2
        v = np.arange(0, N) - N / 2
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = np.exp((-1 * np.square(D)) / (2 * D0 ** 2))
        G = H * F
        G = np.fft.ifftshift(G)
        img_out = np.real(np.fft.ifft2(G))
        self.display_transformed_img(img_out)

    def lowpass_butterworth_filter(self, D0, n):
        img = self.img_rgb
        F = np.fft.fft2(img)
        F = np.fft.fftshift(F)
        M, N = img.shape
        u = np.arange(0, M) - M / 2
        v = np.arange(0, N) - N / 2
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = 1 / np.power(1 + (D / D0), (2 * n))
        G = H * F
        G = np.fft.ifftshift(G)
        img_out = np.real(np.fft.ifft2(G))
        self.display_transformed_img(img_out)

    def lowpass_butterworth_change_n(self, value):
        n = float(value)
        D0 = float(self.lowpass_butterworth_d0_scale.get())
        self.lowpass_butterworth_filter(D0, n)

    def lowpass_butterworth_change_d0(self, value):
        D0 = float(value)
        n = float(self.lowpass_butterworth_n_scale.get())
        self.lowpass_butterworth_filter(D0, n)

    def lowpass_ideal_filter(self, value):
        D0 = float(value)
        img = self.img_rgb
        F = np.fft.fft2(img)
        F = np.fft.fftshift(F)
        M, N= img.shape
        u = np.arange(0, M) - M / 2
        v = np.arange(0, N) - N / 2
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = np.array(D <= D0, 'float')
        G = H * F
        G = np.fft.ifftshift(G)
        img_out = np.real(np.fft.ifft2(G))
        self.display_transformed_img(img_out)

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

        img_origin = cv2.imread(image_path, 0)
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
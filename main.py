import os
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImagePipelineExtension:
    def pre_processing_hook(self, img): return img
    def post_processing_hook(self, img): return img

class PassportCropperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Passport Photo Cropper - Production Build")
        self.root.geometry("900x600")
        self.root.configure(bg="#1e1e1e")
        
        self.input_path = None
        self.processed_img_cv = None
        self.pipeline_extension = ImagePipelineExtension()
        
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        )
        self.setup_ui()
        
    def setup_ui(self):
        left_frame = tk.Frame(self.root, width=280, bg="#2d2d2d")
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        left_frame.pack_propagate(False)
        
        lbl_title = tk.Label(left_frame, text="AI Passport Engine", font=("Arial", 16, "bold"), fg="white", bg="#2d2d2d")
        lbl_title.pack(pady=20)
        
        btn_upload = tk.Button(left_frame, text="Upload / Drop Image", command=self.upload_image, bg="#007bff", fg="white", font=("Arial", 11, "bold"), bd=0, height=2)
        btn_upload.pack(pady=15, padx=20, fill="x")
        
        self.btn_process = tk.Button(left_frame, text="Auto AI Align & Crop", command=self.process_image, bg="#555555", fg="white", font=("Arial", 11, "bold"), bd=0, height=2, state="disabled")
        self.btn_process.pack(pady=15, padx=20, fill="x")
        
        self.btn_save = tk.Button(left_frame, text="Export Passport JPEG", command=self.save_image, bg="#555555", fg="white", font=("Arial", 11, "bold"), bd=0, height=2, state="disabled")
        self.btn_save.pack(pady=15, padx=20, fill="x")
        
        lbl_info = tk.Label(left_frame, text="ICAO Pure Standard:\n3.5 x 4.5 cm @ 300 DPI\nResolution: 413 x 531 px", justify="left", fg="#aaaaaa", bg="#2d2d2d", font=("Arial", 10))
        lbl_info.pack(side="bottom", pady=20)
        
        right_frame = tk.Frame(self.root, bg="#1e1e1e")
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.lbl_orig = tk.Label(right_frame, text="Original Preview", bg="#151515", fg="#666666", width=25, height=15)
        self.lbl_orig.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        self.lbl_proc = tk.Label(right_frame, text="Cropped AI Preview", bg="#151515", fg="#666666", width=25, height=15)
        self.lbl_proc.pack(side="right", fill="both", expand=True, padx=5, pady=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")])
        if file_path:
            self.input_path = file_path
            self.btn_process.configure(state="normal", bg="#1f6aa5")
            self.btn_save.configure(state="disabled", bg="#555555")
            self.lbl_proc.configure(text="Ready to process", image="")
            
            img = Image.open(file_path)
            img.thumbnail((300, 380))
            img_tk = ImageTk.PhotoImage(img)
            self.lbl_orig.configure(image=img_tk, text="")
            self.lbl_orig.image = img_tk

    def process_image(self):
        if not self.input_path: return
        img_cv = cv2.imread(self.input_path)
        img_cv = self.pipeline_extension.pre_processing_hook(img_cv)
        h, w, _ = img_cv.shape
        
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            messagebox.showerror("AI Error", "Facial coordinates detection failed.")
            return
            
        landmarks = results.multi_face_landmarks.landmark
        
        # Exact landmark points mapped to scale
        # Index 33 = Left eye outer corner, Index 263 = Right eye outer corner
        left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
        right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
        chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
        top_head = np.array([landmarks[10].x * w, landmarks[10].y * h])
        
        d_y, d_x = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(d_y, d_x))
        if abs(angle) < 1.5: angle = 0.0
        
        eye_center = (float((left_eye[0] + right_eye[0]) / 2), float((left_eye[1] + right_eye[1]) / 2))
        rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        rotated_img = cv2.warpAffine(img_cv, rot_matrix, (w, h), flags=cv2.INTER_CUBIC)
        
        # Linear geometric matrix point transformations
        def rot_pt(pt, M):
            nx = M[0,0]*pt[0] + M[0,1]*pt[1] + M[0,2]
            ny = M[1,0]*pt[0] + M[1,1]*pt[1] + M[1,2]
            return np.array([nx, ny])
            
        r_chin = rot_pt(chin, rot_matrix)
        r_top = rot_pt(top_head, rot_matrix)
        r_eye_center = rot_pt(eye_center, rot_matrix)
        
        head_height = np.linalg.norm(r_top - r_chin)
        target_h = head_height / 0.42
        target_w = (target_h * (3.5 / 4.5)) * 1.22
        
        crop_y_start = int(r_top[1] - (target_h * 0.28))
        crop_y_end = int(crop_y_start + target_h)
        crop_x_start = int(r_eye_center[0] - (target_w / 2))
        crop_x_end = int(crop_x_start + target_w)
        
        pad_t = max(0, -crop_y_start); pad_b = max(0, crop_y_end - h)
        pad_l = max(0, -crop_x_start); pad_r = max(0, crop_x_end - w)
        if pad_t or pad_b or pad_l or pad_r:
            rotated_img = cv2.copyMakeBorder(rotated_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
            crop_y_start += pad_t; crop_y_end += pad_t; crop_x_start += pad_l; crop_x_end += pad_l
            
        cropped = rotated_img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        self.processed_img_cv = cv2.resize(cropped, (413, 531), interpolation=cv2.INTER_LANCZOS4)
        self.processed_img_cv = self.pipeline_extension.post_processing_hook(self.processed_img_cv)
        
        preview_rgb = cv2.cvtColor(self.processed_img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(preview_rgb)
        img_pil.thumbnail((300, 380))
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.lbl_proc.configure(image=img_tk, text="")
        self.lbl_proc.image = img_tk
        self.btn_save.configure(state="normal", bg="#24a159")

    def save_image(self):
        if self.processed_img_cv is None: return
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("Image Files", "*.jpg")])
        if save_path:
            cv2.imwrite(save_path, self.processed_img_cv, [cv2.IMWRITE_JPEG_QUALITY, 100])
            messagebox.showinfo("Success", "HD Passport Photo Saved.")

if __name__ == '__main__':
    root = tk.Tk()
    app = PassportCropperApp(root)
    root.mainloop()

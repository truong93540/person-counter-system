import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from tkinter import Frame, Label, Tk, Button, Scale, HORIZONTAL, filedialog, messagebox, Radiobutton, StringVar, ttk
import time
import os

# Fix lỗi OpenMP trên một số máy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- KHAI BÁO BIẾN TOÀN CỤC ---
points = []
drawing = False
pause = False
confidence_threshold = 0.2  # Mặc định 20%
detection_mode = "vertical"

# --- THAY ĐỔI QUAN TRỌNG: Đặt mặc định là ---
model_path = "best.pt" 

def update_threshold(value):
    global confidence_threshold
    confidence_threshold = float(value) / 100

def update_mode():
    global detection_mode
    detection_mode = mode_var.get()

def process_video(video_path):
    global pause, points, drawing, model_path
    
    cap = cv2.VideoCapture(video_path)
    total_entered = 0
    count = 0
    pause = False
    save_video = False
    previous_time = time.time()
    
    # Biến lưu khung hình hiện tại
    current_frame = None
    
    # Tạo đường dẫn lưu
    video_name = os.path.basename(video_path).split('.')[0]
    save_dir = os.path.join("output_images", video_name)
    image_dir = os.path.join(save_dir, "image")
    video_save_path = os.path.join(save_dir, f"{video_name}_output.mp4")
    
    os.makedirs(image_dir, exist_ok=True)
    entered_ids = set()

    # Load model
    try:
        print(f"Đang tải model: {model_path}...")
        model = YOLO(model_path)
    except Exception as e:
        messagebox.showerror("Lỗi Model", f"Không tìm thấy file model: {model_path}\nLỗi chi tiết: {e}")
        return

    # --- HÀM VẼ CHUỘT ---
    def draw_polygon(event, x, y, flags, param):
        global points, drawing
        if event == cv2.EVENT_LBUTTONDOWN: 
            points.append((x, y))
            drawing = True
        elif event == cv2.EVENT_RBUTTONDOWN: 
            if len(points) > 2:
                points.append(points[0])
                drawing = False
        elif event == cv2.EVENT_MOUSEMOVE and drawing: 
            pass

    def delete_polygon():
        points.clear()

    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', draw_polygon)

    out = None
    if messagebox.askyesno(u"Lưu video", u"Bạn có muốn lưu video không?"):
        frame_width = 1050
        frame_height = 600
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_save_path, fourcc, fps, (frame_width, frame_height))
        save_video = True

    try:
        while True:
            # 1. XỬ LÝ VIDEO
            if not pause:
                ret, frame = cap.read()
                if not ret:
                    break 
                
                frame_riel = cv2.resize(frame, (1050, 600))
                current_frame = frame_riel.copy() 

                # Chạy YOLO Tracking
                results = model.track(frame_riel, tracker="botsort.yaml", persist=True, classes=0, imgsz=640, verbose=False)
                
                current_count_in_zone = 0 

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.int().cpu().tolist()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    confidences = results[0].boxes.conf.float().cpu().tolist()

                    for box, track_id, confidence in zip(boxes, track_ids, confidences):
                        if confidence < confidence_threshold: continue
                        
                        x1, y1, x2, y2 = box
                        
                        # Vẽ khung người (Mỏng lại cho đẹp)
                        cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # ID nhỏ gọn
                        cv2.putText(current_frame, f'{track_id}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Logic đếm
                        if len(points) >= 3:
                            cx, cy = 0, 0
                            if detection_mode == "vertical": cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            elif detection_mode == "horizontal": cx, cy = (x1 + x2) // 2, y2
                            
                            if cv2.pointPolygonTest(np.array(points), (cx, cy), False) >= 0:
                                current_count_in_zone += 1 
                                cv2.circle(current_frame, (cx, cy), 6, (0, 0, 255), -1)
                                
                                if track_id not in entered_ids:
                                    total_entered += 1
                                    entered_ids.add(track_id)
                                    try:
                                        person_image = frame_riel[y1:y2, x1:x2]
                                        if person_image.size != 0:
                                            cv2.imwrite(os.path.join(image_dir, f"{track_id}.jpg"), cv2.resize(person_image, (548, 548)))
                                    except: pass

                # Tính FPS
                count += 1
                current_time = time.time()
                fps_display = 1 / (current_time - previous_time) if (current_time - previous_time) > 0 else 0
                previous_time = current_time

            # 2. VẼ GIAO DIỆN
            if current_frame is not None:
                frame_show = current_frame.copy()

                if len(points) > 0:
                    cv2.polylines(frame_show, [np.array(points)], isClosed=False, color=(0, 0, 255), thickness=2)
                    if len(points) > 1: 
                         cv2.line(frame_show, points[-1], points[0], (255, 0, 0), 1)

                if len(points) > 2 and cv2.norm(np.array(points[-1]) - np.array(points[0])) < 20:
                     cv2.polylines(frame_show, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

                # --- PHẦN HIỂN THỊ THÔNG SỐ (NỀN TRONG SUỐT + CHỮ NHỎ) ---
                
                # Hàm phụ để vẽ chữ có viền (Giúp chữ nổi bật trên nền video)
                def draw_text_transparent(img, text, pos, color, scale=0.8):
                    # Vẽ viền đen trước (dày hơn)
                    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
                    # Vẽ chữ màu đè lên (mỏng hơn)
                    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

                # Dòng 1: Tổng lượt (Màu Đỏ - Red) - Scale 0.8 (Nhỏ hơn cũ 1.5)
                # draw_text_transparent(frame_show, f'TONG LUOT: {total_entered}', (30, 50), (0, 0, 255), scale=0.8)
                
                # Dòng 2: Hiện tại (Màu Xanh Lá - Green)
                draw_text_transparent(frame_show, f'HIEN TAI: {current_count_in_zone}', (30, 90), (0, 255, 0), scale=0.8)
                
                # Dòng 3: FPS (Màu Trắng)
                draw_text_transparent(frame_show, f'FPS: {fps_display:.1f}', (30, 130), (255, 255, 255), scale=0.7)
                
                # ---------------------------------------------------------

                if pause:
                    cv2.putText(frame_show, "PAUSED", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                cv2.imshow("RGB", frame_show)
                
                if save_video and out is not None and not pause:
                    out.write(frame_show)

            key = cv2.waitKey(1)
            if key == ord("q"): break
            elif key == ord("r"): 
                delete_polygon()
                entered_ids.clear()
                total_entered = 0
                current_count_in_zone = 0
            elif key == 32: pause = not pause

    except Exception as e:
        print(f"Lỗi xảy ra: {e}")
    finally:
        cap.release()
        if save_video and out is not None: out.release()
        cv2.destroyAllWindows()

def process_image(image_path):
    global model_path
    image = cv2.imread(image_path)
    if image is None: return
    
    resized_image = cv2.resize(image, (548, 548))
    model = YOLO(model_path)
    results = model(resized_image)
    count = 0
    
    detections = results[0].boxes
    for box in detections:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        if cls == 0 and conf >= confidence_threshold:
            count += 1
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person: {conf:.2f}"
            cv2.putText(resized_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    cvzone.putTextRect(resized_image, f'COUNT: {count}', (50, 50), scale=1, thickness=1)
    cv2.imshow("Detected Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def choose_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
    if video_path: process_video(video_path)

def choose_image():
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path: process_image(image_path)

# --- GIAO DIỆN TKINTER ---
root = Tk()
root.title(u"Hệ thống Đếm người đi qua khu vực")
root.configure(bg="#f0f0f0")
root.option_add("*Font", "Arial 12")

window_width = 600
window_height = 500
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_c = (screen_width - window_width) // 2
y_c = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_c}+{y_c}")

frame_top = Frame(root, bg="#f0f0f0")
frame_top.pack(pady=10)
frame_bot = Frame(root, bg="#f0f0f0")
frame_bot.pack(pady=10)

Label(frame_top, text=u"Ngưỡng phát hiện (%)", bg="#f0f0f0").grid(row=0, column=0, padx=10)
slider_threshold = Scale(frame_top, from_=0, to=100, orient=HORIZONTAL, length=200, command=update_threshold)
slider_threshold.set(20) 
slider_threshold.grid(row=0, column=1, padx=10)

Label(frame_bot, text=u"Chế độ Đếm:", bg="#f0f0f0").grid(row=0, column=0, padx=10)
mode_var = StringVar(value="vertical")
Radiobutton(frame_bot, text="Góc nghiêng (Tâm)", variable=mode_var, value="vertical", command=update_mode, bg="#f0f0f0").grid(row=0, column=1)
Radiobutton(frame_bot, text="Góc thẳng (Chân)", variable=mode_var, value="horizontal", command=update_mode, bg="#f0f0f0").grid(row=0, column=2)

# --- PHẦN CHỈNH SỬA MENU CHỌN MODEL ---
Label(root, text="Chọn Model YOLO:", bg="#f0f0f0").pack(pady=(20, 5))

# Thêm "Mô hình tự train (best.pt)" vào danh sách
model_options = ["Mô hình tự train (best.pt)", "yolo11n", "yolo11s", "yolo11m"]
combo = ttk.Combobox(root, values=model_options, state="readonly", width=30)
combo.set("Mô hình tự train (best.pt)") 
combo.pack(pady=5)

def on_select(event):
    global model_path
    selected = combo.get()
    
    if selected == "Mô hình tự train (best.pt)":
        model_path = "best.pt"
    elif selected == "yolo11n":
        model_path = "yolo11n.pt"
    elif selected == "yolo11s":
        model_path = "yolo11s.pt"
    elif selected == "yolo11m":
        model_path = "yolo11m.pt"
        
    print(f"Đã chuyển sang dùng model: {model_path}")

combo.bind("<<ComboboxSelected>>", on_select)
# --------------------------------------

btn_image = Button(root, text=u"Chọn ảnh", command=choose_image, width=20, height=2, bg="#2196F3", fg="white", font=("Arial", 11, "bold"))
btn_image.pack(pady=10)

btn_video = Button(root, text=u"Chọn video & Chạy", command=choose_video, width=20, height=2, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"))
btn_video.pack(pady=10)

Label(root, text="Phím tắt: 'R' để xóa vùng vẽ, 'Q' để thoát, 'Space' để tạm dừng", bg="#f0f0f0", fg="gray", font=("Arial", 10, "italic")).pack(side="bottom", pady=10)

root.mainloop()
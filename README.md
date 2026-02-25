# PEOPLE COUNTING SYSTEM USING YOLO

## 📌 Overview

**person-counter-system** is a computer vision project for **counting people passing through a defined area** using **YOLO (Ultralytics)** combined with **OpenCV** and **object tracking (BoT-SORT)**.

The system supports:

- People counting from **videos** or **images**
- Manual drawing of a **Region of Interest (ROI)** using the mouse
- Counting the **number of people currently inside the zone**
- Saving cropped person images and output videos
- A user-friendly **Tkinter GUI**
- Using **pretrained YOLO models** or a **custom-trained model (best.pt)**

---

## 🧠 Technologies Used

- Python 3.8+
- OpenCV
- Ultralytics YOLO (YOLOv8 / YOLO11)
- BoT-SORT Tracking
- Tkinter (GUI)
- NumPy
- CvZone

---

## 📂 Project Structure

```
person-counter-system/
│── main.py              # Main application (GUI + inference)
│── train.ipynb          # YOLO training notebook
│── best.pt              # Custom-trained YOLO model (optional)
│── output_images/       # Saved results
│   └── <video_name>/
│       ├── image/       # Cropped person images
│       └── *_output.mp4 # Output video
└── README.md
```

---

## ⚙️ Environment Setup

### 1️⃣ Install Python

Recommended: **Python 3.9 – 3.11**

### 2️⃣ Install Required Libraries

```bash
pip install opencv-python ultralytics cvzone numpy
```

If `pip` causes issues:

```bash
python -m pip install opencv-python ultralytics cvzone numpy
```

---

## 🚀 How to Run

From the project directory:

```bash
python main.py
```

### Usage Steps

1. Select a **YOLO model** (custom `best.pt` or pretrained models)
2. Adjust the **confidence threshold**
3. Choose **Image** or **Video** input
4. Use **left-click** to draw the counting area (ROI)
5. Use **right-click** to close the polygon

---

## 🎮 Keyboard Controls (Video Mode)

| Key     | Function               |
| ------- | ---------------------- |
| `Space` | Pause / Resume         |
| `R`     | Reset ROI and counters |
| `Q`     | Quit                   |

---

## 📸 Output

- **Current number of people inside the ROI**
- Real-time **FPS display**
- Cropped images of detected persons
- Processed output video (optional)

---

## 🧪 YOLO Training (train.ipynb)

The `train.ipynb` notebook is used to:

- Prepare the dataset
- Train a YOLO model using Ultralytics
- Export the trained model as `best.pt`

After training:

1. Copy `best.pt` into the project directory
2. Select **"Custom trained model (best.pt)"** in the GUI

---

## ⚠️ Notes

- If no GPU is available, YOLO will run on **CPU** (slower)
- For NVIDIA GPUs, installing **PyTorch with CUDA** is recommended
- Moderate input video resolution improves FPS

---

## 📌 Possible Improvements

- Entry/exit direction-based counting
- Real-time camera support
- CSV / database logging
- Web or dashboard interface

---

## 👨‍💻 Author: Nguyễn Văn Trường

Contact Email: truong93540@gmail.com

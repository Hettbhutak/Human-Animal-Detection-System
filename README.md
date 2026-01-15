# Human & Animal Detection and Industrial OCR System

**Author:** Het Bhutak  
**Email:** hetbhutak@gmail.com

This workspace contains two complete computer vision systems:

1. **Human & Animal Detection in Videos**  
   Detects and classifies humans and animals in videos using deep learning.
2. **Industrial/Military OCR System**  
   Extracts and structures text from images of industrial/military stenciled boxes.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Systems](#running-the-systems)
  - [A. Human & Animal Detection](#a-human--animal-detection)
  - [B. Industrial OCR System](#b-industrial-ocr-system)
- [Data Format](#data-format)
- [Outputs](#outputs)
- [References](#references)

---

## Project Structure

```
.
├── human_animal_detection.py
├── streamlit_app_part_a.py
├── part_a_requirements.txt
├── test_videos/
├── outputs/
├── Human and Animal Detection.v3i.yolov5pytorch/
│   ├── data.yaml
│   ├── train/valid/test/ (images & YOLO labels)
│   └── README.*
├── Part2/
│   ├── ocr_system_part_b.py
│   ├── streamlit_app_part_b.py
│   ├── requirements.txt
│   ├── outputs/
│   └── test_images/
└── industrial_ocr/
    ├── outputs/
    └── test_images/
```

---

## Setup & Installation

### 1. Clone the Repository

```sh
git clone <repo-url>
cd <repo-folder>
```

### 2. Install Python Dependencies

#### For Human & Animal Detection

```sh
pip install -r part_a_requirements.txt
```

#### For Industrial OCR

```sh
pip install -r Part2/requirements.txt
```

#### Additional Requirement for OCR

- **Tesseract OCR Engine**  
  - Windows: [Download here](https://github.com/UB-Mannheim/tesseract/wiki)
  - Linux: `sudo apt-get install tesseract-ocr`
  - Mac: `brew install tesseract`

---

## Running the Systems

### A. Human & Animal Detection

#### **Streamlit Web App**

```sh
streamlit run streamlit_app_part_a.py
```

- Upload a video in the web UI.
- Process and view annotated results and statistics.

#### **Command Line (Batch Processing)**

```sh
python human_animal_detection.py
```

- Processes all videos in `test_videos/`.
- Annotated videos saved to `outputs/`.

---

### B. Industrial OCR System

#### **Streamlit Web App**

```sh
streamlit run Part2/streamlit_app_part_b.py
```

- Upload an image in the web UI.
- Extracts text, shows structured data, and displays annotated images.

#### **Command Line (Batch Processing)**

```sh
python Part2/ocr_system_part_b.py
```

- Processes all images in `Part2/test_images/`.
- Results saved as JSON and annotated images in `Part2/outputs/`.

---

## Data Format

### Human & Animal Detection

- **Images**: JPEG/PNG in `Human and Animal Detection.v3i.yolov5pytorch/train/valid/test/images/`
- **Labels**: YOLOv5 format in `train/valid/test/labels/`
  - Each line: `<class_id> <x_center> <y_center> <width> <height>` (normalized)
  - `class_id`: `0` = Animal, `1` = Human

### Industrial OCR

- **Images**: Place test images in `Part2/test_images/`
- **Outputs**: JSON files and annotated images in `Part2/outputs/`

---

## Outputs

- **Annotated Videos**: `outputs/annotated_test.mp4` 
- **OCR Results**: `Part2/outputs/<image_name>_ocr_result.json`
- **Annotated OCR Images**: `Part2/outputs/<image_name>_annotated.jpg`
- <img width="862" height="768" alt="image" src="https://github.com/user-attachments/assets/09af1611-b007-48c8-ab0d-3b06473d3968" />
- <img width="771" height="750" alt="image" src="https://github.com/user-attachments/assets/1ed6d784-d34c-46c4-b666-d8d5f1cd0d11" />


---

## References

- [YOLOv5 Format](https://docs.ultralytics.com/datasets/yolo/)
- [Roboflow Dataset](https://universe.roboflow.com/work-0kted/human-and-animal-detection)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## Notes

- For best results, use clear, high-resolution videos/images.
- All processing is offline; no data is sent to external servers.
- For any issues, check the console output or Streamlit sidebar tips.

---

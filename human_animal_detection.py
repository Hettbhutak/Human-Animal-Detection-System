"""
Part A: Human & Animal Detection System
========================================

MODEL SELECTION JUSTIFICATION:
------------------------------
1. Object Detection: Faster R-CNN with ResNet50 backbone
   - Why: More accurate than YOLO alternatives, good for offline deployment
   - Pre-trained on COCO but we'll fine-tune on our custom dataset
   - Provides bounding boxes with confidence scores

2. Classification: ResNet50 for binary classification (human vs animal)
   - Why: Proven architecture, efficient, works well with transfer learning
   - Good balance between accuracy and inference speed
   - Suitable for offline deployment on standard hardware

DATASET CHOICE:
---------------
Using Open Images V7 subset focusing on:
- Humans: Person, Man, Woman, Boy, Girl categories
- Animals: Dog, Cat, Horse, Elephant, Bird, etc.
Alternative: Custom dataset from Roboflow Universe
Justification: High-quality annotations, diverse scenarios, publicly available

PIPELINE DESIGN:
----------------
1. Load video from ./test_videos/
2. Extract frames
3. Run object detection to get bounding boxes
4. Crop detected objects
5. Run classification on each crop
6. Annotate frame with results
7. Save to ./outputs/
"""

import os
import cv2
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import resnet50
import numpy as np
from PIL import Image
import json
from pathlib import Path

# Create directory structure
def setup_directories():
    """Create required directory structure"""
    dirs = ['datasets', 'models', 'test_videos', 'outputs']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    print("✓ Directory structure created")

# ============================================================================
# PART 1: OBJECT DETECTION MODEL
# ============================================================================

class ObjectDetector:
    """
    Faster R-CNN based object detector
    Detects all objects in the frame and returns bounding boxes
    """
    def __init__(self, confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pre-trained Faster R-CNN
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        self.confidence_threshold = confidence_threshold
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def detect(self, image):
        """
        Detect objects in image
        Returns: list of [x1, y1, x2, y2, confidence]
        """
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Transform and add batch dimension
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        # Extract boxes with confidence > threshold
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # Filter by confidence
        mask = scores > self.confidence_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        
        # Combine boxes and scores
        detections = []
        for box, score in zip(filtered_boxes, filtered_scores):
            detections.append([*box, score])
        
        return detections

# ============================================================================
# PART 2: CLASSIFICATION MODEL
# ============================================================================

class HumanAnimalClassifier:
    """
    ResNet50 based binary classifier
    Classifies cropped objects as 'human' or 'animal'
    """
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load ResNet50 and modify for binary classification
        self.model = resnet50(pretrained=True)
        
        # Modify final layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, 2)  # 2 classes: human, animal
        
        # Load trained weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ Loaded classifier weights from {model_path}")
        else:
            print("⚠ Using pre-trained ResNet50 (not fine-tuned for this task)")
            print("  For best results, train on human/animal dataset")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = ['animal','human']
    
    def classify(self, image_crop):
        """
        Classify a cropped image as human or animal
        Returns: (class_name, confidence)
        """
        # Convert to PIL if numpy array
        if isinstance(image_crop, np.ndarray):
            image_crop = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        
        # Transform and add batch dimension
        img_tensor = self.transform(image_crop).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        class_name = self.classes[predicted.item()]
        conf_value = confidence.item()
        
        return class_name, conf_value

# ============================================================================
# TRAINING PIPELINE (Conceptual - requires dataset)
# ============================================================================

def train_classifier(dataset_path='datasets/roboflow', epochs=10):
    """
    Training pipeline for the classifier
    """
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    import yaml
    
    # Load data.yaml
    with open(f'{dataset_path}/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    
    # Create custom dataset class
    class HumanAnimalDataset(Dataset):
        def __init__(self, image_dir, label_dir, transform=None):
            self.image_dir = Path(image_dir)
            self.label_dir = Path(label_dir)
            self.transform = transform
            self.images = list(self.image_dir.glob('*.jpg'))
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            
            # Read label
            label_path = self.label_dir / f"{img_path.stem}.txt"
            with open(label_path, 'r') as f:
                label = int(f.readline().split()[0])  # Get class ID
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    # Create datasets
    train_dataset = HumanAnimalDataset(
        f'{dataset_path}/train/images',
        f'{dataset_path}/train/labels',
        transform=transform
    )
    
    val_dataset = HumanAnimalDataset(
        f'{dataset_path}/valid/images',
        f'{dataset_path}/valid/labels',
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Now train your model...
    # (rest of training code)

# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

class VideoPipeline:
    """
    End-to-end pipeline for processing videos
    """
    def __init__(self, detector, classifier):
        self.detector = detector
        self.classifier = classifier
        
    def process_video(self, video_path, output_path):
        """
        Process a single video file
        1. Read video frame by frame
        2. Detect objects in each frame
        3. Classify each detection
        4. Annotate frame
        5. Write to output video
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing: {video_path}")
        print(f"  Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        frame_count = 0
        detection_stats = {'human': 0, 'animal': 0}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect objects
            detections = self.detector.detect(frame)
            
            # Process each detection
            for detection in detections:
                x1, y1, x2, y2, conf = detection
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Crop object
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Classify
                class_name, class_conf = self.classifier.classify(crop)
                detection_stats[class_name] += 1
                
                # Draw bounding box and label
                color = (0, 255, 0) if class_name == 'human' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name}: {class_conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Write frame
            out.write(frame)
            
            # Progress indicator
            if frame_count % 30 == 0:
                print(f"  Progress: {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        print(f"✓ Completed: {output_path}")
        print(f"  Detection stats: {detection_stats}")
        
    def process_all_videos(self):
        """
        Automatically process all videos in ./test_videos/
        """
        video_dir = Path('test_videos')
        output_dir = Path('outputs')
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = [f for f in video_dir.iterdir() 
                      if f.suffix.lower() in video_extensions]
        
        if not video_files:
            print("⚠ No videos found in ./test_videos/")
            return
        
        print(f"Found {len(video_files)} video(s) to process")
        
        for video_file in video_files:
            output_path = output_dir / f"annotated_{video_file.name}"
            self.process_video(str(video_file), str(output_path))

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("HUMAN & ANIMAL DETECTION SYSTEM")
    print("=" * 60)
    
    # Setup
    setup_directories()
    
    # Initialize models
    print("\n1. Loading Object Detection Model...")
    detector = ObjectDetector(confidence_threshold=0.5)
    
    print("\n2. Loading Classification Model...")
    classifier = HumanAnimalClassifier(model_path='models/classifier.pth')
    
    # Create pipeline
    print("\n3. Initializing Video Pipeline...")
    pipeline = VideoPipeline(detector, classifier)
    
    # Process videos
    print("\n4. Processing Videos...")
    pipeline.process_all_videos()
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("Check ./outputs/ folder for annotated videos")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
CHALLENGES & IMPROVEMENTS:
--------------------------
1. CHALLENGES:
   - Two-stage pipeline is slower than single-stage detectors
   - Classification accuracy depends on detection quality
   - Small objects may be missed or misclassified
   - Memory usage with high-resolution videos

2. IMPROVEMENTS:
   - Fine-tune both models on domain-specific dataset
   - Implement tracking to reduce redundant classifications
   - Add confidence thresholding for uncertain predictions
   - Use TensorRT for faster inference
   - Implement batch processing for multiple crops
   - Add temporal smoothing for video consistency
"""
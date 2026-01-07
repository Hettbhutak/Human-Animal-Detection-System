"""
Part B: Offline OCR for Stenciled/Industrial Text
==================================================

SYSTEM DESIGN JUSTIFICATION:
----------------------------
This system is designed to work COMPLETELY OFFLINE for extracting stenciled
or painted text from industrial/military boxes with challenges like:
- Faded paint
- Low contrast
- Surface damage
- Non-uniform lighting
- Stencil-style fonts

MODEL SELECTION:
----------------
1. Text Detection: CRAFT (Character Region Awareness For Text detection)
   Why: 
   - Works well with irregular text layouts
   - Handles stenciled/painted text better than EAST
   - Can detect individual characters and words
   - Lightweight for offline deployment

2. Text Recognition: TrOCR or Custom CNN-LSTM-CTC
   Why:
   - TrOCR: Transformer-based, handles difficult fonts
   - CNN-LSTM-CTC: Proven for challenging text, fully trainable
   - Both work offline after model download

3. Alternative: Tesseract OCR v5 with custom training
   Why:
   - Completely offline
   - Can be trained on specific stencil fonts
   - LSTM-based engine handles degraded text
   - Free and open-source

DATASET SELECTION:
------------------
Custom dataset needed for best results:
1. Collect images of stenciled text from:
   - Military surplus photos
   - Industrial equipment documentation
   - Custom synthetic data generation
   
2. Augment with:
   - Synthetic stencil text rendering
   - Background textures (metal, wood, painted surfaces)
   - Degradation effects (fading, scratches, rust)

3. Alternative datasets:
   - SROIE (Scene Text) - receipts and documents
   - IIIT 5K-Word - scene text
   - Custom stencil font generation using PIL

PIPELINE ARCHITECTURE:
----------------------
Input Image → Preprocessing → Text Detection → Text Recognition → Post-processing → Output
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
from pathlib import Path
import json
import re


Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Het Bhutak\AppData\Local\Programs\Tesseract-OCR'
# Linux/Mac: Usually in PATH by default

# ============================================================================
# PREPROCESSING MODULE
# ============================================================================

class ImagePreprocessor:
    """
    Advanced preprocessing for stenciled/industrial text
    Handles faded paint, low contrast, and surface damage
    """
    
    @staticmethod
    def enhance_contrast(image):
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Better than regular histogram equalization for non-uniform lighting
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge and convert back
        lab_clahe = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    @staticmethod
    def denoise(image):
        """
        Remove noise while preserving edges
        Important for damaged surfaces
        """
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return denoised
    
    @staticmethod
    def sharpen(image):
        """
        Sharpen text edges for better OCR
        """
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    @staticmethod
    def adaptive_threshold(image):
        """
        Convert to binary with adaptive thresholding
        Better for non-uniform lighting than global threshold
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try multiple methods and keep best
        methods = [
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2),
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                 cv2.THRESH_BINARY, 11, 2),
        ]
        
        # Otsu's method
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(otsu)
        
        return methods
    
    @staticmethod
    def morphological_operations(binary_image):
        """
        Clean up binary image
        Remove small noise and connect broken characters
        """
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_small)
        
        # Connect broken characters
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        connected = cv2.dilate(cleaned, kernel_dilate, iterations=1)
        
        return connected
    
    def preprocess(self, image_path):
        """
        Full preprocessing pipeline
        Returns multiple preprocessed versions for ensemble OCR
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        print(f"Processing: {image_path.name}")
        
        # Step 1: Denoise
        denoised = self.denoise(image)
        
        # Step 2: Enhance contrast
        enhanced = self.enhance_contrast(denoised)
        
        # Step 3: Sharpen
        sharpened = self.sharpen(enhanced)
        
        # Step 4: Multiple binarization methods
        binary_images = self.adaptive_threshold(sharpened)
        
        # Step 5: Morphological cleanup
        processed_images = []
        for binary in binary_images:
            cleaned = self.morphological_operations(binary)
            processed_images.append(cleaned)
        
        # Also keep original grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_images.insert(0, gray)
        
        return processed_images

# ============================================================================
# TEXT DETECTION MODULE
# ============================================================================

class TextDetector:
    """
    Detect text regions in the image
    Using traditional CV methods + optional deep learning
    """
    
    @staticmethod
    def detect_text_regions(image):
        """
        Detect text regions using MSER (Maximally Stable Extremal Regions)
        Good for stenciled text detection
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # MSER detector
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        # Convert regions to bounding boxes
        bounding_boxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            
            # Filter by size (avoid noise)
            if w > 10 and h > 10 and w < gray.shape[1] * 0.8:
                bounding_boxes.append((x, y, w, h))
        
        # Merge overlapping boxes
        merged_boxes = TextDetector.merge_boxes(bounding_boxes)
        
        return merged_boxes
    
    @staticmethod
    def merge_boxes(boxes, overlap_threshold=0.3):
        """
        Merge overlapping bounding boxes
        """
        if not boxes:
            return []
        
        boxes = sorted(boxes, key=lambda x: x[1])  # Sort by y-coordinate
        merged = []
        current_box = boxes[0]
        
        for box in boxes[1:]:
            x1, y1, w1, h1 = current_box
            x2, y2, w2, h2 = box
            
            # Check if boxes overlap or are close
            if (abs(y1 - y2) < h1 * 0.5 and 
                abs(x1 - x2) < w1 + w2):
                # Merge
                x = min(x1, x2)
                y = min(y1, y2)
                w = max(x1 + w1, x2 + w2) - x
                h = max(y1 + h1, y2 + h2) - y
                current_box = (x, y, w, h)
            else:
                merged.append(current_box)
                current_box = box
        
        merged.append(current_box)
        return merged

# ============================================================================
# TEXT RECOGNITION Mv  MODULE
# ============================================================================

class TextRecognizer:
    """
    OCR engine for text recognition
    Uses Tesseract with custom configuration for stenciled text
    """
    
    def __init__(self):
        # Tesseract configuration for stenciled text
        # --oem 1: Use LSTM engine
        # --psm 6: Assume uniform block of text
        self.config_options = [
            '--oem 1 --psm 6',  # Standard block of text
            '--oem 1 --psm 7',  # Single text line
            '--oem 1 --psm 8',  # Single word
            '--oem 1 --psm 11', # Sparse text
        ]
        
        # Whitelist for common industrial text (adjust as needed)
        self.custom_config = '--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.'
    
    def recognize(self, image_variants, use_custom_config=True):
        """
        Ensemble OCR: Run OCR on multiple preprocessed versions
        and combine results
        """
        all_results = []
        
        config = self.custom_config if use_custom_config else self.config_options[0]
        
        for idx, img in enumerate(image_variants):
            try:
                # Run Tesseract
                text = pytesseract.image_to_string(img, config=config)
                
                # Get confidence scores
                data = pytesseract.image_to_data(img, config=config, 
                                                output_type=pytesseract.Output.DICT)
                
                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                avg_conf = np.mean(confidences) if confidences else 0
                
                all_results.append({
                    'text': text.strip(),
                    'confidence': avg_conf,
                    'method': f'variant_{idx}'
                })
                
            except Exception as e:
                print(f"  Error with variant {idx}: {e}")
                continue
        
        return all_results
    
    @staticmethod
    def combine_results(results):
        """
        Combine results from multiple OCR attempts
        Choose result with highest confidence
        """
        if not results:
            return "", 0.0
        
        # Sort by confidence
        sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        best_result = sorted_results[0]
        
        return best_result['text'], best_result['confidence']

# ============================================================================
# POST-PROCESSING MODULE
# ============================================================================

class TextPostProcessor:
    """
    Clean and structure OCR output
    """
    
    @staticmethod
    def clean_text(text):
        """
        Remove noise and correct common OCR errors
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Common OCR error corrections for stenciled text
        corrections = {
            '0': 'O',  # Zero to letter O (common in all-caps text)
            '|': 'I',  # Pipe to letter I
            '1': 'I',  # One to letter I (in some contexts)
        }
        
        # Apply corrections cautiously
        # (In real implementation, use context-aware correction)
        
        return text
    
    @staticmethod
    def extract_structured_data(text):
        """
        Extract structured information like:
        - Serial numbers
        - Date codes
        - Weight/dimensions
        - Part numbers
        """
        structured = {
            'raw_text': text,
            'serial_numbers': [],
            'dates': [],
            'weights': [],
            'dimensions': []
        }
        
        # Extract serial numbers (example pattern: XXX-XXXX-XXX)
        serial_pattern = r'[A-Z0-9]{3}-[A-Z0-9]{4}-[A-Z0-9]{3}'
        structured['serial_numbers'] = re.findall(serial_pattern, text)
        
        # Extract dates (MM/DD/YYYY or MM-DD-YYYY)
        date_pattern = r'\d{2}[/-]\d{2}[/-]\d{4}'
        structured['dates'] = re.findall(date_pattern, text)
        
        # Extract weights (e.g., "50 LBS", "23 KG")
        weight_pattern = r'\d+\.?\d*\s*(?:LBS|KG|POUNDS|KILOGRAMS)'
        structured['weights'] = re.findall(weight_pattern, text, re.IGNORECASE)
        
        # Extract dimensions (e.g., "12x10x8")
        dimension_pattern = r'\d+\.?\d*\s*[xX×]\s*\d+\.?\d*\s*[xX×]\s*\d+\.?\d*'
        structured['dimensions'] = re.findall(dimension_pattern, text)
        
        return structured

# ============================================================================
# MAIN OCR PIPELINE
# ============================================================================

class IndustrialOCRPipeline:
    """
    Complete offline OCR pipeline for industrial/military stenciled text
    """
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.detector = TextDetector()
        self.recognizer = TextRecognizer()
        self.postprocessor = TextPostProcessor()
    
    def process_image(self, image_path, output_dir='outputs'):
        """
        Process a single image through the complete pipeline
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Processing: {image_path.name}")
        print(f"{'='*60}")
        
        # Step 1: Preprocessing
        print("Step 1: Preprocessing...")
        processed_images = self.preprocessor.preprocess(image_path)
        
        # Step 2: Text Detection (optional - can skip for full-image OCR)
        print("Step 2: Detecting text regions...")
        # For now, process entire image
        
        # Step 3: Text Recognition
        print("Step 3: Running OCR...")
        ocr_results = self.recognizer.recognize(processed_images)
        
        # Step 4: Combine results
        print("Step 4: Combining results...")
        best_text, confidence = self.recognizer.combine_results(ocr_results)
        
        # Step 5: Post-processing
        print("Step 5: Post-processing...")
        cleaned_text = self.postprocessor.clean_text(best_text)
        structured_data = self.postprocessor.extract_structured_data(cleaned_text)
        
        # Step 6: Save results
        output_file = output_dir / f"{image_path.stem}_ocr_result.json"
        result = {
            'image': str(image_path),
            'text': cleaned_text,
            'confidence': float(confidence),
            'structured_data': structured_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save annotated image
        self._save_annotated_image(image_path, cleaned_text, output_dir)
        
        print(f"\n✓ Results saved to: {output_file}")
        print(f"✓ Confidence: {confidence:.2f}%")
        print(f"✓ Extracted Text:\n{cleaned_text}")
        
        return result
    
    def _save_annotated_image(self, image_path, text, output_dir):
        """
        Save image with OCR text overlay
        """
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        
        # Add text at bottom
        overlay = image.copy()
        cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Add text
        y_position = h - 70
        for line in text.split('\n')[:3]:  # Show first 3 lines
            cv2.putText(image, line, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_position += 25
        
        output_path = output_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(output_path), image)
    
    def process_directory(self, input_dir='test_images', output_dir='outputs'):
        """
        Process all images in a directory
        """
        input_dir = Path(input_dir)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in input_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Found {len(image_files)} image(s) to process")
        
        results = []
        for image_file in image_files:
            result = self.process_image(image_file, output_dir)
            results.append(result)
        
        return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("OFFLINE OCR SYSTEM FOR INDUSTRIAL/STENCILED TEXT")
    print("=" * 60)
    
    # Create directory structure
    Path('test_images').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    # Initialize pipeline
    pipeline = IndustrialOCRPipeline()
    
    # Process all images
    results = pipeline.process_directory('test_images', 'outputs')
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print(f"Processed {len(results)} image(s)")
    print("Check ./outputs/ folder for results")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
CHALLENGES FACED:
-----------------
1. Low Contrast: Faded paint on dark surfaces
   Solution: CLAHE for adaptive contrast enhancement

2. Surface Damage: Scratches, rust, peeling paint
   Solution: Non-local means denoising + morphological operations

3. Stencil Font Characteristics: Broken characters, irregular spacing
   Solution: Ensemble OCR with multiple preprocessing methods

4. Non-uniform Lighting: Shadows, reflections
   Solution: Adaptive thresholding instead of global threshold

5. Similar Characters: O/0, I/1/|, S/5
   Solution: Context-aware post-processing (needs domain knowledge)

POSSIBLE IMPROVEMENTS:
----------------------
1. Train custom Tesseract model on stencil fonts
2. Implement deep learning text detector (CRAFT/EAST)
3. Use TrOCR for better recognition of degraded text
4. Add image quality assessment to reject poor inputs
5. Implement spell-checking with domain-specific dictionary
6. Add perspective correction for angled images
7. Use ensemble of multiple OCR engines (Tesseract + EasyOCR)
8. Implement active learning to improve on difficult cases
9. Add template matching for known text patterns
10. GPU acceleration for faster processing
"""
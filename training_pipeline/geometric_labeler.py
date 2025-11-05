"""
Geometric Face Shape Auto-Labeler
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from tqdm import tqdm

class GeometricFaceShapeLabeler:
    """Auto-label faces using geometric ratios from landmarks"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Key landmark indices
        self.landmarks_idx = {
            'forehead_top': 10,
            'chin_bottom': 152,
            'left_cheek': 234,
            'right_cheek': 454,
            'left_jaw': 172,
            'right_jaw': 397,
            'nose_bridge': 6
        }
    
    def extract_landmarks(self, image):
        """Extract facial landmarks"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        points = {}
        for name, idx in self.landmarks_idx.items():
            lm = landmarks.landmark[idx]
            points[name] = (int(lm.x * w), int(lm.y * h))
        
        return points
    
    def calculate_face_ratios(self, points):
        """Calculate geometric ratios for classification"""
        if not points:
            return None
        
        # Face length (forehead to chin)
        face_length = abs(points['chin_bottom'][1] - points['forehead_top'][1])
        
        # Face width (cheek to cheek)
        face_width = abs(points['right_cheek'][0] - points['left_cheek'][0])
        
        # Jaw width
        jaw_width = abs(points['right_jaw'][0] - points['left_jaw'][0])
        
        # Ratios
        length_width_ratio = face_length / face_width if face_width > 0 else 0
        jaw_face_ratio = jaw_width / face_width if face_width > 0 else 0
        
        return {
            'length_width_ratio': length_width_ratio,
            'jaw_face_ratio': jaw_face_ratio,
            'face_length': face_length,
            'face_width': face_width,
            'jaw_width': jaw_width
        }
    
    def classify_face_shape(self, ratios):
        """Rule-based classification with confidence score"""
        if not ratios:
            return None, 0.0
        
        lw_ratio = ratios['length_width_ratio']
        jf_ratio = ratios['jaw_face_ratio']
        
        # Classification rules with confidence
        if lw_ratio < 1.15:
            if jf_ratio > 0.85:
                return 'round', 0.75
            else:
                return 'round', 0.60
        
        elif 1.15 <= lw_ratio < 1.35:
            if 0.70 < jf_ratio < 0.85:
                return 'oval', 0.80
            else:
                return 'oval', 0.65
        
        elif lw_ratio >= 1.35:
            if jf_ratio > 0.80:
                return 'square', 0.70
            elif jf_ratio < 0.65:
                return 'heart', 0.70
            else:
                return 'oblong', 0.65
        
        else:
            if jf_ratio > 0.80:
                return 'square', 0.60
            else:
                return 'heart', 0.60
    
    def label_image(self, image_path):
        """Label a single image"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None, 0.0
            
            points = self.extract_landmarks(image)
            ratios = self.calculate_face_ratios(points)
            label, confidence = self.classify_face_shape(ratios)
            
            return label, confidence
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, 0.0
    
    def label_dataset(self, image_dir, output_file):
        """Label entire dataset and save results"""
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        results = []
        
        print(f"🔍 Labeling {len(image_files)} images...")
        
        for img_path in tqdm(image_files):
            label, confidence = self.label_image(img_path)
            
            if label:
                results.append({
                    'filename': img_path.name,
                    'path': str(img_path),
                    'label': label,
                    'confidence': confidence
                })
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print statistics
        label_counts = {}
        low_confidence = 0
        
        for r in results:
            label_counts[r['label']] = label_counts.get(r['label'], 0) + 1
            if r['confidence'] < 0.7:
                low_confidence += 1
        
        print(f"\nLabeled {len(results)} images")
        print(f"Distribution: {label_counts}")
        print(f"Low confidence (<0.7): {low_confidence} images")
        print(f"Saved to: {output_file}")
        
        return results

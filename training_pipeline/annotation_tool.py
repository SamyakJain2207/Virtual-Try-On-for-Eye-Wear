"""
Interactive Annotation Tool for Correcting Geometric Predictions
"""

import cv2
import json
import numpy as np
from pathlib import Path

class FaceShapeAnnotationTool:
    """Interactive tool to correct geometric predictions"""
    
    def __init__(self, predictions_file, output_file):
        self.predictions_file = predictions_file
        self.output_file = output_file
        
        # Load predictions
        with open(predictions_file, 'r') as f:
            self.data = json.load(f)
        
        # Sort by confidence (lowest first - most uncertain)
        self.data.sort(key=lambda x: x['confidence'])
        
        self.current_idx = 0
        self.corrections = []
        
        self.face_shapes = {
            '1': 'round',
            '2': 'oval',
            '3': 'square',
            '4': 'heart',
            '5': 'oblong'
        }
        
        print("Face Shape Annotation Tool")
        print("=" * 60)
        print("Controls:")
        print("  1 = Round    2 = Oval    3 = Square")
        print("  4 = Heart    5 = Oblong")
        print("  SPACE = Keep prediction")
        print("  S = Skip")
        print("  Q = Quit and save")
        print("=" * 60)
    
    def annotate(self, max_annotations=1000):
        """Start annotation process"""
        
        while self.current_idx < min(len(self.data), max_annotations):
            item = self.data[self.current_idx]
            
            # Load image
            image = cv2.imread(item['path'])
            if image is None:
                self.current_idx += 1
                continue
            
            # Resize for display
            display_img = cv2.resize(image, (600, 600))
            
            # Add info overlay
            self.draw_info(display_img, item)
            
            # Show image
            cv2.imshow('Face Shape Annotation', display_img)
            
            # Get user input
            key = cv2.waitKey(0) & 0xFF
            
            # Process input
            if key == ord('q'):
                print("\nQuitting and saving...")
                break
            
            elif key == ord(' '):  # Space - keep prediction
                self.corrections.append({
                    'filename': item['filename'],
                    'path': item['path'],
                    'original_label': item['label'],
                    'corrected_label': item['label'],
                    'confidence': item['confidence'],
                    'status': 'kept'
                })
                self.current_idx += 1
            
            elif key == ord('s'):  # Skip
                self.current_idx += 1
                continue
            
            elif chr(key) in self.face_shapes:  # Number key
                corrected_label = self.face_shapes[chr(key)]
                self.corrections.append({
                    'filename': item['filename'],
                    'path': item['path'],
                    'original_label': item['label'],
                    'corrected_label': corrected_label,
                    'confidence': item['confidence'],
                    'status': 'corrected'
                })
                self.current_idx += 1
            
            # Show progress
            if self.current_idx % 50 == 0:
                self.print_progress()
        
        cv2.destroyAllWindows()
        self.save_corrections()
    
    def draw_info(self, image, item):
        """Draw info overlay on image"""
        # Background for text
        cv2.rectangle(image, (10, 10), (590, 120), (0, 0, 0), -1)
        
        # Text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(image, f"Predicted: {item['label'].upper()}", 
                    (20, 40), font, 1, (0, 255, 255), 2)
        
        cv2.putText(image, f"Confidence: {item['confidence']:.2f}", 
                    (20, 70), font, 0.7, (255, 255, 255), 2)
        
        cv2.putText(image, f"Progress: {self.current_idx + 1}/{len(self.data)}", 
                    (20, 100), font, 0.7, (255, 255, 255), 2)
        
        # Color code by confidence
        color = (0, 255, 0) if item['confidence'] > 0.7 else (0, 165, 255)
        cv2.rectangle(image, (10, 10), (590, 120), color, 3)
    
    def print_progress(self):
        """Print progress statistics"""
        corrected = sum(1 for c in self.corrections if c['status'] == 'corrected')
        kept = sum(1 for c in self.corrections if c['status'] == 'kept')
        
        print(f"\nProgress: {self.current_idx}/{len(self.data)}")
        print(f"   Corrected: {corrected}")
        print(f"   Kept: {kept}")
        print(f"   Total reviewed: {len(self.corrections)}")
    
    def save_corrections(self):
        """Save corrected labels"""
        with open(self.output_file, 'w') as f:
            json.dump(self.corrections, f, indent=2)
        
        print(f"\nSaved {len(self.corrections)} annotations to {self.output_file}")
        
        # Statistics
        corrected = sum(1 for c in self.corrections if c['status'] == 'corrected')
        kept = sum(1 for c in self.corrections if c['status'] == 'kept')
        
        print(f"\nFinal Statistics:")
        print(f"   Total reviewed: {len(self.corrections)}")
        print(f"   Corrected: {corrected} ({corrected/len(self.corrections)*100:.1f}%)")
        print(f"   Kept: {kept} ({kept/len(self.corrections)*100:.1f}%)")


def main():
    """Run annotation tool"""
    
    # Paths (adjust these to your project)
    predictions_file = "data/processed/geometric_labels.json"
    output_file = "data/processed/corrected_labels.json"
    
    tool = FaceShapeAnnotationTool(predictions_file, output_file)
    tool.annotate(max_annotations=1000)  # Annotate up to 1000 images


if __name__ == "__main__":
    main()

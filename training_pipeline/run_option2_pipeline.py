"""
OPTION 2 - Complete Pipeline Master Script
This runs the entire Option 2 workflow:
1. Geometric auto-labeling
2. Manual correction (annotation tool)
3. CNN training with corrected labels

Expected Result: 80-85% accuracy
"""

import os
import sys
from pathlib import Path

# Add project to path
project_path = "/content/drive/MyDrive/VirtualTryOnWebApp" # change it while executing
sys.path.append(project_path)

from training_pipeline.geometric_labeler import GeometricFaceShapeLabeler
from training_pipeline.annotation_tool import FaceShapeAnnotationTool
from training_pipeline.cnn_trainer import ImprovedCNNTrainer


def step1_geometric_labeling(utk_images_dir, output_file):
    """Step 1: Auto-label using geometric rules"""
    
    print("\n" + "="*60)
    print("STEP 1: GEOMETRIC AUTO-LABELING")
    print("="*60)
    
    labeler = GeometricFaceShapeLabeler()
    results = labeler.label_dataset(utk_images_dir, output_file)
    
    print(f"\nStep 1 Complete: {len(results)} images auto-labeled")
    print(f"Saved to: {output_file}")
    
    return results


def step2_manual_correction(geometric_labels_file, corrected_labels_file, max_corrections=1000):
    """Step 2: Manually correct predictions"""
    
    print("\n" + "="*60)
    print("STEP 2: MANUAL CORRECTION TOOL")
    print("="*60)
    print(f"You will review up to {max_corrections} images")
    print("This should take ~1-1.5 hours")
    print("\nPress any key to start...")
    input()
    
    tool = FaceShapeAnnotationTool(geometric_labels_file, corrected_labels_file)
    tool.annotate(max_annotations=max_corrections)
    
    print("\nStep 2 Complete: Corrections saved")
    print(f"Saved to: {corrected_labels_file}")


def step3_train_cnn(corrected_labels_file, project_path):
    """Step 3: Train CNN on corrected labels"""
    
    print("\n" + "="*60)
    print("STEP 3: CNN TRAINING")
    print("="*60)
    
    trainer = ImprovedCNNTrainer(project_path)
    
    # Load corrected dataset
    X, y = trainer.load_corrected_dataset(corrected_labels_file)
    
    # Train
    trainer.train(X, y, epochs=50)
    
    # Save
    model_path = trainer.save_model("face_shape_cnn_option2.h5")
    
    print("\nStep 3 Complete: Model trained and saved")
    print(f"Model: {model_path}")
    
    return trainer


def run_complete_pipeline():
    """Run the complete Option 2 pipeline"""
    
    print("\n" + "="*60)
    print("OPTION 2: BALANCED ACCURACY PIPELINE")
    print("="*60)
    print("\nThis pipeline will:")
    print("1. Auto-label UTK dataset using geometric rules (~5 min)")
    print("2. Let you correct 1000 predictions (~1-1.5 hours)")
    print("3. Train CNN on corrected labels (~30-60 min)")
    print("\nExpected accuracy: 80-85%")
    print("="*60)
    
    # Configuration
    project_path = "/content/drive/MyDrive/VirtualTryOnWebApp"
    utk_images_dir = os.path.join(project_path, "data/raw/UTKFace")
    geometric_labels_file = os.path.join(project_path, "data/processed/geometric_labels.json")
    corrected_labels_file = os.path.join(project_path, "data/processed/corrected_labels.json")
    
    # Check if UTK dataset exists
    if not os.path.exists(utk_images_dir):
        print(f"\nError: UTK dataset not found at {utk_images_dir}")
        print("Please download and extract UTK Face dataset first")
        return
    
    # Step 1: Geometric labeling
    step1_geometric_labeling(utk_images_dir, geometric_labels_file)
    
    # Step 2: Manual correction
    step2_manual_correction(geometric_labels_file, corrected_labels_file, max_corrections=1000)
    
    # Step 3: Train CNN
    trainer = step3_train_cnn(corrected_labels_file, project_path)
    
    print("\n" + "="*60)
    print("OPTION 2 PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nYour trained model is ready to use!")
    print(f"Expected accuracy: 80-85%")
    print(f"\nNext steps:")
    print("1. Test the model on new images")
    print("2. Integrate with your Virtual Try-On app")
    print("3. Deploy to production")


if __name__ == "__main__":
    run_complete_pipeline()

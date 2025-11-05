"""
Updated CNN Trainer with Corrected Labels
This replaces the random label assignment with your corrected labels
"""

import os
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from tqdm import tqdm
import joblib

class ImprovedCNNTrainer:
    """Train CNN with corrected labels from annotation tool"""
    
    def __init__(self, project_path):
        self.project_path = project_path
        self.data_processed_path = Path(project_path) / "data" / "processed"
        self.models_path = Path(project_path) / "models"
        self.models_path.mkdir(exist_ok=True)
        
        self.image_size = (224, 224)
        self.face_shapes = ['round', 'oval', 'square', 'heart', 'oblong']
        self.num_classes = len(self.face_shapes)
        
        self.model = None
        self.history = None
        self.label_encoder = LabelEncoder()
    
    def load_corrected_dataset(self, corrected_labels_file):
        """Load dataset with corrected labels"""
        
        print("Loading corrected labels...")
        
        with open(corrected_labels_file, 'r') as f:
            labels_data = json.load(f)
        
        images = []
        labels = []
        
        print(f"Loading {len(labels_data)} images...")
        
        for item in tqdm(labels_data):
            img_path = item['path']
            label = item['corrected_label']
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            labels.append(label)
        
        X = np.array(images)
        y = np.array(labels)
        
        print(f" Loaded: {X.shape[0]} images")
        print(f"Label distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"   {label}: {count}")
        
        return X, y
    
    def build_cnn_model(self):
        """Build CNN architecture"""
        
        model = models.Sequential([
            # Data Augmentation
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classification head
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("CNN Model Built:")
        self.model.summary()
        
        return model
    
    def train(self, X, y, epochs=50, validation_split=0.2):
        """Train the CNN model"""
        
        if self.model is None:
            self.build_cnn_model()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Save label encoder
        joblib.dump(self.label_encoder, self.models_path / "label_encoder.pkl")
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=validation_split, random_state=42, stratify=y_encoded
        )
        
        print(f"\nTraining set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=str(self.models_path / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        print("\n🚀 Starting training...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print("\nTraining completed!")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        
        y_test_encoded = self.label_encoder.transform(y_test)
        
        test_loss, test_acc = self.model.evaluate(X_test, y_test_encoded, verbose=0)
        
        print(f"\nTest Accuracy: {test_acc:.3f}")
        print(f"Test Loss: {test_loss:.3f}")
        
        return test_acc, test_loss
    
    def save_model(self, filename="face_shape_cnn_corrected.h5"):
        """Save trained model"""
        
        save_path = self.models_path / filename
        self.model.save(save_path)
        
        print(f"\nModel saved: {save_path}")
        
        return str(save_path)


def main():
    """Main training pipeline"""
    
    project_path = "/content/drive/MyDrive/VirtualTryOnWebApp"
    corrected_labels_file = os.path.join(project_path, "data/processed/corrected_labels.json")
    
    # Initialize trainer
    trainer = ImprovedCNNTrainer(project_path)
    
    # Load corrected dataset
    X, y = trainer.load_corrected_dataset(corrected_labels_file)
    
    # Train model
    trainer.train(X, y, epochs=50)
    
    # Save model
    trainer.save_model()
    
    print("\nTraining pipeline complete!")


if __name__ == "__main__":
    main()

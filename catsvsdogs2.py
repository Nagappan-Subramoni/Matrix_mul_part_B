import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory
import time
import argparse


def setup_device(use_gpu=True):
    """Configure TensorFlow to use CPU or GPU"""
    if use_gpu:
        # Check if GPU is available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU available: {len(gpus)} GPU(s) found")
                print(f"GPU names: {[gpu.name for gpu in gpus]}")
                return "GPU"
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
                print("Falling back to CPU")
                tf.config.set_visible_devices([], 'GPU')
                return "CPU"
        else:
            print("No GPU found, using CPU")
            return "CPU"
    else:
        # Force CPU usage
        tf.config.set_visible_devices([], 'GPU')
        print("Forced CPU usage")
        return "CPU"


def load_datasets(data_dir):
    """Load train, validation, and test datasets"""
    train_path = os.path.join(data_dir, 'train')
    validation_path = os.path.join(data_dir, 'validation')
    test_path = os.path.join(data_dir, 'test')

    train_dataset = image_dataset_from_directory(
                   train_path,
                   image_size=(180, 180),
                   batch_size=32)
    validation_dataset = image_dataset_from_directory(
                          validation_path,
                          image_size=(180, 180),
                          batch_size=32)
    test_dataset = image_dataset_from_directory(
                    test_path,
                    image_size=(180, 180),
                    batch_size=32)
    
    return train_dataset, validation_dataset, test_dataset


def create_base_model():
    """Create the base CNN model without data augmentation"""
    inputs = keras.Input(shape=(180, 180, 3))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])
    return model


def get_data_augmented(flip="horizontal", rotation=0.1, zoom=0.2):
    """Create data augmentation pipeline"""
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip(flip),
        keras.layers.RandomRotation(rotation),
        keras.layers.RandomZoom(zoom)
    ])
    return data_augmentation


def create_augmented_model():
    """Create CNN model with data augmentation and dropout"""
    data_augmentation = get_data_augmented()
    
    inputs = keras.Input(shape=(180, 180, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])
    return model


def train_and_evaluate(model, train_dataset, validation_dataset, test_dataset, epochs=10, model_name="Model"):
    """Train model and evaluate performance"""
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    start_time = time.time()
    history = model.fit(train_dataset,
                       epochs=epochs,
                       validation_data=validation_dataset)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")
    
    return history, training_time, test_acc


def main():
    parser = argparse.ArgumentParser(description='Train CNN with CPU or GPU option')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu',
                       help='Device to use for training (default: gpu)')
    parser.add_argument('--data_dir', type=str, 
                       default='C:\\Users\\nagappans\\ai_ml\\ai_ml\\NNProjects\\pictures\\cats_vs_dogs_small',
                       help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs to train (default: 10)')
    
    args = parser.parse_args()
    
    # Setup device
    device_used = setup_device(use_gpu=(args.device == 'gpu'))
    print(f"Using device: {device_used}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, validation_dataset, test_dataset = load_datasets(args.data_dir)
    
    # Train base model
    print("Creating base model...")
    base_model = create_base_model()
    base_history, base_time, base_acc = train_and_evaluate(
        base_model, train_dataset, validation_dataset, test_dataset, 
        epochs=args.epochs, model_name="Base Model"
    )
    
    # Train augmented model
    print("\nCreating augmented model with dropout...")
    aug_model = create_augmented_model()
    aug_history, aug_time, aug_acc = train_and_evaluate(
        aug_model, train_dataset, validation_dataset, test_dataset, 
        epochs=args.epochs, model_name="Augmented Model"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Device used: {device_used}")
    print(f"Epochs: {args.epochs}")
    print(f"\nBase Model:")
    print(f"  Training time: {base_time:.2f} seconds")
    print(f"  Test accuracy: {base_acc:.3f}")
    print(f"\nAugmented Model:")
    print(f"  Training time: {aug_time:.2f} seconds")
    print(f"  Test accuracy: {aug_acc:.3f}")
    print(f"\nImprovement: {(aug_acc - base_acc):.3f}")


if __name__ == "__main__":
    # If running without command line arguments, use default settings
    try:
        main()
    except SystemExit:
        # Fallback for environments that don't support argparse
        print("Running with default settings...")
        
        # Default configuration
        data_dir = 'C:\\Users\\nagappans\\ai_ml\\ai_ml\\NNProjects\\pictures\\cats_vs_dogs_small'
        epochs = 10
        use_gpu = True  # Change to False to force CPU usage
        
        # Setup device
        device_used = setup_device(use_gpu=use_gpu)
        print(f"Using device: {device_used}")
        
        # Load datasets
        print("Loading datasets...")
        train_dataset, validation_dataset, test_dataset = load_datasets(data_dir)
        
        # Train base model
        print("Creating base model...")
        base_model = create_base_model()
        base_history, base_time, base_acc = train_and_evaluate(
            base_model, train_dataset, validation_dataset, test_dataset, 
            epochs=epochs, model_name="Base Model"
        )
        
        # Train augmented model
        print("\nCreating augmented model with dropout...")
        aug_model = create_augmented_model()
        aug_history, aug_time, aug_acc = train_and_evaluate(
            aug_model, train_dataset, validation_dataset, test_dataset, 
            epochs=epochs, model_name="Augmented Model"
        )
        
        # Summary
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Device used: {device_used}")
        print(f"Epochs: {epochs}")
        print(f"\nBase Model:")
        print(f"  Training time: {base_time:.2f} seconds")
        print(f"  Test accuracy: {base_acc:.3f}")
        print(f"\nAugmented Model:")
        print(f"  Training time: {aug_time:.2f} seconds")
        print(f"  Test accuracy: {aug_acc:.3f}")
        print(f"\nImprovement: {(aug_acc - base_acc):.3f}")
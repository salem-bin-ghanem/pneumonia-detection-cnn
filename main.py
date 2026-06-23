"""
Pneumonia Detection using Convolutional Neural Networks (CNN)

This script trains a CNN model to classify chest X-ray images into two classes:
NORMAL and PNEUMONIA.

Project steps:
1. Load chest X-ray images from train, validation, and test folders.
2. Preprocess images by resizing and normalizing pixel values.
3. Build a CNN model using TensorFlow/Keras.
4. Train the model and monitor validation performance.
5. Evaluate the trained model using accuracy, classification report, and confusion matrix.
6. Save plots and the trained model inside the results folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

# Configuration

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"
RESULTS_DIR = "results"

MODEL_PATH = os.path.join(RESULTS_DIR, "pneumonia_cnn_model.keras")


def create_results_folder():
    """
    Create the results folder if it does not already exist.

    Input:
        None

    Output:
        None
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    """
    Load and preprocess the chest X-ray dataset using ImageDataGenerator.

    The images are resized to IMAGE_SIZE and normalized by rescaling pixel values
    from the range [0, 255] to [0, 1].

    Input:
        None

    Output:
        train_generator: Generator for training images.
        val_generator: Generator for validation images.
        test_generator: Generator for test images.
    """

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    validation_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    val_generator = validation_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    test_generator = validation_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def build_cnn_model():
    """
    Build a Convolutional Neural Network for binary image classification.

    The model uses convolutional layers for feature extraction, max pooling for
    dimensionality reduction, dropout to reduce overfitting, and dense layers
    for final classification.

    Input:
        None

    Output:
        model: Compiled Keras CNN model.
    """

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation="relu"),
        Dropout(0.5),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(model, train_generator, val_generator):
    """
    Train the CNN model using the training and validation datasets.

    Early stopping is used to stop training when validation loss stops improving.
    ModelCheckpoint saves the best model during training.
    ReduceLROnPlateau is used to reduce learning rate when validation loss plateaus.
    Dynamic class weights are added to automatically balance majority/minority loss penalties.

    Input:
        model: Compiled Keras CNN model.
        train_generator: Training data generator.
        val_generator: Validation data generator.

    Output:
        history: Training history object containing loss and accuracy values.
    """

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    train_classes = train_generator.classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_classes),
        y=train_classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nApplying Class Weights: {class_weight_dict}")

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    return history


def plot_training_history(history):
    """
    Plot and save training accuracy/loss and validation accuracy/loss.

    Input:
        history: Training history object returned by model.fit().

    Output:
        None
    """

    # Accuracy plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_plot.png"))
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "loss_plot.png"))
    plt.close()


def evaluate_model(model, test_generator):
    """
    Evaluate the trained CNN model on the test dataset.

    This function prints the test accuracy, classification report, and saves
    a confusion matrix plot.

    Input:
        model: Trained Keras CNN model.
        test_generator: Test data generator.

    Output:
        None
    """

    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    predictions = model.predict(test_generator)
    predicted_classes = (predictions > 0.5).astype("int32").flatten()

    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    cm = confusion_matrix(true_classes, predicted_classes)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels
    )

    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()


def main():
    """
    Main function that runs the full pneumonia detection pipeline.

    Input:
        None

    Output:
        None
    """

    print("Starting Pneumonia Detection CNN Project...")

    create_results_folder()

    print("\nLoading dataset...")
    train_generator, val_generator, test_generator = load_data()

    print("\nBuilding CNN model...")
    model = build_cnn_model()
    model.summary()

    print("\nTraining model...")
    history = train_model(model, train_generator, val_generator)

    print("\nSaving training plots...")
    plot_training_history(history)

    print("\nEvaluating model...")
    evaluate_model(model, test_generator)

    print("\nProject completed successfully.")
    print(f"Model and results saved in the '{RESULTS_DIR}' folder.")


if __name__ == "__main__":
    main()

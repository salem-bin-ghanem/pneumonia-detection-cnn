# Pneumonia Detection using Convolutional Neural Networks (CNN)

## Project Description

This project aims to build a machine learning model that can detect pneumonia from chest X-ray images. Pneumonia is a serious lung infection, and early detection can support faster medical diagnosis and treatment.

The model uses a Convolutional Neural Network (CNN) to classify chest X-ray images into two classes:

* NORMAL
* PNEUMONIA

CNNs are suitable for this task because they can automatically learn visual patterns from images, such as edges, textures, shapes, and opacity patterns in the lung area.

## Objectives

The main objectives of this project are:

* Build a deep learning model for chest X-ray image classification.
* Train the model using labeled chest X-ray images.
* Preprocess medical images by resizing and normalizing them.
* Use data augmentation to improve generalization.
* Handle class imbalance using class weights.
* Evaluate model performance using accuracy, precision, recall, F1-score, confusion matrix, ROC curve, and AUC score.
* Visualize the training process and final results using plots.
* Compare repeated runs and select the final model based on average performance and stability.

## Dataset

The project uses a chest X-ray image dataset containing two classes:

* NORMAL
* PNEUMONIA

The dataset is divided into:

* Training set
* Validation set
* Test set

The test set contains 880 images:

* 238 NORMAL images
* 642 PNEUMONIA images

The dataset is imbalanced because the number of PNEUMONIA images is higher than the number of NORMAL images. To reduce bias toward the majority class, class weights are used during training.

## Dataset Structure

The dataset should be placed inside the `data/` folder using the following structure:

```text
data/
│── train/
│   │── NORMAL/
│   │── PNEUMONIA/
│
│── val/
│   │── NORMAL/
│   │── PNEUMONIA/
│
│── test/
│   │── NORMAL/
│   │── PNEUMONIA/
```

If the dataset is not already split, the `split_dataset.py` script can be used to reorganize the dataset into training, validation, and test folders.

## Methodology

The project follows these main steps:

1. Load chest X-ray images from the dataset folders.
2. Resize all images to the same input size.
3. Normalize pixel values from 0-255 to 0-1.
4. Apply data augmentation to the training images.
5. Build a CNN model using TensorFlow/Keras.
6. Train the model using the training dataset.
7. Monitor model performance using the validation dataset.
8. Apply class weights to handle class imbalance.
9. Evaluate the trained model on the test dataset.
10. Save the trained model and visual evaluation results.

## Final Model

The final model is a Convolutional Neural Network for binary image classification.

The final model uses:

* Image size: 160 x 160
* Batch size: 32
* Maximum epochs: 20
* Binary classification with sigmoid output
* Adam optimizer with learning rate 0.0001
* Data augmentation
* Class weights
* Early stopping
* Model checkpoint
* ReduceLROnPlateau callback

The model architecture contains:

* Three convolutional blocks with 32, 64, and 128 filters
* `padding="same"` in the convolutional layers
* Batch normalization after each convolutional layer
* Max pooling layers for dimensionality reduction
* A dense layer with 128 neurons
* Dropout with rate 0.5 to reduce overfitting
* A sigmoid output layer for binary prediction

## Why CNN?

A Convolutional Neural Network is suitable for this project because the input data are images. CNNs can automatically learn spatial features from images, such as edges, textures, and lung opacity patterns.

Compared to a fully connected neural network, CNNs are more efficient for image classification because they preserve spatial information and use fewer parameters.

## Data Preprocessing and Augmentation

The images are resized to 160 x 160 pixels so that all inputs have the same shape.

Pixel values are normalized from the range 0-255 to the range 0-1. This helps the neural network train more efficiently and more stably.

Data augmentation is applied only to the training set. The augmentation includes:

* Rotation
* Zooming
* Width shifting
* Height shifting
* Horizontal flipping

Validation and test images are only normalized, because they should represent unseen real data without artificial augmentation.

## Overfitting Prevention

Several techniques are used to reduce overfitting:

* Data augmentation
* Dropout
* Batch normalization
* Early stopping
* Validation monitoring
* Model checkpointing
* ReduceLROnPlateau

Data augmentation creates slightly modified versions of the training images, which helps the model generalize better. Early stopping stops training when the validation loss stops improving. ReduceLROnPlateau reduces the learning rate when validation loss stops improving, which can make training more stable.

## Class Imbalance

The dataset is imbalanced because PNEUMONIA images are more frequent than NORMAL images.

If the model is trained without considering this imbalance, it may become biased toward the majority class. To reduce this problem, class weights are used during training. This gives more importance to the minority class and helps the model learn both classes more fairly.

## Model Selection

Several model configurations were tested during the project.

One previous model achieved a high single-run accuracy of 96%. However, repeated experiments showed that this model was less stable across multiple runs.

The final model was selected based on average performance and stability across 5 repeated runs, not only based on the highest single accuracy. This makes the final result more reliable and reproducible.

## Final Results Across 5 Runs

The final model was evaluated over 5 repeated runs.

|                    Run |  Accuracy |        AUC | Precision NORMAL | Precision PNEUMONIA | Recall NORMAL | Recall PNEUMONIA | F1-score NORMAL | F1-score PNEUMONIA |
| ---------------------: | --------: | ---------: | ---------------: | ------------------: | ------------: | ---------------: | --------------: | -----------------: |
|                      1 |      0.94 |     0.9779 |             0.91 |                0.95 |          0.87 |             0.97 |            0.89 |               0.96 |
|                      2 |      0.94 |     0.9742 |             0.93 |                0.94 |          0.84 |             0.98 |            0.88 |               0.96 |
|                      3 |      0.90 |     0.9793 |             0.75 |                0.98 |          0.96 |             0.88 |            0.84 |               0.93 |
|                      4 |      0.91 |     0.9809 |             0.76 |                0.98 |          0.96 |             0.89 |            0.85 |               0.93 |
|                      5 |      0.94 |     0.9821 |             0.91 |                0.95 |          0.87 |             0.97 |            0.89 |               0.96 |
|               **Mean** | **0.926** | **0.9789** |        **0.852** |           **0.960** |     **0.900** |        **0.938** |       **0.870** |          **0.948** |
| **Standard Deviation** | **0.017** | **0.0027** |        **0.080** |           **0.017** |     **0.050** |        **0.044** |       **0.021** |          **0.015** |

## Result Interpretation

Across the 5 repeated runs, the final model achieved:

* Average accuracy: 92.6%
* Average AUC score: 0.9789
* Average PNEUMONIA precision: 96.0%
* Average PNEUMONIA recall: 93.8%
* Average PNEUMONIA F1-score: 94.8%

The high AUC score shows that the model separates NORMAL and PNEUMONIA images well.

The PNEUMONIA recall is especially important because false negatives are more critical in medical image classification. A false negative means that a pneumonia case is incorrectly classified as NORMAL.

The results show that the final model performs strongly and consistently, especially for detecting pneumonia cases. The low standard deviation of accuracy and AUC also shows that the model is relatively stable across repeated runs.

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* ROC curve
* AUC score

Accuracy shows the overall percentage of correctly classified images.

Precision shows how many predicted positive cases were actually correct.

Recall shows how many real positive cases were detected by the model.

F1-score balances precision and recall.

The confusion matrix shows correct and incorrect predictions for each class.

The ROC curve and AUC score show how well the model separates the two classes at different thresholds.

## Results Folder

The `results/` folder contains the outputs of the final model runs.

```text
results/
│── run1/
│   │── accuracy_plot.png
│   │── loss_plot.png
│   │── confusion_matrix.png
│   │── classification_report.png
│   │── roc_curve.png
│   │── pneumonia_cnn_model.keras
│
│── run2/
│── run3/
│── run4/
│── run5/
│
│── runs_summary.csv
```

Each run folder contains the plots, classification report, confusion matrix, ROC curve, and saved model for that run.

The ` runs_summary.csv` file contains the performance metrics for all 5 runs, including the mean and standard deviation.

## Tools and Technologies

The project was developed using:

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* OpenCV
* Scikit-learn
* Visual Studio Code
* Git and GitHub

## Project Structure

```text
pneumonia-detection-cnn/
│── README.md
│── main.py
│── split_dataset.py
│── requirements.txt
│── .gitignore
│
│── data/
│   │── train/
│   │── val/
│   │── test/
│
│── results/
│   │── run1/
│   │── run2/
│   │── run3/
│   │── run4/
│   │── run5/
│   │──  runs_summary.csv
```

## Installation

Clone the repository:

```bash
git clone https://github.com/salem-bin-ghanem/pneumonia-detection-cnn.git
cd pneumonia-detection-cnn
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## How to Run the Project

Make sure the dataset is placed inside the `data/` folder using the correct structure.

If the dataset is not already split, run:

```bash
python split_dataset.py
```

To train and evaluate the CNN model, run:

```bash
python main.py
```

After running the script, the output files will be saved inside the `results/` folder.

## Limitations

This project is for educational purposes only and should not be used as a real medical diagnosis system.

The main limitations are:

* The dataset is imbalanced.
* The model was trained on a limited dataset.
* The model may not generalize perfectly to X-ray images from different hospitals or imaging devices.
* The model only classifies images into NORMAL and PNEUMONIA.
* Real medical diagnosis requires clinical evaluation by medical professionals.

## Future Improvements

Possible future improvements include:

* Using a larger and more diverse dataset.
* Testing transfer learning models such as VGG16, ResNet, or EfficientNet.
* Performing more hyperparameter tuning.
* Adding Grad-CAM or heatmaps to explain which lung regions influenced the prediction.
* Comparing multiple CNN architectures.
* Improving threshold selection to reduce false negatives.

## Team Members

* Zoltan Buday
* Tsogjargal Ganbaatar
* Salem Bin Ghanem

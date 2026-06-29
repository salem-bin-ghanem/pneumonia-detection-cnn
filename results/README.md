# Results Folder

This folder contains the final evaluation results of the pneumonia detection CNN model.

The final model was evaluated over five repeated runs in order to measure not only the best single performance, but also the average performance and stability of the model.

## Folder Structure

```text
results/
│── run1/
│   │── accuracy_plot.png
│   │── loss_plot.png
│   │── confusion_matrix.png
│   │── classification_report.png
│   │── roc_curve.png
│
│── run2/
│   │── accuracy_plot.png
│   │── loss_plot.png
│   │── confusion_matrix.png
│   │── classification_report.png
│   │── roc_curve.png
│
│── run3/
│   │── accuracy_plot.png
│   │── loss_plot.png
│   │── confusion_matrix.png
│   │── classification_report.png
│   │── roc_curve.png
│
│── run4/
│   │── accuracy_plot.png
│   │── loss_plot.png
│   │── confusion_matrix.png
│   │── classification_report.png
│   │── roc_curve.png
│
│── run5/
│   │── accuracy_plot.png
│   │── loss_plot.png
│   │── confusion_matrix.png
│   │── classification_report.png
│   │── roc_curve.png
│
│── runs_summary.csv
│── README.md
```

## Content of Each Run Folder

Each run folder contains the visual and numerical evaluation outputs for one training and testing run of the final CNN model.

The files included in each run folder are:

* `accuracy_plot.png`: shows the training and validation accuracy during training.
* `loss_plot.png`: shows the training and validation loss during training.
* `confusion_matrix.png`: shows the correct and incorrect predictions for the NORMAL and PNEUMONIA classes.
* `classification_report.png`: shows precision, recall, F1-score, and support for both classes.
* `roc_curve.png`: shows the ROC curve and AUC score for the model.

## Summary File

The file `runs_summary.csv` contains the performance metrics for all five runs, including:

* Accuracy
* AUC
* Precision for NORMAL
* Precision for PNEUMONIA
* Recall for NORMAL
* Recall for PNEUMONIA
* F1-score for NORMAL
* F1-score for PNEUMONIA
* Mean values across the five runs
* Standard deviation values across the five runs

## Final Average Results

Across the five repeated runs, the final model achieved:

| Metric                      |  Value |
| --------------------------- | -----: |
| Average Accuracy            |  0.926 |
| Average AUC                 | 0.9789 |
| Average PNEUMONIA Precision |  0.960 |
| Average PNEUMONIA Recall    |  0.938 |
| Average PNEUMONIA F1-score  |  0.948 |

## Interpretation

The average accuracy of 92.6% shows that the model correctly classified most test images.

The average AUC score of 0.9789 is close to 1, which indicates strong separation between the NORMAL and PNEUMONIA classes.

The PNEUMONIA recall of 93.8% is especially important because false negatives are more critical in this project. A false negative means that a real pneumonia case is incorrectly classified as NORMAL.

The low standard deviation of accuracy and AUC shows that the model was relatively stable across repeated runs.

## Note About Model Files

The trained `.keras` model files are not included in this repository because of their large file size. However, the model can be generated locally by running `main.py`.

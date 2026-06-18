# Dataset Folder

This folder contains the chest X-ray dataset used for training, validation, and testing the pneumonia detection model.

## Expected Structure

```text
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Folder Description

* train/: Images used to train the CNN model.
* val/: Images used to validate the model during training and monitor performance.
* test/: Images used to evaluate the final performance of the trained model.
* NORMAL/: Chest X-ray images of healthy patients.
* PNEUMONIA/: Chest X-ray images of patients diagnosed with pneumonia.

```
```

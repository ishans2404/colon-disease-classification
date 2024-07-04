# Colon Disease Classification Using EfficientNetB0

This repository contains a Jupyter notebook that implements a deep learning model to classify colon diseases using the [WCE Curated Colon Disease Dataset](https://www.kaggle.com/datasets/francismon/curated-colon-dataset-for-deep-learning). The model leverages the EfficientNetB0 architecture to differentiate between four categories: normal, ulcerative colitis, polyps, and esophagitis.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Notebook Overview](#notebook-overview)
  - [Importing Libraries](#importing-libraries)
  - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
  - [Data Augmentation](#data-augmentation)
  - [Model Building](#model-building)
  - [Model Compilation and Training](#model-compilation-and-training)
  - [Model Evaluation](#model-evaluation)
  - [Visualization](#visualization)
- [Results](#results)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Introduction
Early diagnosis of colon diseases can significantly improve treatment outcomes. This project aims to develop a robust deep learning model to classify various colon diseases using endoscopic images. By leveraging transfer learning with the EfficientNetB0 architecture, the model achieves high accuracy with limited data.

## Dataset
The dataset used in this project is the [WCE Curated Colon Disease Dataset](https://www.kaggle.com/datasets/francismon/curated-colon-dataset-for-deep-learning). It contains images categorized into four classes:
- Normal
- Ulcerative Colitis
- Polyps
- Esophagitis

## Requirements
To run the notebook, you need the following libraries:
- os
- cv2
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- tqdm

You can install the required packages using:
```bash
pip install -r requirements.txt
```

## Notebook Overview
The notebook follows these steps:

### Importing Libraries
I start by importing essential libraries for data manipulation, image processing, visualization, and deep learning.

### Data Loading and Preprocessing
Images and labels are loaded from the dataset directory. Images are resized to 224x224 pixels, and labels are encoded to numerical values.

### Data Augmentation
To improve the model's generalization, I applied data augmentation techniques such as rotation, width/height shift, zoom, and horizontal flip.

### Model Building
I used the EfficientNetB0 model with pre-trained weights from ImageNet and added custom classification layers to adapt it for our specific task.

### Model Compilation and Training
The model is compiled using the Adam optimizer and categorical cross-entropy loss function. It is then trained using the augmented data.

### Model Evaluation
The model's performance is evaluated on a separate test set, and metrics such as accuracy are reported.

### Visualization
Training and validation accuracy and loss are plotted to visualize the learning process and identify potential overfitting or underfitting.

## Results
The final model achieves high accuracy in classifying the different colon diseases. Detailed results and evaluation metrics can be found in the notebook.

## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/colon-disease-classification.git
```
2. Navigate to the project directory:
```bash
cd colon-disease-classification
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```
4. Open the Jupyter notebook:
```bash
jupyter notebook colon_disease_classification.ipynb
```
5. Run the cells in the notebook to train and evaluate the model.

## Acknowledgements
- The dataset used in this project is from [Kaggle](https://www.kaggle.com/datasets/francismon/curated-colon-dataset-for-deep-learning).
- The EfficientNetB0 model is part of the TensorFlow Keras applications.

---

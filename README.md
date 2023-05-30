# 🔴 Artificial neural networks

This repository contains an implementation of an Artificial Neural Network (ANN) in the Artificial_neural_network.ipynb notebook.

## Overview

The Artificial_neural_network.ipynb notebook demonstrates the process of building, training, and evaluating an ANN using TensorFlow and Scikit-learn. The ANN is trained on a dataset called Churn_Modelling.csv, which contains information about bank customers and whether they have churned or not.

## Contents

The notebook consists of the following sections:

- Data Preprocessing: In this section, the dataset is imported and preprocessed. Categorical variables are encoded using label encoding and one-hot encoding. The dataset is split into training and test sets, and feature scaling is applied.

- Building the ANN: This section involves initializing the ANN and adding the input, hidden, and output layers using the Keras Sequential API.

- Training the ANN: The ANN is compiled with appropriate optimizer and loss function, and it is trained on the training set.

- Making Predictions and Evaluating the Model: The trained ANN is used to make predictions on new data and evaluate its performance. The predictions are compared to the actual values using a confusion matrix and accuracy score.

## Usage

To run the code in the notebook, you will need the following dependencies:

- Python 3
- TensorFlow
- Scikit-learn
- NumPy
- Pandas

You can install the necessary dependencies using pip:

```bash
pip install tensorflow scikit-learn numpy pandas
```

Once you have the dependencies installed, you can open and run the Artificial_neural_network.ipynb notebook in Jupyter or any other compatible environment. Follow the instructions in the notebook to execute the code cells step by step.

## Conclusion

The Artificial_neural_network.ipynb notebook provides a practical example of building and training an ANN using TensorFlow and Scikit-learn. It demonstrates the process of preprocessing the data, constructing the ANN architecture, training the model, and evaluating its performance. The notebook can serve as a starting point for implementing ANN models in various domains and datasets.


# 🟠 Convolutional neural networks

This Jupyter Notebook (`Convolutional_neural_network.ipynb`) is an implementation of a Convolutional Neural Network (CNN) using TensorFlow and Keras. It demonstrates the steps involved in building and training a CNN to classify images of cats and dogs.

## Getting Started

To run this notebook, follow the steps below:

1. Click on the "Open in Colab" button or access the notebook through this link: [Convolutional_neural_network.ipynb](https://colab.research.google.com/drive/1_L_Gmm04f-J4a0MwOQrkVYfzhtdGypAS).

2. Mount your Google Drive by running the code cell that starts with `from google.colab import drive`. This step allows access to the necessary dataset files stored in your Google Drive.

3. Follow the code cells sequentially to perform data preprocessing, build the CNN architecture, and train the model.

4. Feel free to modify the code or experiment with different parameters to enhance the model's performance.

## Dataset

The dataset used for this project consists of two folders: `training_set` and `test_set`. Each folder contains subfolders for the respective classes (cats and dogs) with corresponding images.

## Dependencies

The following dependencies are used in this notebook:

- NumPy
- TensorFlow
- Keras

These dependencies are already included in the Google Colab environment.

## Results

The CNN model is trained for 25 epochs on the training set and evaluated on the test set. The model's performance is measured using binary cross-entropy loss and accuracy metrics.

## Making Predictions

A code snippet is provided at the end of the notebook to demonstrate how to make predictions on a single image using the trained model.
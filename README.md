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
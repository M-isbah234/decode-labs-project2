# decode-labs-project2
# Iris Data Classification Pipeline

This repository contains a modular machine learning pipeline built to classify the classic Iris dataset using the K-Nearest Neighbors (KNN) algorithm. Developed as part of the DecodeLabs AI Engineering Internship (Project 2).

## Project Overview
The goal of this project is to build a complete end-to-end classification pipeline following the structured Input-Process-Output framework. It handles data preprocessing, handles data leakage constraints by scaling features correctly, tunes the hyperparameter $K$ using the elbow method, and evaluates final performance using a confusion matrix and classification metrics.

## Pipeline Architecture
1. **Data Preprocessing (Input):** Loads the Iris dataset, performs a stratified 80/20 train-test split to preserve class distribution, and normalizes features using `StandardScaler`.
2. **Hyperparameter Tuning (Process):** Iterates through $K$ values from 1 to 20, plotting the weighted F1 error rate to identify the optimal number of neighbors.
3. **Evaluation (Output):** Trains the final model using the best $K$ value and outputs a classification report along with a visual confusion matrix heatmap.

## Prerequisites
Make sure you have the following packages installed in your Python environment:
* numpy
* matplotlib
* seaborn
* scikit-learn

You can install them all via pip:
```bash
pip install numpy matplotlib seaborn scikit-learn

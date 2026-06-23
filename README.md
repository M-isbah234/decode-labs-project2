# Iris Data Classification with KNN

This is my implementation for Project 2 of the DecodeLabs internship. The goal of this project was to step away from rule-based programming and build a foundational supervised learning pipeline from scratch using the classic Iris dataset and the K-Nearest Neighbors (KNN) algorithm.

## How the Code Works

The notebook is split into a clean, step-by-step workflow:

1. **Data Prep:** Loads the dataset and applies a stratified 80/20 train-test split to ensure all three flower classes are balanced across both sets.
2. **Feature Scaling:** Uses `StandardScaler` to normalize the features, preventing larger numerical scales from biasing the distance calculations.
3. **Finding K:** Iterates through $K$ values from 1 to 20 and uses the Elbow Method (plotting the weighted F1 error rate) to visually determine the optimal number of neighbors.
4. **Final Evaluation:** Trains the final model with the best $K$ value and prints out a clean classification report along with a Seaborn confusion matrix heatmap.

## Requirements

To run this notebook, you will need standard data science libraries installed:

```bash
pip install numpy matplotlib seaborn scikit-learn

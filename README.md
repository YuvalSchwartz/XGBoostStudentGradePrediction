# Student Grade Prediction using XGBoost

This repository contains a machine learning model built with XGBoost to predict students grades in both mathematics and foreign language subjects. The dataset used for training and testing the model is sourced from Kaggle.

## Usage

1. Ensure you have Python installed.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the `main.py` script to generate predictions for students grades.

## Dataset

The dataset used for this project is sourced from [Kaggle](Omer's todo, add link to dataset).

## Model Training

The XGBoost model is trained using the provided dataset. You can adjust hyperparameters and experiment with different configurations to improve performance.
We used Sklearn in order to select most influencing factors.

## Evaluation

MAE evaluation metrics is used to assess the performance of the model. Results are provided in the output of the prediction script.
Evaluator object is created in order to perform all needed evaluations (utils class, as object).

## Statistical tests

Scores are compared to SOTA (LR model) based on 3 statistical tests:
1. Confidence interval.
2. Single-Sample t-test.
3. Confidence interval with bootstraping.

## Authors

Yuval Schwartz, Omer Hanan

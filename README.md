## Bayesian Price Simulation

This repository provides a Bayesian framework to simulate the relationship between product prices and the likelihood of a customer making a purchase. It includes functions for generating synthetic data, building and sampling Bayesian models, evaluating model performance, continual training with new data, and generating conformal predictions.

# Features
Data Generation: Generates synthetic data simulating a "buy" decision based on price.
Bayesian Modeling: Build and sample a logistic regression model using PyMC for binary classification (buy vs. no-buy).
Model Evaluation: Evaluates the model's accuracy and stability using test data and posterior mean estimates.
Continual Training: Updates the model using new data while maintaining the previous model's knowledge.
Conformal Prediction: Implements split conformal prediction to provide prediction sets with a user-defined significance level.
Interactive Prediction: Allows the user to input price values and get corresponding predictions using the trained Bayesian model.,

## How to Use

# Generate Data & Train Model:
Run bayesianpricessimulation.py to generate synthetic data and train the model. The script will display the results of model training and prediction accuracy.

# Interactive User Prediction:
Once the model is trained, the script will prompt you to input a price value. Based on your input, the system will output the predicted probability of a "buy" decision, the corresponding prediction set, and the calibration threshold.

# Model Evaluation & Stability Testing:
You can evaluate the model's stability by running the test_model_stability() function and check how consistent the model's parameters (alpha and beta) are across different trials.

# Continual Training:
If new data becomes available, you can update the model using the update_model_with_new_data() function to incorporate the new information.

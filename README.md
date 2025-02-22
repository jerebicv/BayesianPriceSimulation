#Bayesian Price Simulation
This repository provides a Bayesian framework to simulate the relationship between product prices and the likelihood of a customer making a purchase. It includes functions for generating synthetic data, building and sampling Bayesian models, evaluating model performance, continual training with new data, and generating conformal predictions.

Features
Data Generation: Generates synthetic data simulating a "buy" decision based on price.
Bayesian Modeling: Build and sample a logistic regression model using PyMC for binary classification (buy vs. no-buy).
Model Evaluation: Evaluates the model's accuracy and stability using test data and posterior mean estimates.
Continual Training: Updates the model using new data while maintaining the previous model's knowledge.
Conformal Prediction: Implements split conformal prediction to provide prediction sets with a user-defined significance level.
Interactive Prediction: Allows the user to input price values and get corresponding predictions using the trained Bayesian model.
Requirements
Python 3.x
PyMC3 (or PyMC4, depending on your installation)
NumPy
Pandas
Scikit-learn
ArviZ
Seaborn
Matplotlib
You can install the required dependencies using pip:

bash
Kopiraj
Uredi
pip install pymc3 numpy pandas scikit-learn arviz seaborn matplotlib
File Descriptions
functions.py
This file contains various functions related to the Bayesian price simulation process:

Data Generation:

generate_synthetic_data(n=1000, price_dist="normal", price_params=None)
Generates synthetic data with a binary target variable (buy) based on a logistic model. The price distribution can either be "normal" or "uniform".
Bayesian Model Building & Sampling:

build_and_sample_model(data, draws=2000, tune=1000, init="auto", start=None)
Builds a Bayesian logistic regression model using PyMC3 and performs MCMC sampling.
Model Evaluation:

evaluate_model(trace, test_data=None, threshold=0.5)
Evaluates the model by comparing predicted probabilities to actual test data.
Continual Training:

update_model_with_new_data(old_trace, new_data, draws=2000, tune=1000)
Updates the model using new data, leveraging previous model parameters as starting values.
Conformal Prediction:

conformal_prediction(trace, calib_data, alpha=0.1)
Implements a conformal prediction method using calibration data to generate prediction sets with a specified significance level.
Model Stability Testing:

test_model_stability(model_func, data, n_trials=5)
Evaluates model stability by performing multiple refittings with the same data.
User Interaction Example:

user_example(predict_instance_func)
Allows the user to input a price value and outputs the corresponding prediction set and probability using the conformal prediction function.
bayesianpricessimulation.py
This script demonstrates how to use the functions defined in functions.py to:

Generate synthetic price data.
Build a Bayesian logistic regression model.
Evaluate model performance on test data.
Generate conformal predictions.
Interactively prompt the user for predictions based on a given price.
It includes code for:

Generating a large synthetic dataset using a normal or uniform price distribution.
Splitting the data into training and calibration sets for model evaluation.
Visualizing the price distribution.
Sampling the Bayesian model and evaluating performance.
Using conformal prediction to provide uncertainty-aware predictions.
How to Use
Generate Data & Train Model:

Run bayesianpricessimulation.py to generate synthetic data and train the model. The script will display the results of model training and prediction accuracy.
Interactive User Prediction:

Once the model is trained, the script will prompt you to input a price value. Based on your input, the system will output the predicted probability of a "buy" decision, the corresponding prediction set, and the calibration threshold.
Model Evaluation & Stability Testing:

You can evaluate the model's stability by running the test_model_stability() function and check how consistent the model's parameters (alpha and beta) are across different trials.
Continual Training:

If new data becomes available, you can update the model using the update_model_with_new_data() function to incorporate the new information.
Example Usage
Here is how to use the functions interactively:

python
Kopiraj
Uredi
# Import necessary functions
from functions import *

# Generate synthetic data
data = generate_synthetic_data(n=100000, price_dist="normal", price_params=price_params)

# Split the data into training and calibration sets
train_data, calib_data = train_test_split(data, test_size=0.2, random_state=42)

# Build and sample the Bayesian model
model, trace = build_and_sample_model(train_data, draws=1500, tune=800)

# Evaluate the model's performance on new data
test_data = generate_synthetic_data(n=1000000, price_dist="normal", price_params=price_params)
accuracy = evaluate_model(trace, test_data=test_data, threshold=0.5)
print(f"Model Accuracy: {accuracy:.3f}")

# Use the conformal prediction function
predict_instance_func = conformal_prediction(trace, calib_data, alpha=0.05)

# Launch user example (interactive prediction)
user_example(predict_instance_func)
Contributions
Feel free to fork the repository, create an issue, or submit a pull request if you'd like to contribute improvements, bug fixes, or enhancements.

License
This project is licensed under the MIT License.


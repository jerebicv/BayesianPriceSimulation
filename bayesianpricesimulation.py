import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Get the absolute path of the directory containing functions.py
module_path = os.path.abspath('/content/drive/MyDrive/BayesianPriceSimulation')

# Add the directory to sys.path if it's not already there
if module_path not in sys.path:
    sys.path.append(module_path)

# Now you should be able to import functions
from functions import *

# Define parameters for generating synthetic price data and logistic regression model

price_params = {
    "loc": 155,        # Mean price centered around the middle of the range
    "scale": 5,        # Controls spread; a lower value means less variation
    "min": 15.99,      # Lower bound for valid prices
    "max": 250,        # Upper bound for valid prices
    "low": 15.99,      # Ensures consistency in price range
    "high": 250,       # Ensures consistency in price range
    "alpha_true": 7,   # Intercept in the logistic model; influences baseline buy probability
    "beta_true": -0.05 # Sensitivity of buying probability to price (negative means higher price → lower buy chance)
}

# For interactive prediction, build a model based on a single simulation run.
# The price distribution can be chosen as either:
#   - "uniform" for a shifted price distribution
#   - "normal" for a mixed price distribution

# Generate synthetic dataset with 100,000 samples using a uniform price distribution
# (Switch to "normal" if a mixed distribution is preferred)

data = generate_synthetic_data(n=100000, price_dist="normal", price_params=price_params)
#data = generate_synthetic_data(n=100000, price_dist="uniform", price_params=price_params)

# Split the generated data into training and calibration sets (80% training, 20% calibration)
train_data, calib_data = train_test_split(data, test_size=0.2, random_state=42)

# Create the distribution plot
sns.displot(data, x="price", hue="buy", kind="kde")

# Display the plot
plt.show()

# Build and sample the Bayesian model using the training data.
# It performs 1500 draws for sampling and 800 tuning steps for MCMC.
model, trace = build_and_sample_model(train_data, draws=1500, tune=800)

# Evaluate the performance of the Bayesian model using the calibration data and a 0.5 decision threshold.
evaluate_model(trace, test_data=train_data, threshold=0.5)

# Generate new test data to evaluate model stability
test_data = generate_synthetic_data(n=1000000, price_dist="normal", price_params=price_params)

# Evaluate the performance of the Bayesian model using the calibration data and a 0.5 decision threshold.
evaluate_model(trace, test_data=test_data, threshold=0.5)

# Create conformal prediction function using calibration data and a significance level (alpha) of 0.05.
predict_instance_func = conformal_prediction(trace, test_data, alpha=0.05)

# Launch user example (interactive prompt)
user_example(predict_instance_func)
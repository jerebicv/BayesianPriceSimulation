import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ----------------------------
# 1. Data Generation
# ----------------------------
def generate_synthetic_data(n=1000, price_dist="normal", price_params=None):
    """
    Generates synthetic data for a binary 'buy' decision.
    
    Price distribution can be defined by:
      - price_dist: "normal" or "uniform"
      - price_params: dictionary with distribution parameters.
          For "normal": expected keys: 'loc', 'scale', 'min', 'max'.
                      Defaults: loc=55, scale=15, min=10, max=100.
          For "uniform": expected keys: 'low', 'high'.
                      Defaults: low=10, high=100.
    
    The buy decision is simulated using a logistic model with true parameters:
      alpha_true = -5, beta_true = 0.1
    
    Returns:
      pd.DataFrame: DataFrame with columns 'price' and 'buy'.
    """
    if price_params is None:
        if price_dist == "normal":
            price_params = {"loc": 55, "scale": 15, "min": 10, "max": 100}
        elif price_dist == "uniform":
            price_params = {"low": 10, "high": 100}

    if price_dist == "normal":
        price = np.random.normal(loc=price_params["loc"],
                                 scale=price_params["scale"],
                                 size=n)
        price = np.clip(price, price_params["min"], price_params["max"])
    elif price_dist == "uniform":
        price = np.random.uniform(low=price_params["low"],
                                  high=price_params["high"],
                                  size=n)
    else:
        raise ValueError("price_dist must be either 'normal' or 'uniform'.")

    # Define true logistic parameters for simulation
    alpha_true = price_params["alpha_true"]  # Base intercept (lower means lower probability at high prices)
    beta_true = price_params["beta_true"]  # Smaller slope for a smoother probability curve
    p_buy = 1 / (1 + np.exp(-(alpha_true + beta_true * price)))
    buy = np.random.binomial(n=1, p=p_buy)

    return pd.DataFrame({"price": price, "buy": buy})


# ----------------------------
# 2. Bayesian Model Building & Sampling
# ----------------------------
def build_and_sample_model(data, draws=2000, tune=1000, init="auto", start=None):
    """
    Builds and samples a Bayesian logistic regression model using PyMC.
    
    Model:
        buy ~ Bernoulli(p)  where p = sigmoid(alpha + beta * price)
    
    Priors:
        alpha ~ Normal(0, 10)
        beta  ~ Normal(0, 10)
    
    Parameters:
      data   : pd.DataFrame with columns 'price' and 'buy'
      draws  : int, number of posterior samples to draw
      tune   : int, number of tuning steps
      init   : initialization method for MCMC sampling
      start  : dictionary with starting values for parameters (for continual training)
    
    Returns:
      tuple: (model, trace) where trace is an InferenceData object.
    """
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        p = pm.math.sigmoid(alpha + beta * data["price"].values)
        y_obs = pm.Bernoulli("y_obs", p=p, observed=data["buy"].values)
        trace = pm.sample(draws, tune=tune, init=init, start=start, progressbar=False)
    return model, trace


# ----------------------------
# 3. Model Evaluation
# ----------------------------
def evaluate_model(trace, test_data=None, threshold=0.5):
    """
    Evaluates a Bayesian logistic regression model using the posterior mean estimates.
    
    Parameters:
      trace      : InferenceData object containing posterior samples.
      test_data  : pd.DataFrame with columns 'price' and 'buy'. If None, synthetic test data is generated.
      threshold  : float, threshold for converting predicted probabilities to binary predictions.
    
    Returns:
      float: Accuracy ratio on the test data.
    """
    if test_data is None:
        test_data = generate_synthetic_data(n=200)
    
    alpha_post = trace.posterior["alpha"].mean().item()
    beta_post = trace.posterior["beta"].mean().item()
    test_prices = test_data["price"].values
    test_probs = 1 / (1 + np.exp(-(alpha_post + beta_post * test_prices)))
    test_predictions = (test_probs >= threshold).astype(int)
    accuracy = accuracy_score(test_data["buy"].values, test_predictions)
    return accuracy


# ----------------------------
# 4. Continual Training / Model Updating
# ----------------------------
def update_model_with_new_data(old_trace, new_data, draws=2000, tune=1000):
    """
    Demonstrates continual training by updating the model with new data.
    
    Posterior means from the old trace are used as starting values.
    
    Parameters:
      old_trace: Previous InferenceData object.
      new_data : pd.DataFrame with new data for updating.
      draws    : int, number of posterior samples to draw.
      tune     : int, number of tuning steps.
    
    Returns:
      tuple: (updated_model, updated_trace)
    """
    init_vals = {
        "alpha": old_trace.posterior["alpha"].mean().item(),
        "beta": old_trace.posterior["beta"].mean().item()
    }
    model_new, trace_new = build_and_sample_model(new_data, draws=draws, tune=tune, start=init_vals)
    return model_new, trace_new


# ----------------------------
# 5. Conformal Prediction
# ----------------------------
def conformal_prediction(trace, calib_data, alpha=0.1):
    """
    Implements a simple split conformal prediction method.
    
    Uses calibration data to compute nonconformity scores and determines a threshold.
    
    Returns:
      function: A function that, for a given price value, returns a tuple:
                (prediction_set, predicted_probability, calibration_threshold)
    """
    alpha_post = trace.posterior["alpha"].mean().item()
    beta_post = trace.posterior["beta"].mean().item()
    calib_price = calib_data["price"].values
    calib_prob = 1 / (1 + np.exp(-(alpha_post + beta_post * calib_price)))
    calib_scores = np.abs(calib_data["buy"].values - calib_prob)
    threshold = np.quantile(calib_scores, 1 - alpha)

    def predict_instance(price_value):
        prob = 1 / (1 + np.exp(-(alpha_post + beta_post * price_value)))
        scores = {0: np.abs(0 - prob), 1: np.abs(1 - prob)}
        prediction_set = [cls for cls, score in scores.items() if score <= threshold]
        return prediction_set, prob, threshold

    return predict_instance

# ----------------------------
# 6. Stability Testing
# ----------------------------
def test_model_stability(model_func, data, n_trials=5):
    """
    Evaluates model stability by re-fitting the model multiple times.
    
    Parameters:
      model_func: Function used to build and sample the model.
      data      : pd.DataFrame for model fitting.
      n_trials  : int, number of independent trials.
    
    Returns:
      list: A list of tuples (alpha_est, beta_est) for each trial.
    """
    estimates = []
    for i in range(n_trials):
        _, trace_i = model_func(data, draws=1000, tune=500)
        alpha_est = trace_i.posterior["alpha"].mean().item()
        beta_est = trace_i.posterior["beta"].mean().item()
        estimates.append((alpha_est, beta_est))
        print(f"Trial {i+1}: alpha = {alpha_est:.3f}, beta = {beta_est:.3f}")
    return estimates

# ----------------------------
# 7. User Interaction Example
# ----------------------------
def user_example(predict_instance_func):
    """
    Prompts a user to input a price value and outputs the corresponding prediction.
    
    Parameters:
      predict_instance_func: function that takes a price value and returns a prediction tuple.
    """
    try:
        user_input = input("Enter a price value: ")
        price_value = float(user_input)
    except ValueError:
        print("Invalid input. Please enter a numeric value for the price.")
        return

    pred_set, pred_prob, thresh = predict_instance_func(price_value)
    print(f"\nResults for price: {price_value}")
    print(f"Predicted probability of buy: {pred_prob:.3f}")
    print(f"Conformal prediction set: {pred_set}")
    print(f"Calibration threshold: {thresh:.3f}\n")



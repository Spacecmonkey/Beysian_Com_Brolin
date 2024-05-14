import numpy as np
import matplotlib.pyplot as plt
import pystan

# Generate synthetic data
np.random.seed(42)
t = np.arange(60)
true_trend = 100 + 2 * t
true_seasonality = 10 * np.sin(2 * np.pi * t / 12)
observations = true_trend + true_seasonality + np.random.normal(0, 5, size=60)

# Plot the synthetic data
plt.figure(figsize=(10, 5))
plt.plot(t, observations, label='Observed')
plt.plot(t, true_trend + true_seasonality, label='True Trend + Seasonality', linestyle='dashed')
plt.legend()
plt.show()

# Define the BSTS model in Stan
bsts_model = """
data {
    int<lower=0> T;  // number of time points
    vector[T] y;     // observed values
}
parameters {
    vector[T] trend;
    vector[12] seasonality;  // 12 months in a year
    real<lower=0> sigma_trend;
    real<lower=0> sigma_obs;
}
model {
    // Priors
    trend[1] ~ normal(0, 10);
    for (t in 2:T)
        trend[t] ~ normal(trend[t-1], sigma_trend);  // Random walk
    seasonality ~ normal(0, 10);
    
    // Likelihood
    for (t in 1:T)
        y[t] ~ normal(trend[t] + seasonality[(t % 12) + 1], sigma_obs);
}
"""

# Fit the model using synthetic data
data = {'T': len(observations), 'y': observations}
sm = pystan.StanModel(model_code=bsts_model)
fit = sm.sampling(data=data, iter=1000, chains=4)

# Extract the results
results = fit.extract()
print(fit)

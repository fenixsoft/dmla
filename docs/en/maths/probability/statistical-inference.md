# Statistical Inference

In the [previous chapter](probability-basics.md), we learned how to describe probability distributions and calculate probabilities given distribution parameters. However, in practical applications, we often face the opposite kind of problem: given observed data, how do we infer the parameters of the distribution? This is precisely the problem that **statistical inference** aims to solve.

Statistical inference is the theoretical foundation of model training in machine learning. When we train a model, we are essentially performing statistical inference: inferring model parameters from limited training data, and then using those parameters to predict new data. This chapter introduces two main types of inference methods -- point estimation and interval estimation -- as well as the two statistical philosophies of the frequentist and Bayesian schools.

## Point Estimation

**Estimation** refers to using sample data to infer the value or range of a population parameter. Simply put, we cannot directly observe the entire population (e.g., all people), so we must infer population characteristics from limited samples; this process is estimation.

**Point estimation** uses sample data to compute a specific numerical value as an estimate of a population parameter. For example, using the sample mean $\bar{x}$ to estimate the population mean $\mu$, using the sample proportion $\hat{p}$ to estimate the population proportion $p$, using the sample variance $s^2$ to estimate the population variance $\sigma^2$, and so on -- these all fall under the scope of point estimation.

The problem that point estimation addresses is: given a parameter, what method should be used to compute an estimate from sample data? The same parameter often has multiple possible estimation methods. For instance, to measure a country's household income, the population mean can be estimated using the sample mean, the sample median, or even the average of the sample maximum and minimum. Different estimation methods produce different results -- some are more accurate, some more stable, and some simpler to compute. Therefore, we need a set of principles to evaluate the quality of estimators and to select or construct optimal estimators accordingly. The two most commonly used methods for constructing estimators in statistics are maximum likelihood estimation and maximum a posteriori estimation.

### Maximum Likelihood Estimation

**Maximum Likelihood Estimation (MLE)** is the most classical point estimation method. Its core idea is to choose, among all possible parameter values, the one that makes the "observed data most likely to occur" as the estimate.

Consider an intuitive example: suppose you flip a coin 10 times and observe 8 heads and 2 tails. Now you want to estimate the probability $p$ of heads. $p$ could be 0.5, 0.6, 0.7, 0.8, or any other value, but which value best explains the observed outcome of "8 heads and 2 tails"? Intuitively, $p=0.8$ is the most reasonable choice, because if the true probability of heads were really 0.8, then the probability of observing "8 heads and 2 tails" would indeed be maximized. This is the idea behind MLE: let the data "speak for itself" and choose the parameter that best explains the observed data. Mathematically, given observed data $X = \{x_1, x_2, \ldots, x_n\}$, MLE finds the parameter $\theta$ such that:

$$\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta)$$

where $L(\theta) = P(X|\theta)$ is called the likelihood function, which represents "the probability of observing data $X$ under parameter $\theta$." Note the difference between the likelihood function and the probability function: probability gives the likelihood of data given a fixed parameter, while likelihood measures the plausibility of a parameter given fixed data. MLE follows a standard mathematical procedure:

- **Step 1: Write the likelihood function**. Assuming the samples are independent and identically distributed (i.e., each sample does not affect the others and all come from the same probability distribution), the probability of each observation $x_i$ is $P(x_i|\theta)$, and the probability of all observations occurring simultaneously is the product of the individual probabilities:

    $$L(\theta) = \prod_{i=1}^n P(x_i|\theta)$$

- **Step 2: Take the logarithm to obtain the log-likelihood function**. Directly handling the product form of the likelihood function is cumbersome (multiplying many small numbers yields extremely tiny values), and the product rule makes differentiation complicated. Taking the logarithm transforms the product into a sum, which facilitates computation without affecting the location of the maximum (since the logarithm is a monotonically increasing function):

    $$\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log P(x_i|\theta)$$

- **Step 3: Differentiate with respect to the parameter and set the derivative to zero**. The standard method for finding the maximum of a function is to differentiate, set the derivative to zero, and solve for the parameter value that achieves the extremum:

    $$\frac{\partial \ell(\theta)}{\partial \theta} = 0$$

- **Step 4: Solve the equation to obtain the estimate**. Solving for $\theta$ from the equation in Step 3 yields the maximum likelihood estimate $\hat{\theta}_{MLE}$.

Below are two concrete examples that illustrate how MLE works:

- **Example 1** (MLE for the Bernoulli distribution): The Bernoulli distribution is the simplest probability distribution, with only two possible outcomes (such as heads or tails of a coin), where the parameter $p$ represents the probability of one of the outcomes. The following code simulates a coin-flipping experiment, demonstrates how to estimate $p$ from observed data using MLE, and verifies the result through visualization of the likelihood function curve and mathematical derivation.

    ```python runnable
    import numpy as np
    import matplotlib.pyplot as plt

    # MLE for Bernoulli distribution
    def bernoulli_mle(data):
        """MLE for Bernoulli distribution parameter p"""
        return np.mean(data)  # Proportion of heads

    # Simulate coin flips
    true_p = 0.8
    n = 100
    flips = np.random.binomial(1, true_p, n)

    # MLE estimate
    p_mle = bernoulli_mle(flips)

    print(f"True parameter: p = {true_p}")
    print(f"Observed data: {n} flips, {flips.sum()} heads")
    print(f"MLE estimate: p̂ = {p_mle:.4f}")
    print()

    # Visualize the likelihood function
    p_values = np.linspace(0.01, 0.99, 100)
    k = flips.sum()

    # Likelihood function: L(p) = p^k * (1-p)^(n-k)
    # Log-likelihood: l(p) = k*log(p) + (n-k)*log(1-p)
    log_likelihood = k * np.log(p_values) + (n - k) * np.log(1 - p_values)

    plt.figure(figsize=(10, 5))
    plt.plot(p_values, log_likelihood, 'b-', linewidth=2)
    plt.axvline(p_mle, color='r', linestyle='--', label=f'MLE: p = {p_mle:.2f}')
    plt.axvline(true_p, color='g', linestyle=':', label=f'True value: p = {true_p}')
    plt.xlabel('Parameter p')
    plt.ylabel('Log-likelihood l(p)')
    plt.title(f'Log-likelihood Function of Bernoulli Distribution (n={n}, k={k})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

    # Verification: MLE is the sample mean
    print("Mathematical derivation verification:")
    print(f"  Likelihood function: L(p) = p^{k} * (1-p)^{n-k}")
    print(f"  Log-likelihood: l(p) = {k}*log(p) + {n-k}*log(1-p)")
    print(f"  Set dl/dp = 0: {k}/p - {n-k}/(1-p) = 0")
    print(f"  Solution: p̂ = {k}/{n} = {k/n:.4f}")
    ```

    As the simulation results show, the MLE for the Bernoulli distribution is simply the sample mean $k/n$, which aligns perfectly with intuition. So what value do the likelihood function and this seemingly complex procedure provide? First, it is a **proof rather than a guess** -- intuition tells us that "8 heads and 2 tails" corresponds to $p=0.8$, but the derivation proves that this conclusion holds under any circumstances. Second, it is a **general method** -- the Bernoulli distribution is the simplest example, and its conclusion happens to be intuitive, but the conclusions for other distributions are often not as straightforward (e.g., the variance estimation for the normal distribution in Example 2 below). Finally, the derivation process reveals the essence of MLE: "set the derivative to zero and solve to maximize the likelihood function." Understanding this principle is necessary to judge when MLE should be used and when a better method may exist.

- **Example 2** (MLE for the normal distribution): Unlike the Bernoulli distribution, which requires estimating only one parameter, the normal distribution has two parameters: the mean $\mu$ and the variance $\sigma^2$. Following the standard MLE derivation procedure:
    First, write the likelihood function for the normal distribution:
    $$L(\mu,\sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_i-\mu)^2}{2\sigma^2}}$$
    Then, take the logarithm:
    $$\ell(\mu,\sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i-\mu)^2$$
    Finally, differentiate with respect to $\mu$ and $\sigma^2$ respectively, set the derivatives to zero, and solve:

    - $\hat{\mu}_{MLE} = \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ (sample mean)
    - $\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$ (sample variance)

    The following code generates normally distributed data, computes the estimates using these formulas, and visualizes the difference between the true distribution and the estimated distribution.

    ```python runnable
    import numpy as np
    import matplotlib.pyplot as plt

    # MLE for normal distribution
    true_mu, true_sigma = 5.0, 2.0
    n = 1000
    data = np.random.normal(true_mu, true_sigma, n)
    # MLE estimates
    mu_mle = np.mean(data)
    sigma2_mle = np.mean((data - mu_mle) ** 2)  # MLE uses n as denominator
    sigma_mle = np.sqrt(sigma2_mle)

    print("=== Normal Distribution Parameter Estimation ===")
    print(f"True parameters: μ = {true_mu}, σ = {true_sigma}")
    print(f"MLE estimates: μ̂ = {mu_mle:.4f}, σ̂ = {sigma_mle:.4f}")
    print()

    # Visualization
    x = np.linspace(true_mu - 4*true_sigma, true_mu + 4*true_sigma, 1000)
    def normal_pdf(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    plt.figure(figsize=(10, 6))

    # True distribution
    plt.plot(x, normal_pdf(x, true_mu, true_sigma), 'g-', linewidth=2, label=f'True distribution: N({true_mu}, {true_sigma}²)')
    # MLE estimated distribution
    plt.plot(x, normal_pdf(x, mu_mle, sigma_mle), 'r--', linewidth=2, label=f'MLE estimate: N({mu_mle:.2f}, {sigma_mle:.2f}²)')
    # Histogram
    plt.hist(data, bins=30, density=True, alpha=0.3, color='blue', edgecolor='black')
    plt.xlabel('x')
    plt.ylabel('Probability density')
    plt.title('MLE for Normal Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    ```

### Maximum A Posteriori Estimation

**Maximum A Posteriori Estimation (MAP)** is another point estimation method. Its core idea is to incorporate not only the observed data but also our **prior knowledge** about the parameter, finding the parameter value that maximizes the posterior probability as the estimate.

Consider an intuitive example: suppose you have a coin and flip it 10 times, observing 8 heads and 2 tails. Using MLE, you would estimate $p=0.8$. But if you know beforehand that this coin was provided by a reputable casino (which tends to be fair), then you would consider the $p=0.8$ estimate too extreme; the observed 8 heads out of 10 flips is likely a rare chance event, and the true coin is unlikely to be so biased. In this case, your "prior knowledge" tells you the coin should be fair ($p$ close to 0.5), while the observed data suggests $p$ might be 0.8. MAP combines both: posterior = likelihood x prior, selecting the parameter value that maximizes the posterior probability. The result lies somewhere between 0.5 and 0.8, respecting both the data and the prior. Mathematically, MAP can be expressed as finding the parameter that maximizes the posterior probability $P(\theta|X)$:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|X)$$

According to [Bayes' theorem](probability-basics.md#bayes-theorem), the posterior probability can be expanded as:

$$P(\theta|X) = \frac{P(X|\theta)P(\theta)}{P(X)}$$

Since $P(X)$ is the marginal probability (normalizing constant) of the data, which does not depend on $\theta$, it can be ignored during maximization. Therefore, MAP is equivalent to:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(X|\theta)P(\theta)$$

That is, maximizing "likelihood x prior." Compared to MLE, MAP has an additional factor $P(\theta)$, which is how prior knowledge exerts its influence.

Now that we have learned both MLE and MAP, we can compare them to clarify their respective use cases. MLE and MAP represent two different statistical philosophies. When the prior distribution is uniform (i.e., treating all parameter values equally), $P(\theta)$ is a constant, and MAP gives the same estimate as MLE. When the sample size is very large (approaching infinity), the information content of the data far outweighs the prior, and the influence of the prior is "overwhelmed" (e.g., if a large amount of observed data consistently contradicts the prior, then the prior needs to be revised); in this case, the estimates from MAP and MLE also converge. Therefore, there is no need to use MAP in these scenarios. MAP is primarily useful when data is scarce and prior knowledge is valuable.

| Method | Optimization Objective | Philosophical Stance | Applicable Scenario |
|--------|----------------------|---------------------|-------------------|
| MLE | Maximize likelihood $P(X\|\theta)$ | Frequentist: parameters are fixed values, trust only the data | Sufficient data, no prior knowledge |
| MAP | Maximize posterior $P(\theta\|X) = P(X\|\theta)P(\theta)$ | Bayesian: parameters have a prior distribution, data updates beliefs | Limited data, prior knowledge available |

The following code illustrates the coin-flipping case, providing an intuitive comparison between MLE and MAP. Suppose the true probability of heads is $p=0.7$ (slightly biased toward heads), but we only conducted 10 flips (small sample size). The code uses a Beta(5,5) prior distribution. The Beta distribution is a probability distribution over probability values themselves; the parameters $\alpha=5, \beta=5$ indicate that we believe the coin should be fair ($p$ close to 0.5). The code computes both MLE and MAP estimates and visualizes the relationship among the likelihood function, prior distribution, and posterior distribution.

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# MLE vs MAP: coin flip example
# Assume we suspect the coin is not fair, prior believes p is close to 0.5

true_p = 0.7
n = 10  # Small sample size, prior influence is more pronounced
flips = np.random.binomial(1, true_p, n)
k = flips.sum()

# MLE estimate
p_mle = k / n

# MAP estimate (Beta prior)
# Using Beta(α, β) as prior, equivalent to having pre-observed α-1 heads and β-1 tails
alpha, beta = 5, 5  # Prior believes p ≈ 0.5
p_map = (k + alpha - 1) / (n + alpha + beta - 2)

print(f"True parameter: p = {true_p}")
print(f"Observed data: n={n}, k={k}")
print(f"MLE estimate: p̂ = {p_mle:.4f}")
print(f"MAP estimate (Beta({alpha},{beta}) prior): p̂ = {p_map:.4f}")
print()

# Visualization
p_values = np.linspace(0.01, 0.99, 100)

# Likelihood function
log_likelihood = k * np.log(p_values) + (n - k) * np.log(1 - p_values)

# Prior (Beta distribution)
from math import gamma as gamma_func
def beta_pdf(x, a, b):
    return (gamma_func(a + b) / (gamma_func(a) * gamma_func(b))) * (x ** (a-1)) * ((1-x) ** (b-1))

prior = np.array([beta_pdf(p, alpha, beta) for p in p_values])

# Posterior (proportional to likelihood x prior)
log_posterior = log_likelihood + np.log(prior + 1e-10)

# Normalize for visualization
posterior = np.exp(log_posterior - log_posterior.max())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Likelihood
axes[0].plot(p_values, np.exp(log_likelihood - log_likelihood.max()), 'b-', linewidth=2)
axes[0].axvline(p_mle, color='r', linestyle='--', label=f'MLE: {p_mle:.2f}')
axes[0].set_xlabel('p')
axes[0].set_ylabel('Likelihood (normalized)')
axes[0].set_title('Likelihood Function')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Prior
axes[1].plot(p_values, prior, 'g-', linewidth=2)
axes[1].axvline(0.5, color='r', linestyle='--', label='Prior mean: 0.5')
axes[1].set_xlabel('p')
axes[1].set_ylabel('Prior density')
axes[1].set_title(f'Prior Distribution Beta({alpha}, {beta})')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Posterior
axes[2].plot(p_values, posterior, 'purple', linewidth=2)
axes[2].axvline(p_map, color='r', linestyle='--', label=f'MAP: {p_map:.2f}')
axes[2].axvline(true_p, color='g', linestyle=':', label=f'True value: {true_p}')
axes[2].set_xlabel('p')
axes[2].set_ylabel('Posterior density (normalized)')
axes[2].set_title('Posterior Distribution')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("Key insights:")
print(f"  With a small sample size (n={n}), the MAP estimate is strongly influenced by the prior")
print(f"  MLE may overfit the data (p̂={p_mle:.2f}), while MAP is more robust (p̂={p_map:.2f})")
print(f"  With a large sample size, the prior influence diminishes, and MLE and MAP converge")
```

### Unbiasedness and Consistency

Since the same parameter can often be estimated by multiple methods, how do we judge which estimator is more appropriate? Statistics proposes two important criteria: unbiasedness and consistency.

**Unbiasedness** means that the expected value of the estimator is exactly equal to the true parameter value. In layman's terms, if we repeatedly sample many times and compute an estimate each time using the same estimator, the average of these estimates will converge to the true parameter -- neither systematically too high nor systematically too low. Mathematically, unbiasedness is expressed as: $E[\hat{\theta}] = \theta$. To give a concrete example: using the sample mean $\bar{x}$ to estimate the population mean $\mu$. If we repeatedly sample 100 times and compute the sample mean each time, the average of these 100 sample means will be very close to $\mu$. Therefore, the sample mean is an **unbiased estimate** of the population mean.

However, not all estimators are unbiased. In the earlier [MLE Example 2](#maximum-likelihood-estimation), the variance estimate for the normal distribution was a classic case of bias. This is because the sample variance is computed using the sample mean $\bar{x}$, which is "closer" to the sample data than the true mean $\mu$ (after all, $\bar{x}$ is computed from these very data points). Since the data is closer to $\bar{x}$, $(x_i - \bar{x})^2$ is systematically smaller than $(x_i - \mu)^2$, so MLE underestimates the true variance. Using $n-1$ as the denominator corrects this bias: the denominator becomes smaller, the result becomes larger, which exactly offsets the underestimation. Therefore:

- Using the formula with denominator $n-1$, $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$, yields an **unbiased estimate**.
- Using the formula with denominator $n$, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ (i.e., MLE), yields a **biased estimate**, systematically underestimating the variance.

**Consistency** means that as the sample size approaches infinity, the estimator converges to the true parameter value. In layman's terms, the more data, the more accurate the estimate, eventually "approaching" the true value. Mathematically, this is expressed as: $\hat{\theta}_n \xrightarrow{p} \theta \quad \text{as} \quad n \to \infty$. MLE generally satisfies consistency: as the sample size increases, the MLE estimate gets closer and closer to the true parameter. This is also why MLE and MAP converge with large samples: when enough data is available, the "signal" of the data is strong enough, and the influence of the prior naturally diminishes.

Unbiasedness and consistency, while seemingly similar, focus on different aspects. Unbiasedness is a requirement for the "average performance" of an estimator -- even with a small amount of data, the expected value of the estimator is correct. Consistency is a requirement for the "asymptotic performance" of an estimator -- a large amount of data is needed to ensure the estimate is close to the true value. They are independent criteria: an estimator can be unbiased but inconsistent (e.g., using only the first observation $x_1$ to estimate the mean -- the expectation is correct, but it does not improve with more data), or biased but consistent (e.g., the MLE variance estimate, whose bias diminishes as the sample size increases). In practice, consistency is often more important, because as long as the sample is large enough, the bias of a biased but consistent estimator will disappear, while an unbiased but inconsistent estimator can never accurately estimate the parameter.

## Interval Estimation

Point estimation provides a single numerical value as an estimate of the parameter, which is concise and clear. However, it has a flaw: it does not tell us how reliable the estimate is. For example, suppose we estimate the average income of residents in a city to be 5000 yuan. Was this estimate based on a sample of 10 people or 10,000 people? If it was based on 10 people, the estimate is not very reliable; if based on 10,000, it is much more reliable. A point estimate, being a single number, cannot reflect this difference.

**Interval estimation** solves this problem by providing not a single value, but a range, along with a degree of confidence. For example, saying "the average income is between 4800 and 5200 yuan with 95% confidence" is much more informative than simply saying "5000 yuan" -- it provides both the estimate and a quantification of uncertainty.

A **confidence interval** is the specific form of interval estimation. For a parameter $\theta$, a confidence interval is a random interval $[L, U]$ that satisfies:

$$P(L \leq \theta \leq U) = 1 - \alpha$$

where $\alpha$ is called the significance level, and $1-\alpha$ is called the **confidence level**, typically taken as 0.95 or 0.99 in practice. The true meaning of a 95% confidence interval is not "there is a 95% probability that the parameter lies within this interval," but rather "if we repeatedly sample many times and compute a confidence interval each time, approximately 95% of these intervals will contain the true parameter." The difference between these two statements lies in what is considered random:

- "The parameter has a probability of falling within the interval" implies that the parameter is a random variable and the interval is fixed -- this is the Bayesian perspective.
- "95% of intervals contain the parameter" treats the parameter as a fixed value and the interval as random -- this is the frequentist perspective.

To give a concrete example: suppose the true mean is $\mu=50$. A particular sample yields the confidence interval $[48, 52]$. From the frequentist perspective, $\mu$ is fixed at 50, and the interval $[48, 52]$ either contains 50 (probability = 1) or does not (probability = 0); it does not make sense to say there is a "95% probability" of containing it. The meaning of 95% is that if we repeat the sampling 100 times, approximately 95 of the computed intervals will contain $\mu=50$, and approximately 5 will not. This is an evaluation of the long-run performance of the estimation method, not a probability statement about a specific interval. Once a particular confidence interval is computed, its "fate" is determined -- it either contains the parameter or it does not; there is no probability involved.

The following code demonstrates the meaning of confidence intervals through a simulation experiment: set the true mean $\mu=100$, conduct 50 independent sampling experiments, compute a 95% confidence interval each time, and observe how many intervals contain the true value.

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# Confidence interval demonstration
true_mu = 100
true_sigma = 15
n = 30
n_experiments = 50

# Simulate multiple samples and compute confidence intervals
intervals = []
contains_true = []

for i in range(n_experiments):
    sample = np.random.normal(true_mu, true_sigma, n)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)  # Unbiased estimate
    
    # 95% confidence interval
    # Using t-distribution approximation (normal distribution works for large sample sizes)
    from math import sqrt
    margin = 1.96 * sample_std / sqrt(n)  # Simplified using normal distribution
    lower = sample_mean - margin
    upper = sample_mean + margin
    
    intervals.append((lower, upper))
    contains_true.append(lower <= true_mu <= upper)

# Visualization
plt.figure(figsize=(12, 8))
for i, ((lower, upper), contains) in enumerate(zip(intervals, contains_true)):
    color = 'green' if contains else 'red'
    plt.plot([lower, upper], [i, i], color=color, linewidth=1.5)

plt.axvline(true_mu, color='blue', linestyle='--', linewidth=2, label=f'True mean μ={true_mu}')
plt.xlabel('Mean')
plt.ylabel('Experiment index')
plt.title(f'95% Confidence Intervals ({sum(contains_true)}/{n_experiments} contain true value)')
plt.legend()
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
plt.close()
```

**Standard Error (SE)** is a concept related to the confidence level that quantifies the uncertainty of an interval estimate. It reflects the "degree of fluctuation" of the estimator. The confidence interval = point estimate +/- critical value x standard error. For example, at the 95% confidence level, the confidence interval for the sample mean is $\bar{x} \pm 1.96 \times SE(\bar{x})$ (1.96 is the critical value of the standard normal distribution corresponding to the 95% confidence level), where $SE(\bar{x})$ is the standard error. From this formula, it can be seen that the larger the standard error, the wider the confidence interval (greater estimation uncertainty); the smaller the standard error, the narrower the confidence interval (more precise estimate).

Standard error and standard deviation are similar in name and function, but they focus on different things: standard deviation measures the dispersion of the raw data and describes the data itself; standard error measures the dispersion of the estimator and describes the estimate. For example, suppose we survey 100 people in a city about their income, and the sample mean is 5000 yuan. The incomes of these 100 people vary; standard deviation describes the dispersion of these 100 individual incomes. If we then draw another 100 people, and another 100, repeating this 100 times, we would get 100 different sample means, which also vary among themselves; standard error describes the dispersion of these sample means.

For the sample mean, the standard error is $SE(\bar{x}) = \frac{\sigma}{\sqrt{n}}$. This formula shows that the larger the sample size, the smaller the standard error. Mathematically, this is easy to explain: $n$ is in the denominator, meaning that doubling the sample size reduces the standard error to $\frac{1}{\sqrt{2}} \approx 0.71$ of its original value. This matches our intuition: more data yields more stable estimates with less fluctuation. In practice, the population standard deviation $\sigma$ is usually unknown and must be estimated by the sample standard deviation $s$: $\hat{SE}(\bar{x}) = \frac{s}{\sqrt{n}}$. This is why the term $\frac{\sigma}{\sqrt{n}}$ in the confidence interval formula is actually implemented as $\frac{s}{\sqrt{n}}$.

## Hypothesis Testing

We have covered two types of estimation methods: point estimation provides a specific numerical value for the parameter, and interval estimation provides a plausible range for the parameter. Both are forms of "active exploration" -- we already have observed data and want to know the parameter. In practice, we also encounter another type of "passive judgment" problem: verifying whether a hypothesis about a parameter holds. For example, a new algorithm claims a 10% improvement over the old one -- is this claim credible? Is a feature correlated with the target variable -- is this a real pattern or just random fluctuation in the data? In A/B testing, the new variant performs better -- is this a genuine improvement or random error? These types of problems require a systematic method for assessing the plausibility of hypotheses, which is precisely the domain of **hypothesis testing**.

The logic of hypothesis testing can be likened to a "court trial": we first assume the defendant is innocent (the null hypothesis), then evaluate whether the evidence is strong enough to overturn this assumption. The stronger the evidence (the more extreme the data), the stronger the case for rejecting the presumption of innocence. This analogy highlights two key features of hypothesis testing: first, we do not directly prove that a hypothesis "holds"; rather, we determine whether there is sufficient evidence to "reject" it. Second, the conclusion is always "reject" or "cannot reject"; there is no notion of "accept." Even if we cannot reject the null hypothesis, it only means the evidence is insufficient -- it does not mean the null hypothesis is definitely true.

Let us again use the coin-flipping example to illustrate the workflow of hypothesis testing: suppose we have a coin and want to determine whether it is fair. We flip it 100 times and observe 65 heads and 35 tails. This result deviates considerably from the expectation of "approximately 50 heads for a fair coin," but how much deviation is enough to call it unfair? It could be that random chance caused the deviation, or the coin itself might be biased. We can quantify this judgment through the following steps:

1. **State the hypotheses**. The null hypothesis $H_0$: "the coin is fair, probability of heads $p=0.5$"; the alternative hypothesis $H_1$: "the coin is not fair, $p \neq 0.5$". Note that $H_0$ is the conservative "presumption of innocence," while $H_1$ is the "accusation of guilt" we want to verify.
2. **Choose the significance level**. Set $\alpha = 0.05$, meaning: we are only willing to reject $H_0$ if the data is so extreme that its probability under a fair coin is less than 5%. This is analogous to the "standard of evidence" in court.
3. **Compute the test statistic**. The number of heads $k=65$ is the test statistic; it directly reflects the degree of deviation between the data and the null hypothesis.
4. **Compute the p-value**. If the coin is truly fair ($p=0.5$), what is the probability of observing 65 or more heads (and symmetrically, 35 or fewer heads)? This probability is the p-value, approximately 0.0035 (detailed calculation in the code below).
5. **Make a decision**. p = 0.0035 < α = 0.05, reject $H_0$. Conclusion: there is statistically significant evidence that the coin is not fair.

The most critical and easily misunderstood concept in this process is the p-value. The p-value is the probability of observing the current data, or data more extreme, under the assumption that the null hypothesis is true. To understand the p-value in terms familiar to programmers: imagine you are testing a random number generator that claims to be "fair" and should generate digits 0-9 uniformly. You run a test and observe that out of 100 numbers generated, 90 are 9s. If the generator were truly fair, the probability of observing such an extreme result would be extremely small (a very small p-value). This p-value is not "the probability that the generator is unfair," but rather "how surprised we would be to observe such extreme data if the generator were fair."

Therefore, the p-value is **not** the probability that the null hypothesis is true; rather, it is the degree of surprise at observing such data if the null hypothesis were true. A small p-value indicates a high degree of surprise -- the data is inconsistent with the null hypothesis, giving us reason to doubt it. This understanding is the core idea of hypothesis testing: we never know whether the null hypothesis is truly correct; we only know whether the current data is "compatible" with it. A small p-value means the data is extreme and unlikely under the null hypothesis, giving us reason to doubt it. At the same time, it must be acknowledged that "doubt" is not "falsification" -- extreme data could also be a chance event; it is just that the probability is very low.

Hypothesis testing provides a scientific, quantifiable, and actionable basis for problems in machine learning such as feature selection (testing whether a feature is correlated with the target variable) and model comparison (testing whether model A is significantly better than model B). The following program provides a visual simulation of hypothesis testing.

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# Hypothesis testing demonstration: Is the coin fair?
# H0: p = 0.5 (coin is fair)
# H1: p ≠ 0.5 (coin is not fair)
n = 100
flips = np.random.binomial(1, 0.65, n)  # True p = 0.65 (not fair)
k = flips.sum()

# Compute p-value (two-tailed test)
# Under H0, k ~ Binomial(n, 0.5)
from math import comb

def binomial_pmf(k, n, p):
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

# Compute probability of observing k or more extreme values
p_value = 0
expected = n * 0.5

for i in range(n + 1):
    if abs(i - expected) >= abs(k - expected):  # More extreme than k
        p_value += binomial_pmf(i, n, 0.5)

# Visualization
k_values = np.arange(0, n + 1)
pmf = [binomial_pmf(i, n, 0.5) for i in k_values]

plt.figure(figsize=(12, 5))

# Plot the distribution
plt.bar(k_values, pmf, color='steelblue', edgecolor='black', alpha=0.7, label='Distribution under H0')

# Mark extreme region
extreme_mask = np.array([abs(i - expected) >= abs(k - expected) for i in k_values])
extreme_k = k_values[extreme_mask]
extreme_pmf = np.array(pmf)[extreme_mask]
plt.bar(extreme_k, extreme_pmf, color='red', edgecolor='black', alpha=0.7, label=f'Extreme region (p-value)')

# Mark observed value
plt.axvline(k, color='orange', linestyle='--', linewidth=2, label=f'Observed value k={k}')
plt.axvline(expected, color='green', linestyle=':', linewidth=2, label=f'Expected value E[k]={expected}')

plt.xlabel('Number of heads k')
plt.ylabel('Probability')
plt.title(f'Hypothesis Testing: Is the Coin Fair? (n={n})')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.xlim(30, 70)
plt.tight_layout()
plt.show()
plt.close()

print("=== Hypothesis Testing Results ===")
print(f"H0: Coin is fair (p = 0.5)")
print(f"Observed data: n = {n}, k = {k} (proportion of heads = {k/n:.2%})")
print(f"p-value: {p_value:.4f}")
print(f"Significance level α = 0.05")
print()
if p_value < 0.05:
    print(f"Conclusion: p-value ({p_value:.4f}) < α (0.05), reject H0")
    print("      There is statistically significant evidence that the coin is not fair")
else:
    print(f"Conclusion: p-value ({p_value:.4f}) ≥ α (0.05), cannot reject H0")
    print("      There is not enough evidence to conclude the coin is not fair")
```

## Summary

This chapter introduced the main methods of statistical inference, which form a complete chain from data to conclusions:

**Point estimation** addresses the question "what is the parameter?" MLE relies solely on data, selecting the parameter that best explains the observed results. MAP incorporates prior knowledge, providing more robust estimates when data is insufficient. These two methods represent the different philosophies of the frequentist and Bayesian schools, but they converge under large sample sizes.

**Interval estimation** addresses the question "how reliable is the estimate?" Confidence intervals provide a plausible range for the parameter, and the confidence level quantifies the probability that this range contains the true parameter (from the frequentist perspective) or that the parameter falls within this range (from the Bayesian perspective). Standard error describes the degree of fluctuation of the estimator; the larger the sample size, the more precise the estimate.

**Hypothesis testing** addresses the question "does a hypothesis hold?" The p-value quantifies the degree of agreement between the data and the hypothesis; a small p-value indicates that the data is extreme under the hypothesis, giving reason to doubt it. Hypothesis testing can only "reject" or "cannot reject"; there is no notion of "accept."

These three types of methods are ubiquitous in machine learning: model training is essentially parameter estimation (MLE or MAP), regularization can be interpreted as MAP estimation with a prior, and model evaluation involves confidence intervals and hypothesis testing. With statistical inference as a foundation, we can later understand why machine learning "works" and when it might "fail."

## Exercises

1. Why is the MLE variance estimate biased? Why does dividing by $n-1$ yield an unbiased estimate?
   <details>
   <summary>Reference Answer</summary>

   When we use the sample mean $\bar{x}$ to estimate $\mu$, the sample data is "closer" to $\bar{x}$ than to $\mu$, because $\bar{x}$ is computed from this very data. This causes $\sum(x_i - \bar{x})^2$ to be systematically smaller than $\sum(x_i - \mu)^2$.

   Mathematically, it can be shown that $E[\sum(x_i - \bar{x})^2] = (n-1)\sigma^2$, so dividing by $n-1$ yields an unbiased estimate.

   Intuitively, computing the sample mean uses one "degree of freedom," leaving $n-1$ degrees of freedom.

   </details>

2. Under what circumstances do MLE and MAP give the same estimate?
   <details>
   <summary>Reference Answer</summary>

   When the prior is a uniform distribution (uninformative prior), MLE and MAP give the same estimate. In this case, $P(\theta)$ is a constant, and maximizing the posterior is equivalent to maximizing the likelihood.

   Another case is when the sample size approaches infinity, the influence of the prior is "overwhelmed" by the data, and MLE and MAP converge.

   </details>

3. Explain why a "95% confidence interval" does not mean "there is a 95% probability that the parameter falls within this interval."
   <details>
   <summary>Reference Answer</summary>

   In the frequentist framework, the parameter is a fixed value, not a random variable. Therefore, it is incorrect to say "the parameter has a certain probability of falling somewhere."

   The correct interpretation of a 95% confidence interval is: if we repeatedly sample and compute confidence intervals, approximately 95% of these intervals will contain the true parameter. This is a confidence statement about the "method," not a probability statement about the "parameter."

   In the Bayesian framework, the parameter is treated as a random variable, and a "95% credible interval" does mean that there is a 95% probability that the parameter falls within this interval.

   </details>

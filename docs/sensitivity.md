# Sensitivity Analysis for ODE Models

## Introduction

- **Sensitivity analysis (SA)** quantifies how uncertainty or variation in model parameters influences model outputs.
- In **ordinary differential equation (ODE)** models, parameters represent biological processes (reaction rates, transport constants, synthesis/decay rates, volumes). Their values may be uncertain.
- SA helps to  
  - Evaluate **robustness** and **stability**  
  - Identify **key parameters**  
  - Support **model reduction**, **parameter estimation**, and **experimental design**  
  - Prioritize **data acquisition**
- Two major categories:
  - **Local sensitivity analysis (LSA)** – small perturbations around nominal parameters  
  - **Global sensitivity analysis (GSA)** – variations across full parameter ranges

## Key Concepts

- **General ODE model**  
  `dx/dt = f(x(t), p, t)`, with parameters `p` and outputs `y = g(x,p)`
- **Parameter sensitivity** measures  
  \( \partial y / \partial p_i \)
- **Sensitivity targets** include:
  - State trajectories  
  - Steady states  
  - Peak times and amplitudes  
  - Cost functions  
- **Uncertainty types**
  - Parameter uncertainty  
  - Structural/model uncertainty  
  - Initial-condition uncertainty

    
## Local Sensitivity Analysis (LSA)

### Concept

- Studies the effect of **small** perturbations in parameters.
- Useful when model is calibrated or when exploring local behavior.
- Uses derivatives or numerical approximations.

### Methods

#### Finite Difference Sensitivities

- Perturb one parameter at a time:  
  \( S_i \approx [y(p_i + \Delta) - y(p_i)] / \Delta \)
- Simple and derivative-free.
- Sensitive to numerical noise and step size.

#### Forward Sensitivity Equations (FSE)

- Augment ODE system with sensitivity equations:  
  \( dS_{x,p_i}/dt = (\partial f/\partial x) S_{x,p_i} + (\partial f/\partial p_i) \)
- Accurate and efficient for models with **few parameters**.
- Disadvantage: number of additional ODEs = states × parameters.

#### Adjoint Sensitivity Analysis

- Computes sensitivities of a scalar objective with **many parameters**.
- Solves adjoint ODE backward in time.
- Efficient for high-dimensional parameter spaces.
- More complex to implement.

### Interpretation

- **Scaled sensitivities** allow comparison:  
  \( S_i^{scaled} = (p_i / y)(\partial y/\partial p_i) \)
- Detect:
  - Stiff vs sloppy parameters  
  - Parameters dominating specific observables  
  - Reduction candidates


## 4. Global Sensitivity Analysis (GSA)

### 4.1 Concept

- Investigates the effect of parameter variability across **full parameter ranges**.
- Captures nonlinearities and parameter interactions.
- Essential when parameters are highly uncertain.

### 4.2 Sampling Strategies

- **Monte Carlo (MC)**  
- **Latin Hypercube Sampling (LHS)** – more efficient stratified sampling  
- **Sobol or Halton sequences** – low-discrepancy quasi-random samples
- Parameter distributions based on prior knowledge or uncertainty bounds.

---

## 5. Variance-based Global Sensitivity Methods

### 5.1 Sobol Sensitivity Analysis

- Decomposes output variance into contributions from each parameter.
- Provides:
  - **First-order index** \( S_i \): effect of parameter alone  
  - **Total-order index** \( S_{Ti} \): parameter including interactions  
- Strengths:
  - Model-agnostic  
  - Detects nonlinear interactions  
- Weakness:
  - Computationally expensive for many parameters
- Implementations:
  - SALib, UQLab, pyPESTO, Matlab UQ Toolbox

### 5.2 Extended FAST (Fourier Amplitude Sensitivity Test)

- Uses spectral decomposition to estimate variance contributions.
- Efficient for high-dimensional systems.
- Interpretation for interactions less intuitive than Sobol.

---

## 6. Screening and Factor Prioritization

### 6.1 Morris Method (Elementary Effects)

- Qualitative GSA for identifying influential parameters.
- Computes multiple elementary effects via one-at-a-time perturbations.
- Outputs:
  - **μ**: mean effect → overall importance  
  - **σ**: standard deviation → interactions/nonlinearities  
- Advantages:
  - Very efficient  
  - Suitable for high-dimensional models  
- Disadvantages:
  - Not a full quantitative sensitivity measure

### 6.2 One-Factor-at-a-Time (OAT)

- Varies one parameter while holding others fixed.
- Simple and intuitive.
- Misses interactions → only preliminary use.

---

## 7. Derivative-Based Global Sensitivity Measures (DGSM)

- Based on expected squared derivatives:  
  \( \nu_i = E[(\partial y / \partial p_i)^2] \)
- Efficient when local sensitivities (FSE or adjoint) are available.
- Can provide bounds for total Sobol indices.
- Useful when traditional variance-based GSA is too expensive.

---

## 8. Sensitivity of Steady States and Stability

### 8.1 Steady-State Sensitivity

- Steady-state \(x^\*\) satisfies \( f(x^\*, p) = 0 \).
- Sensitivity:  
  \( \partial x^\*/\partial p_i = -J^{-1} (\partial f/\partial p_i)|_{x^\*} \)
- Key in metabolic and signaling networks.

### 8.2 Bifurcation Sensitivity

- Quantifies parameter effects on bifurcation points (Hopf, saddle-node, pitchfork).
- Important in oscillatory or switch-like systems.
- Tools include AUTO and MATCONT.

---

## 9. Sensitivity in Parameter Estimation and Identifiability

### 9.1 Structural Identifiability

- Determines if parameters can be uniquely inferred from noise-free data.
- Related to rank of the sensitivity matrix.

### 9.2 Practical Identifiability

- Focuses on uncertainty with real data.
- Poor sensitivities → wide confidence intervals.
- Methods:
  - Fisher Information Matrix (FIM)  
  - Profile likelihood  
  - Bayesian posteriors

---

## 10. Role in Experimental Design

- SA helps guide **optimal experiment design**, for example:
  - Choosing informative sampling time points  
  - Selecting perturbations that maximize parameter identifiability  
  - Minimizing uncertainty
- Metrics:
  - **D-optimality** – maximize determinant of FIM  
  - **Robust design** to account for uncertain priors

---

## 11. Best Practices

- Define specific goals and outputs before SA.  
- Combine **local** and **global** approaches.  
- Use normalized sensitivities for comparability.  
- Apply GSA when:
  - Parameters are uncertain  
  - Dynamics are nonlinear  
  - Interactions are expected  
- Check numerical accuracy of ODE solver.  
- Document all choices: sampling strategy, priors, solver tolerances, metrics.

---

## 12. Summary

- Sensitivity analysis is central to understanding ODE models in systems biology.  
- **Local methods** (FSE, adjoint, finite differences) give derivative-based information near nominal parameters.  
- **Global methods** (Sobol, FAST, Morris) explore full parameter uncertainty and interactions.  
- Links to **identifiability**, **parameter estimation**, **bifurcation analysis**, and **experimental design**.  
- Combining multiple methods gives the most reliable insight.


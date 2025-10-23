# Statistical Learning: Regression and Classification Analysis

## Overview

This project explores fundamental concepts in statistical learning including ridge regression bias-variance tradeoff, multiple linear regression analysis, classification algorithms, and cross-validation techniques. The work demonstrates theoretical derivations, practical implementations in R, and empirical analysis of academic salary data and binary classification problems. Key topics covered include ridge regression estimators, bootstrap methods, logistic regression, quadratic discriminant analysis, k-nearest neighbors, and leave-one-out cross-validation formulations.

## Mathematical Formulation

The project centers on several core mathematical problems:

### Ridge Regression Analysis

Consider the regression model:

$$
Y = f(\mathbf{x}) + \varepsilon, \quad \text{where } \mathbb{E}[\varepsilon] = 0 \text{ and } \text{Var}(\varepsilon) = \sigma^2
$$

With true function $f(\mathbf{x}) = \mathbf{x}^T\boldsymbol{\beta}$, we compare the OLS estimator:

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}
$$

Against the ridge regression estimator:

$$
\widetilde{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{Y}
$$

The expected value and covariance matrix of the ridge estimator are:

$$
\mathbb{E}[\widetilde{\boldsymbol{\beta}}] = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\boldsymbol{\beta}
$$

$$
\text{Cov}[\widetilde{\boldsymbol{\beta}}] = \sigma^2(\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{X}((\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1})^T
$$

### Bias-Variance Decomposition

For prediction at new point $\mathbf{x}_0$, the expected MSE decomposes as:

$$
\mathbb{E}[(y_0 - \widetilde{f}(\mathbf{x}_0))^2] = \sigma^2 + \text{Var}[\widetilde{f}(\mathbf{x}_0)] + (\mathbb{E}[\widetilde{f}(\mathbf{x}_0)] - f(\mathbf{x}_0))^2
$$

Where the bias term is:

$$
\text{Bias}^2 = (x_0^T\boldsymbol{\beta} - x_0^T(\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{X}\boldsymbol{\beta})^2
$$

### Leave-One-Out Cross-Validation

For linear models, LOOCV can be computed efficiently using:

$$
\text{CV} = \frac{1}{N} \sum_{i=1}^N \left(\frac{y_i - \hat{y}_i}{1 - h_i}\right)^2
$$

Where $h_i = \mathbf{x}_i^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}_i$ are the leverage values.

### Classification Models

For binary classification, logistic regression uses:

$$
P(Y = 1 | \mathbf{X} = \mathbf{x}) = \frac{e^{\mathbf{x}^T\boldsymbol{\beta}}}{1 + e^{\mathbf{x}^T\boldsymbol{\beta}}}
$$

QDA assumes class-conditional Gaussians with different covariance matrices, while KNN uses local majority voting with Euclidean distance.

## Method and Algorithms

The project implements several key statistical learning methods:

**Ridge Regression**: Analytical derivation of bias and variance formulas, followed by numerical computation across a grid of regularization parameters $\lambda \in [0, 2]$. The bias-variance tradeoff is visualized by plotting these components against $\lambda$.

**Bootstrap Sampling**: Implementation of 1000 bootstrap samples to estimate the uncertainty in $R^2$ values, providing empirical distribution and confidence intervals for model performance metrics.

**Multiple Linear Regression**: Analysis of academic salary data using OLS with categorical variables (rank, discipline, sex), including interaction terms and model diagnostics. Log transformation applied to address heteroscedasticity and normality violations.

**Classification Algorithms**: Comparison of logistic regression, QDA, and KNN ($k=25$) on binary classification data. Models evaluated using confusion matrices, ROC curves, and AUC metrics on a 70-30 train-test split.

**Cross-Validation**: Theoretical derivation of the LOOCV formula for linear models using matrix algebra and the Sherman-Morrison formula, avoiding the computational cost of fitting $N$ separate models.

## Repository Structure

```
├── statistical_learning_analysis.Rmd    # Main R analysis file
├── statistical_learning_analysis.html   # Compiled HTML output
├── README.md                             # Project documentation
└── requirements.txt                      # R package dependencies
```

## Quick Start

### Environment Setup

This project requires R (≥4.0.0) and the following packages:

```bash
# Install R packages
Rscript -e "install.packages(c('knitr', 'rmarkdown', 'ggplot2', 'dplyr', 'tidyr', 'carData', 'class', 'pROC', 'plotROC', 'ggmosaic', 'ISLR', 'MASS', 'ggfortify', 'GGally'))"
```

### Minimal Run Command

```bash
# Render the R Markdown document
Rscript -e "rmarkdown::render('statistical_learning_analysis.Rmd')"
```

## Reproducing Results

To reproduce all results from the analysis:

1. **Install Dependencies**: Use the command above to install required R packages
2. **Set Random Seeds**: The analysis uses `set.seed(4268)` for bootstrap sampling and `set.seed(2023)` for data splitting to ensure reproducibility
3. **Data Sources**:
   - Academic salary analysis uses `Salaries` dataset from `carData` package
   - Binary classification uses data from TidyTuesday repository
4. **Execute Analysis**: Run the complete R Markdown document:

```bash
# Full analysis pipeline
Rscript -e "
set.seed(4268)
rmarkdown::render('statistical_learning_analysis.Rmd', output_format = 'all')
"
```

The analysis will generate bias-variance tradeoff plots, bootstrap distributions, classification performance metrics, and theoretical derivations.

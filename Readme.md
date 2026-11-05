# R Machine Learning Pipeline

A lightweight, automated R utility for data preprocessing, classification, regression modeling, probability distributions, clustering analysis, and exploratory data visualization.

---

## 1. Prerequisites & Setup

### Install Required Dependencies

Run the following command in R to install all required packages:

```R
install.packages(c(
  "caret", 
  "class", 
  "nnet", 
  "rpart", 
  "e1071", 
  "randomForest", 
  "glmnet",
  "corrplot",
  "cluster",
  "psych"
))
```

### Load Libraries & Source Script

Load the libraries and fetch the pipeline code directly from GitHub:

```R
library(caret)
library(class)
library(nnet)
library(rpart)
library(e1071)
library(randomForest)
library(glmnet)
library(corrplot)
library(cluster)
library(psych)

# Source functions directly from GitHub
source("https://raw.githubusercontent.com/R-Offficial/R-Data/main/Prepare.R")
```

---

## 2. Core Functions & Usage

### 2.1 `prepare_data()`

Performs automated data cleaning:
- Strips whitespace and turns blank strings into `NA`.
- Removes duplicate rows.
- Removes observations with missing target values.
- Splits data into training and test sets.
- Imputes missing numerical values (mean/median) and categorical values (mode) using training set statistics.
- One-hot encodes categorical predictors and aligns train/test columns.
- Scales features using standard (`z-score`) or `minmax` normalization.

```R
result <- prepare_data(
  data, 
  target     = names(data)[ncol(data)], # Target column name (defaults to last column)
  scale_type = "standard",              # Scaling: "standard" or "minmax"
  num_method = "mean",                  # Imputation: "mean" or "median"
  p          = 0.8,                     # Training split proportion (0 to 1)
  seed       = 123                      # Random seed for reproducibility
)
```

#### Extracting Prepared Outputs

Extract the partitioned and scaled matrices and labels from the returned list:

```R
scale_train_x <- result$scale_train_x
scale_test_x  <- result$scale_test_x

train_y <- result$train_y
test_y  <- result$test_y
```

Additional returned metadata:
- `result$imputation_info`: List of values used for numerical/categorical imputations.
- `result$duplicate_count`: Number of duplicate rows removed.

---

### 2.2 `classification_model()`

Trains a classifier on the prepared matrices, evaluates test performance with comprehensive metrics, and automatically renders a 4-panel diagnostic dashboard (`plot = TRUE`).

```R
clf <- classification_model(
  scale_train_x = scale_train_x,
  scale_test_x  = scale_test_x,
  train_y       = train_y,
  test_y        = test_y,
  model         = "decision_tree", # "knn", "logistic", "decision_tree", "naive_bayes", "svm", "random_forest"
  k             = 5,               # Number of neighbors (used when model = "knn")
  plot          = TRUE             # Automatically display 4-panel diagnostic plots
)
```

#### Supported Classification Models:
- `"knn"`: K-Nearest Neighbors (`class::knn`) with winning vote confidence analysis.
- `"logistic"`: Multinomial Logistic Regression (`nnet::multinom`) with probability curves.
- `"decision_tree"`: Decision Tree Classification (`rpart::rpart`) with tree visualization.
- `"naive_bayes"`: Naive Bayes (`e1071::naiveBayes`) with class-conditional distributions.
- `"svm"`: Support Vector Machine with Radial Kernel (`e1071::svm`) with decision boundaries and support vectors.
- `"random_forest"`: Random Forest (`randomForest::randomForest`) with variable importance and tree diagnostics.

#### Diagnostic Visualizations Generated (`plot = TRUE`):
1. **Actual vs Predicted Curve**: Sequential test observations sorted by true class, highlighting matches and marking misclassifications with prominent red indicators.
2. **Confusion Matrix Heatmap**: Full color-graded tile heatmap displaying cell frequencies and exact percentages.
3. **Probability Curve / Class Performance**: S-curve of predicted probabilities (binary) or grouped bar chart of Precision, Recall, and F1-score per class.
4. **Model-Specific Visualization**:
   - `decision_tree`: Draws the full tree diagram with splits, questions, and leaf classifications.
   - `random_forest`: Top feature importance barplot (Mean Decrease Gini).
   - `svm`: Decision boundary slice plot with highlighted support vectors.
   - `logistic`: Top coefficient weights / odds ratios barplot.
   - `naive_bayes`: Class prior / distribution barplot.
   - `knn`: Vote confidence histogram.

**Returns:**
- `clf$model`: The trained model object.
- `clf$predictions`: Factor vector of test set predictions.
- `clf$probabilities`: Matrix or vector of predicted class probabilities.
- `clf$confusion_matrix`: Confusion matrix table.
- `clf$class_metrics`: Precision, Recall, Specificity, F1-Score, and Support per class.
- `clf$overall_metrics`: Accuracy, Balanced Accuracy, Cohen's Kappa, Macro averages, Weighted averages, and ROC AUC (for binary).
- `clf$accuracy`: Overall accuracy score.
- `clf$balanced_accuracy`: Balanced accuracy score.
- `clf$kappa`: Cohen's Kappa score.
- `clf$macro_f1`: Macro-averaged F1 score.
- `clf$tree_table`: Tree #1 rule table (for Random Forest).
- `clf$support_vectors`: Total count and per-class breakdown of Support Vectors (for SVM).

---

### 2.3 `regression_model()`

Trains a regression model, calculates standard and advanced performance metrics ($MAE$, $MSE$, $RMSE$, $R^2$, Adjusted $R^2$, $MAPE$, Pearson $r$, $NRMSE$, and Explained Variance), and generates a 4-panel diagnostic dashboard (`plot = TRUE`).

```R
reg <- regression_model(
  scale_train_x = scale_train_x,
  scale_test_x  = scale_test_x,
  train_y       = train_y,
  test_y        = test_y,
  model         = "decision_tree", # "linear", "multiple_linear", "polynomial", "lasso", "ridge", "elastic_net", "decision_tree", "random_forest", "svm"
  degree        = 2,               # Degree for "polynomial"
  alpha         = 0.5,             # Mixing parameter for "elastic_net" (0 = ridge, 1 = lasso)
  k             = 5,               # Cross-validation folds for regularized models
  plot          = TRUE             # Automatically display 4-panel diagnostic plots
)
```

#### Supported Regression Models:
- `"linear"`: Simple Linear Regression (single predictor).
- `"multiple_linear"`: Multiple Linear Regression (`stats::lm`).
- `"polynomial"`: Polynomial Regression without interactions.
- `"lasso"`: L1 Regularization (`glmnet::cv.glmnet` with $\alpha = 1$).
- `"ridge"`: L2 Regularization (`glmnet::cv.glmnet` with $\alpha = 0$).
- `"elastic_net"`: Elastic Net Regularization (`glmnet::cv.glmnet` with user-defined $\alpha$).
- `"decision_tree"`: Regression Tree (`rpart::rpart`) with tree visualization.
- `"random_forest"`: Random Forest Regressor (`randomForest::randomForest`) with feature importance.
- `"svm"`: Support Vector Regressor (`e1071::svm`) with support vectors and error margin tube.

#### Diagnostic Visualizations Generated (`plot = TRUE`):
1. **Predict and Actual Curve**: Test observations sorted by actual value, showing the smooth blue Actual curve, dashed red Predicted curve, and connecting dotted residual lines.
2. **Actual vs Predicted Scatter Plot**: Ground truth vs predictions with ideal $y = x$ reference line and linear trend line.
3. **Residuals vs Predicted Plot**: Prediction residuals against predicted values with horizontal zero-error line and error bounds.
4. **Model-Specific Visualization**:
   - `decision_tree`: Draws the full regression tree diagram with split criteria and node values.
   - `random_forest`: Top feature importance barplot (%IncMSE or IncNodePurity).
   - `svm`: Residuals plot highlighting Support Vectors and $\pm \epsilon$ margin band.
   - `lasso`, `ridge`, `elastic_net`: Cross-validation MSE curve vs $\log(\lambda)$.
   - `linear`, `multiple_linear`, `polynomial`: Residuals Normal Q-Q plot.

**Returns:**
- `reg$model`: The trained model object.
- `reg$predictions`: Predicted continuous values.
- `reg$actual`: Ground truth test values.
- `reg$residuals`: Errors ($Actual - Predicted$).
- `reg$prediction_table`: `data.frame` of actuals, predictions, residuals, and percentage errors.
- `reg$metrics`: Summary table of MAE, MSE, RMSE, $R^2$, Adjusted $R^2$, MAPE, Pearson $r$, NRMSE, Explained Variance, Max Error, AIC, and BIC.
- `reg$MAE`, `reg$MSE`, `reg$RMSE`, `reg$R2`, `reg$adjusted_R2`, `reg$MAPE`, `reg$pearson_r`, `reg$nrmse`: Direct metric values.
- `reg$tree_table`: Tree #1 rule table (for Random Forest).
- `reg$support_vectors`: Total count and indices of Support Vectors (for SVM).

---

### 2.4 Standalone Machine Learning Visualization Functions

You can also call the dedicated plotting helpers directly at any time:

```R
# 1. Draw Predict vs Actual Curve (Classification or Regression)
draw_predict_actual(actual = test_y, predicted = predictions, model_name = "Model")

# 2. Draw Decision Tree
draw_decision_tree(model = clf$model, title = "Classification Tree")

# 3. Draw Random Forest Diagnostics (Error vs Trees & Feature Importance)
draw_random_forest(model = clf$model, title = "Random Forest", standalone = TRUE)

# 4. Draw SVM Decision Boundary & Support Vectors
draw_svm(model = clf$model, data = train_df, title = "SVM Decision Boundary")

# 5. Draw Confusion Matrix Heatmap
draw_confusion_matrix(cm = clf$confusion_matrix, title = "Confusion Matrix Heatmap")
```

---

### 2.5 `probability_distribution()`

A unified function for computing and visualizing probability distributions — Binomial, Poisson, Normal, and Geometric.

```R
probability_distribution(
  dist           = "binomial",   # Distribution: "binomial", "poisson", "normal", "geometric"
  action         = "pmf",        # Action: "pmf", "cdf", "prob", "quantile", "fit", "sample", "compare", "joint", "density"
  size           = NULL,         # Binomial: number of trials
  prob           = NULL,         # Success probability (Binomial / Geometric)
  lambda         = NULL,         # Poisson rate parameter
  mean           = 0,            # Normal mean
  sd             = 1,            # Normal standard deviation
  x              = NULL,         # Values for PMF/CDF, or query values for prob
  prob_type      = "exact",      # Probability type: "exact", "leq", "geq", "between"
  quantile_p     = NULL,         # Probability for quantile lookup (e.g. 0.90)
  observed       = NULL,         # Observed frequencies for fitting
  n_sample       = 1000,         # Sample size for random generation
  seed           = 123,          # Random seed
  compare_dist   = NULL,         # Second distribution for comparison
  compare_params = NULL,         # Named list of params for second distribution
  joint_matrix   = NULL,         # Numeric matrix for joint distribution
  plot           = TRUE          # Produce plots?
)
```

#### Supported Actions:

| Action | Description |
|--------|-------------|
| `"pmf"` | PMF (discrete) or PDF (continuous) values with plot |
| `"cdf"` | Cumulative distribution function values with plot |
| `"prob"` | Probability queries — exact, $P(X \leq x)$, $P(X \geq x)$, $P(a \leq X \leq b)$ |
| `"quantile"` | Percentile / quantile lookup |
| `"fit"` | Fit distribution to observed frequencies, compute expected counts |
| `"sample"` | Generate random sample, compare empirical vs theoretical moments |
| `"compare"` | Side-by-side PMF comparison of two distributions |
| `"joint"` | Joint and marginal probability table from a matrix |
| `"density"` | Normal density curve plot |

**Returns:** A list containing computed values, tables, and parameters (varies by action).

---

### 2.6 `clustering_analysis()`

Performs end-to-end unsupervised clustering analysis: data exploration, correlation, PCA, K-Means, Hierarchical Clustering, and statistical tests — all in a single function call.

```R
result <- clustering_analysis(
  data          = df,            # Input data frame
  num_cols      = NULL,          # Numeric columns for clustering (NULL = auto-detect all)
  k             = 3,            # Number of clusters
  max_k         = 10,           # Maximum k for Elbow method
  hc_method     = "ward.D2",    # Hierarchical linkage method
  profile_cols  = NULL,          # Columns to profile by cluster (defaults to num_cols)
  cat_cols      = NULL,          # Categorical columns for chi-squared tests
  test_var      = NULL,          # Numeric variable for statistical tests
  test_clusters = c(1, 2),      # Which two clusters for pairwise tests
  cor_threshold = 0.7,          # Threshold for flagging high correlations
  seed          = 123,          # Random seed
  plot          = TRUE           # Produce plots?
)
```

#### What It Does (in order):

1. Prints dataset dimensions, structure, summary, and missing values
2. Removes duplicate rows
3. Scales numeric columns (z-score)
4. Plots histograms for each numeric column
5. Computes and plots correlation matrix; flags pairs above `cor_threshold`
6. Runs PCA — prints variance explained, plots scree plot and PC1 vs PC2
7. Runs Elbow method for k = 1 to `max_k`
8. K-Means clustering with centroid reporting
9. Silhouette analysis with plot
10. Hierarchical clustering with dendrogram
11. Cross-tabulation of K-Means vs Hierarchical assignments
12. Cluster profiling (mean of each feature per cluster)
13. Statistical tests on `test_var`: confidence intervals, Welch t-test, Wilcoxon, ANOVA, Tukey HSD
14. Chi-squared tests for each column in `cat_cols`

**Returns:**
- `result$scaled_data`: Scaled numeric matrix.
- `result$cor_matrix`: Correlation matrix.
- `result$high_corr`: Pairs exceeding the correlation threshold.
- `result$pca`: `prcomp` object.
- `result$variance_explained`: Proportion of variance per PC.
- `result$wcss`: Within-cluster SS vector (for Elbow plot).
- `result$kmeans`: K-Means result object.
- `result$km_centroids`: Centroid means per cluster.
- `result$km_silhouette`: Mean silhouette width.
- `result$hclust`: Hierarchical clustering object.
- `result$hc_clusters`: Hierarchical cluster assignments.
- `result$cross_table`: K-Means vs Hierarchical cross-tabulation.
- `result$cluster_profile`: Aggregate means by cluster.
- `result$ttest`: Welch t-test result.
- `result$wilcox`: Wilcoxon rank-sum test result.
- `result$anova`: ANOVA result.
- `result$tukey`: Tukey HSD result.
- `result$chisq_tests`: Named list of chi-squared test results.
- `result$data`: Original data with `km_cluster` and `hc_cluster` columns added.

---

### 2.7 `generate_plot()`

A unified plotting function supporting 7 key chart types: histograms, frequency polygons, bar charts, box plots, pie charts, line plots, and scatter plots.

```R
generate_plot(
  data,
  type        = c("hist", "freq_poly", "bar", "box", "pie", "line", "scatter"),
  x           = NULL,        # Primary variable column name
  y           = NULL,        # Secondary variable column name (optional / dependent on type)
  group       = NULL,        # Categorical variable for coloring / symbols
  breaks      = NULL,        # Custom bin breaks (hist / freq_poly)
  horiz       = FALSE,       # Horizontal bars for type = "bar"
  show_labels = TRUE,        # Value / count labels on bars & bins
  add_reg     = TRUE,        # Linear regression trendline for scatter
  show_grid   = TRUE,        # Background grid
  col         = NULL,        # Custom color vector
  main        = NULL,        # Plot title
  xlab        = NULL,        # X-axis label
  ylab        = NULL,        # Y-axis label
  ...
)
```

#### Supported Plot Types:

| `type` | Description | Typical Arguments |
|---|---|---|
| `"hist"` | Histogram with bin labels | `x = "Marks", breaks = seq(50, 100, 10)` |
| `"freq_poly"` | Frequency polygon with points and lines | `x = "Marks", breaks = seq(50, 100, 10)` |
| `"bar"` | Vertical or horizontal count bar plot | `x = "Department"`, `horiz = TRUE` |
| `"box"` | Univariate or grouped box plot with IQR/SD | `x = "Marks"` or `x = "Department", y = "Marks"` |
| `"pie"` | Pie chart with slices, percentages, and labels | `x = "Department"` |
| `"line"` | Line plot with category axis | `x = "Marks", y = "Student_ID"` |
| `"scatter"` | Scatter plot with regression line & correlation | `x = "Study_Hours", y = "Marks", group = "Department"` |

---

## 3. End-to-End Workflows

### 3.1 Classification Example (`iris`)

```R
# 1. Source script and libraries
source("https://raw.githubusercontent.com/R-Offficial/R-Data/main/Prepare.R")

# 2. Run preprocessing
result <- prepare_data(
  data       = iris, 
  target     = "Species", 
  scale_type = "standard", 
  num_method = "median", 
  p          = 0.8,
  seed       = 123
)

# 3. Extract train/test sets
scale_train_x <- result$scale_train_x
scale_test_x  <- result$scale_test_x
train_y       <- result$train_y
test_y        <- result$test_y

# 4. Fit classifier
clf_res <- classification_model(
  scale_train_x = scale_train_x,
  scale_test_x  = scale_test_x,
  train_y       = train_y,
  test_y        = test_y,
  model         = "random_forest"
)

# View accuracy
print(clf_res$accuracy)
```

---

### 3.2 Regression Example (`mtcars`)

```R
# 1. Source script and libraries
source("https://raw.githubusercontent.com/R-Offficial/R-Data/main/Prepare.R")

# 2. Run preprocessing
result <- prepare_data(
  data       = mtcars, 
  target     = "mpg", 
  scale_type = "minmax", 
  num_method = "mean", 
  p          = 0.8,
  seed       = 123
)

# 3. Extract train/test sets
scale_train_x <- result$scale_train_x
scale_test_x  <- result$scale_test_x
train_y       <- result$train_y
test_y        <- result$test_y

# 4. Fit regression model
reg_res <- regression_model(
  scale_train_x = scale_train_x,
  scale_test_x  = scale_test_x,
  train_y       = train_y,
  test_y        = test_y,
  model         = "lasso",
  k             = 5
)

# View error metrics
print(reg_res$metrics)
```

---

### 3.3 Probability Distribution Examples

```R
source("https://raw.githubusercontent.com/R-Offficial/R-Data/main/Prepare.R")

# Binomial PMF and CDF
probability_distribution(dist = "binomial", action = "pmf", size = 12, prob = 0.25)
probability_distribution(dist = "binomial", action = "cdf", size = 12, prob = 0.25)

# Probability queries
# P(X = 4) for Binomial(12, 0.25)
probability_distribution(dist = "binomial", action = "prob", size = 12, prob = 0.25,
                          x = 4, prob_type = "exact")

# P(X <= 3) for Poisson(5)
probability_distribution(dist = "poisson", action = "prob", lambda = 5,
                          x = 3, prob_type = "leq")

# P(40 < X < 55) for Normal(50, 8)
probability_distribution(dist = "normal", action = "prob", mean = 50, sd = 8,
                          x = c(40, 55), prob_type = "between")

# 90th percentile of N(50, 8)
probability_distribution(dist = "normal", action = "quantile", mean = 50, sd = 8,
                          quantile_p = 0.90)

# Fit Binomial to observed frequencies
probability_distribution(dist = "binomial", action = "fit",
                          observed = c(4, 15, 30, 28, 17, 6))

# Fit Poisson to observed frequencies
probability_distribution(dist = "poisson", action = "fit",
                          observed = c(10, 25, 30, 20, 10, 4, 1))

# Random sampling from Normal(25, 5)
probability_distribution(dist = "normal", action = "sample",
                          mean = 25, sd = 5, n_sample = 500)

# Compare Binomial(10, 0.5) vs Poisson(5)
probability_distribution(dist = "binomial", action = "compare",
                          size = 10, prob = 0.5,
                          compare_dist = "poisson",
                          compare_params = list(lambda = 5))

# Joint probability table
mat <- matrix(c(10,15,5, 20,25,10, 5,10,15), nrow = 3, byrow = TRUE)
rownames(mat) <- c("X1","X2","X3"); colnames(mat) <- c("Y1","Y2","Y3")
probability_distribution(action = "joint", joint_matrix = mat)
```

---

### 3.4 Clustering Analysis Example

```R
source("https://raw.githubusercontent.com/R-Offficial/R-Data/main/Prepare.R")

df <- read.csv("online_shoppers_intention.csv")

result <- clustering_analysis(
  data          = df,
  num_cols      = c("Administrative", "Administrative_Duration",
                    "Informational", "Informational_Duration",
                    "ProductRelated", "ProductRelated_Duration",
                    "BounceRates", "ExitRates", "PageValues", "SpecialDay"),
  k             = 3,
  max_k         = 10,
  hc_method     = "ward.D2",
  test_var      = "PageValues",
  test_clusters = c(1, 2),
  cat_cols      = c("VisitorType", "Revenue"),
  cor_threshold = 0.7,
  seed          = 123
)

# Access results
result$km_centroids        # Cluster centroids
result$km_silhouette       # Silhouette score
result$pca                 # PCA object
result$variance_explained  # Variance per PC
result$ttest               # t-test result
result$anova               # ANOVA result
result$tukey               # Tukey HSD
result$chisq_tests         # Chi-squared tests
result$data                # Data with cluster labels
```

---

### 3.5 Data Visualization Examples (`generate_plot`)

```R
source("https://raw.githubusercontent.com/R-Offficial/R-Data/main/Prepare.R")

# 1. Histogram
generate_plot(df, type = "hist", x = "Marks", breaks = seq(50, 100, by = 10))

# 2. Frequency Polygon
generate_plot(df, type = "freq_poly", x = "Marks", breaks = seq(50, 100, by = 10))

# 3. Bar Plots (Vertical & Horizontal)
generate_plot(df, type = "bar", x = "Department")
generate_plot(df, type = "bar", x = "Department", horiz = TRUE)

# 4. Box Plots (Univariate & Grouped)
generate_plot(df, type = "box", x = "Marks")
generate_plot(df, type = "box", x = "Department", y = "Marks")

# 5. Pie Chart (with Categories & Percentages)
generate_plot(df, type = "pie", x = "Department")

# 6. Line Plot (mapped to index or custom ID)
generate_plot(df, type = "line", x = "Marks", y = "Student_ID")

# 7. Scatter Plot (Regression Line, Grouping & Correlation)
generate_plot(df, type = "scatter", x = "Study_Hours", y = "Marks", group = "Department")
```
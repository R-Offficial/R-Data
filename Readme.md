# R Machine Learning Pipeline

A lightweight R utility for automated data preprocessing, classification, and regression modeling.

## 1. Prerequisites & Setup

### Install Required Packages
Run the following command in R to install the required dependencies:

```R
install.packages(c(
  "caret", 
  "class", 
  "nnet", 
  "rpart", 
  "e1071", 
  "randomForest", 
  "glmnet"
))
```

### Load the Functions
Load the required libraries and source the pipeline script directly from GitHub:

```R
library(caret)
library(class)
library(nnet)
library(rpart)
library(e1071)
library(randomForest)
library(glmnet)

# Load functions directly from GitHub
source("https://raw.githubusercontent.com/R-Offficial/R-Data/main/Prepare.R")
```

---

## 2. Function Calls & Parameters

### `prepare_data()`
Cleans empty strings, removes duplicate rows, imputes missing values, one-hot encodes categorical variables, splits data into train/test sets, and scales features.

```R
prep <- prepare_data(
  data, 
  target = names(data)[ncol(data)], # Target column name (defaults to last column)
  scale_type = "standard",          # Options: "standard" or "minmax"
  num_method = "mean",              # Imputation: "mean" or "median" (categorical uses mode)
  p = 0.8,                          # Train split ratio (0 to 1)
  seed = 123                        # Random seed for reproducibility
)
```
**Returns:** `scale_train_x`, `scale_test_x`, `train_y`, `test_y`, `imputation_info`, `duplicate_count`

---

### `classification_model()`
Trains a classification model and outputs the confusion matrix, class-wise metrics, and overall accuracy/F1 scores.

```R
clf <- classification_model(
  scale_train_x = prep$scale_train_x,
  scale_test_x  = prep$scale_test_x,
  train_y       = prep$train_y,
  test_y        = prep$test_y,
  model         = "knn",            # Options: "knn", "logistic", "decision_tree", "naive_bayes", "svm", "random_forest"
  k             = 5                 # Number of neighbors (used only when model = "knn")
)
```
**Returns:** `model`, `predictions`, `confusion_matrix`, `class_metrics`, `overall_metrics`, `accuracy`

---

### `regression_model()`
Trains a regression model and computes MAE, MSE, RMSE, $R^2$, Adjusted $R^2$, and MAPE.

```R
reg <- regression_model(
  scale_train_x = prep$scale_train_x,
  scale_test_x  = prep$scale_test_x,
  train_y       = prep$train_y,
  test_y        = prep$test_y,
  model         = "multiple_linear", # Options: "linear", "multiple_linear", "polynomial", "lasso", "ridge", "elastic_net", "decision_tree", "random_forest", "svm"
  degree        = 2,                 # Polynomial degree (for "polynomial")
  alpha         = 0.5,               # Elastic net mixing parameter 0-1 (for "elastic_net")
  k             = 5                  # Cross-validation folds (for "lasso", "ridge", "elastic_net")
)
```
**Returns:** `model`, `predictions`, `actual`, `residuals`, `prediction_table`, `metrics`, `MAE`, `MSE`, `RMSE`, `R2`, `adjusted_R2`, `MAPE`

---

## 3. Quick-Start Examples

### Classification Workflow (`iris` dataset)

```R
# 1. Load script
source("https://raw.githubusercontent.com/R-Offficial/R-Data/main/Prepare.R")

# 2. Prepare data
prep_clf <- prepare_data(
  data = iris, 
  target = "Species", 
  scale_type = "standard", 
  num_method = "median", 
  p = 0.8
)

# 3. Run Random Forest Classifier
res_clf <- classification_model(
  scale_train_x = prep_clf$scale_train_x,
  scale_test_x  = prep_clf$scale_test_x,
  train_y       = prep_clf$train_y,
  test_y        = prep_clf$test_y,
  model         = "random_forest"
)
```

### Regression Workflow (`mtcars` dataset)

```R
# 1. Load script
source("https://raw.githubusercontent.com/R-Offficial/R-Data/main/Prepare.R")

# 2. Prepare data
prep_reg <- prepare_data(
  data = mtcars, 
  target = "mpg", 
  scale_type = "minmax", 
  num_method = "mean", 
  p = 0.8
)

# 3. Run Lasso Regression
res_reg <- regression_model(
  scale_train_x = prep_reg$scale_train_x,
  scale_test_x  = prep_reg$scale_test_x,
  train_y       = prep_reg$train_y,
  test_y        = prep_reg$test_y,
  model         = "lasso",
  k             = 5
)
```
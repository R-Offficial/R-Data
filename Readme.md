# R Machine Learning Pipeline

A lightweight, automated R utility for data preprocessing, classification, and regression modeling.

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
  "glmnet"
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

Trains a classifier on the prepared matrices and evaluates test performance.

```R
clf <- classification_model(
  scale_train_x = scale_train_x,
  scale_test_x  = scale_test_x,
  train_y       = train_y,
  test_y        = test_y,
  model         = "knn",  # Options: "knn", "logistic", "decision_tree", "naive_bayes", "svm", "random_forest"
  k             = 5       # Number of neighbors (used when model = "knn")
)
```

#### Supported Classification Models:
- `"knn"`: K-Nearest Neighbors (`class::knn`)
- `"logistic"`: Multinomial Logistic Regression (`nnet::multinom`)
- `"decision_tree"`: Decision Tree Classification (`rpart::rpart`)
- `"naive_bayes"`: Naive Bayes (`e1071::naiveBayes`)
- `"svm"`: Support Vector Machine with Radial Kernel (`e1071::svm`)
- `"random_forest"`: Random Forest (`randomForest::randomForest`)

**Returns:**
- `clf$model`: The trained model object.
- `clf$predictions`: Factor vector of test set predictions.
- `clf$confusion_matrix`: Confusion matrix table.
- `clf$class_metrics`: Precision, Recall, and F1-Score per class.
- `clf$overall_metrics`: Accuracy, Macro averages, and Weighted averages.
- `clf$accuracy`: Overall accuracy score.

---

### 2.3 `regression_model()`

Trains a regression model and calculates standard performance metrics ($MAE$, $MSE$, $RMSE$, $R^2$, Adjusted $R^2$, and $MAPE$).

```R
reg <- regression_model(
  scale_train_x = scale_train_x,
  scale_test_x  = scale_test_x,
  train_y       = train_y,
  test_y        = test_y,
  model         = "multiple_linear", # Options: "linear", "multiple_linear", "polynomial", "lasso", "ridge", "elastic_net", "decision_tree", "random_forest", "svm"
  degree        = 2,                 # Degree for "polynomial"
  alpha         = 0.5,               # Mixing parameter for "elastic_net" (0 = ridge, 1 = lasso)
  k             = 5                  # Cross-validation folds for regularized models
)
```

#### Supported Regression Models:
- `"linear"`: Simple Linear Regression (single predictor).
- `"multiple_linear"`: Multiple Linear Regression (`stats::lm`).
- `"polynomial"`: Polynomial Regression without interactions.
- `"lasso"`: L1 Regularization (`glmnet::cv.glmnet` with $\alpha = 1$).
- `"ridge"`: L2 Regularization (`glmnet::cv.glmnet` with $\alpha = 0$).
- `"elastic_net"`: Elastic Net Regularization (`glmnet::cv.glmnet` with user-defined $\alpha$).
- `"decision_tree"`: Regression Tree (`rpart::rpart`).
- `"random_forest"`: Random Forest Regressor (`randomForest::randomForest`).
- `"svm"`: Support Vector Regressor (`e1071::svm`).

**Returns:**
- `reg$model`: The trained model object.
- `reg$predictions`: Predicted continuous values.
- `reg$actual`: Ground truth values.
- `reg$residuals`: Errors ($Actual - Predicted$).
- `reg$prediction_table`: `data.frame` of actuals, predictions, and residuals.
- `reg$metrics`: Summary table of MAE, MSE, RMSE, $R^2$, Adjusted $R^2$, and MAPE.

---

## 3. End-to-End Workflows

### 3.1 Classification Example (`iris`)

```R
# 1. Source script and libraries
library(caret); library(class); library(nnet); library(rpart); library(e1071); library(randomForest); library(glmnet)
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
library(caret); library(class); library(nnet); library(rpart); library(e1071); library(randomForest); library(glmnet)
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
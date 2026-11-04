#' @importFrom caret createDataPartition
#' @importFrom stats median model.matrix sd setNames
#' @export
#' 
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

prepare_data <- function(data,
                         target = names(data)[ncol(data)],
                         scale_type = "standard",
                         num_method = "mean",
                         p = 0.8,
                         seed = 123) {

  # ==========================================
  # 1. VALIDATION AND INITIAL CHECKS
  # ==========================================

  if (!is.data.frame(data) || nrow(data) == 0) {
    stop("Provide a non-empty data frame.")
  }

  if (anyDuplicated(names(data))) {
    stop("Duplicate column names found.")
  }

  if (!target %in% names(data)) {
    stop("Target column not found.")
  }

  scale_type <- tolower(scale_type)
  num_method <- tolower(num_method)

  if (!scale_type %in% c("standard", "minmax")) {
    stop("Use standard or minmax scaling.")
  }

  if (!num_method %in% c("mean", "median")) {
    stop("Use mean or median imputation.")
  }

  if (p <= 0 || p >= 1) {
    stop("p must be between 0 and 1.")
  }

  cat("\n========== DATASET INFORMATION ==========\n")

  cat("Original dimensions:", dim(data), "\n")
  cat("Target variable:", target, "\n")

  # Work on a copy
  data <- as.data.frame(data)

  # ==========================================
  # 2. CLEAN EMPTY STRINGS
  # ==========================================

  for (col in names(data)) {

    if (is.character(data[[col]]) ||
        is.factor(data[[col]])) {

      x <- trimws(as.character(data[[col]]))

      x[x == ""] <- NA

      data[[col]] <- x
    }
  }

  # ==========================================
  # 3. CHECK AND REMOVE DUPLICATES
  # ==========================================

  duplicate_count <- sum(duplicated(data))

  cat("\nDuplicate rows found:", duplicate_count, "\n")

  data <- data[!duplicated(data), , drop = FALSE]

  cat("Dimensions after duplicate removal:",
      dim(data), "\n")

  # ==========================================
  # 4. HANDLE MISSING TARGET VALUES
  # ==========================================

  missing_target <- sum(is.na(data[[target]]))

  cat("\nMissing target values:", missing_target, "\n")

  if (missing_target > 0) {

    cat("Removing rows with missing target values.\n")

    data <- data[!is.na(data[[target]]), , drop = FALSE]
  }

  if (nrow(data) < 2) {
    stop("Not enough observations after cleaning.")
  }

  # Classification target
  if (is.character(data[[target]])) {
    data[[target]] <- factor(data[[target]])
  }

  if (is.factor(data[[target]]) &&
      nlevels(droplevels(data[[target]])) < 2) {
    stop("Target must contain at least two classes.")
  }

  # ==========================================
  # 5. SEPARATE X AND Y
  # ==========================================

  X <- data[, setdiff(names(data), target),
            drop = FALSE]

  Y <- data[[target]]

  if (ncol(X) == 0) {
    stop("No predictor columns available.")
  }

  # Remove unused factor levels
  for (col in names(X)) {
    if (is.factor(X[[col]])) {
      X[[col]] <- droplevels(X[[col]])
    }
  }

  # ==========================================
  # 6. TRAIN-TEST SPLIT
  # ==========================================

  set.seed(seed)

  train_index <- caret::createDataPartition(
    Y,
    p = p,
    list = FALSE
  )

  X_train <- X[train_index, , drop = FALSE]
  X_test  <- X[-train_index, , drop = FALSE]

  train_y <- Y[train_index]
  test_y  <- Y[-train_index]

  cat("\n========== TRAIN TEST SPLIT ==========\n")

  cat("Training rows:", nrow(X_train), "\n")
  cat("Testing rows:", nrow(X_test), "\n")

  # ==========================================
  # 7. MISSING VALUE IMPUTATION
  # ==========================================

  imputation_info <- list()

  mode_value <- function(x) {

    x <- x[!is.na(x)]

    if (length(x) == 0) {
      stop("A categorical column contains only missing values.")
    }

    values <- unique(x)

    values[which.max(tabulate(match(x, values)))]
  }

  for (col in names(X_train)) {

    if (is.numeric(X_train[[col]])) {

      # Numerical: mean or median
      fill_value <- if (num_method == "mean") {
        mean(X_train[[col]], na.rm = TRUE)
      } else {
        median(X_train[[col]], na.rm = TRUE)
      }

      if (!is.finite(fill_value)) {
        stop(paste("Invalid imputation value in", col))
      }

      X_train[[col]][is.na(X_train[[col]])] <- fill_value
      X_test[[col]][is.na(X_test[[col]])] <- fill_value

      imputation_info[[col]] <- fill_value

    } else {

      # Categorical: mode
      fill_value <- mode_value(X_train[[col]])

      X_train[[col]][is.na(X_train[[col]])] <- fill_value
      X_test[[col]][is.na(X_test[[col]])] <- fill_value

      imputation_info[[col]] <- fill_value
    }
  }

  cat("\n========== MISSING VALUE IMPUTATION ==========\n")
  print(imputation_info)

  # ==========================================
  # 8. CATEGORICAL ENCODING
  # ==========================================

  # Convert categorical columns into factors
  for (col in names(X_train)) {

    if (!is.numeric(X_train[[col]])) {

      all_levels <- unique(c(
        as.character(X_train[[col]]),
        as.character(X_test[[col]])
      ))

      X_train[[col]] <- factor(
        X_train[[col]],
        levels = all_levels
      )

      X_test[[col]] <- factor(
        X_test[[col]],
        levels = all_levels
      )
    }
  }

  # One-hot encoding
  scale_train_x <- model.matrix(
    ~ . - 1,
    data = X_train
  )

  scale_test_x <- model.matrix(
    ~ . - 1,
    data = X_test
  )

  # Align test columns with training columns
  missing_columns <- setdiff(
    colnames(scale_train_x),
    colnames(scale_test_x)
  )

  for (col in missing_columns) {
    scale_test_x <- cbind(
      scale_test_x,
      setNames(
        data.frame(rep(0, nrow(scale_test_x))),
        col
      )
    )
  }

  scale_test_x <- scale_test_x[
    , colnames(scale_train_x), drop = FALSE
  ]

  # ==========================================
  # 9. FEATURE SCALING
  # ==========================================

  if (scale_type == "standard") {

    center <- colMeans(scale_train_x)

    spread <- apply(scale_train_x, 2, sd)

    spread[spread == 0 | is.na(spread)] <- 1

    scale_train_x <- scale(
      scale_train_x,
      center = center,
      scale = spread
    )

    scale_test_x <- scale(
      scale_test_x,
      center = center,
      scale = spread
    )

  } else {

    min_values <- apply(scale_train_x, 2, min)

    max_values <- apply(scale_train_x, 2, max)

    ranges <- max_values - min_values

    ranges[ranges == 0] <- 1

    scale_train_x <- sweep(
      sweep(scale_train_x, 2, min_values, "-"),
      2, ranges, "/"
    )

    scale_test_x <- sweep(
      sweep(scale_test_x, 2, min_values, "-"),
      2, ranges, "/"
    )
  }

  # ==========================================
  # 10. FINAL REPORT
  # ==========================================

  cat("\n========== FINAL REPORT ==========\n")

  cat("Training feature dimensions:",
      dim(scale_train_x), "\n")

  cat("Testing feature dimensions:",
      dim(scale_test_x), "\n")

  cat("Remaining training missing values:",
      sum(is.na(scale_train_x)), "\n")

  cat("Remaining testing missing values:",
      sum(is.na(scale_test_x)), "\n")

  # Return everything
  return(list(
    scale_train_x = scale_train_x,
    scale_test_x = scale_test_x,
    train_y = train_y,
    test_y = test_y,
    imputation_info = imputation_info,
    duplicate_count = duplicate_count
  ))
}



classification_model <- function(scale_train_x,
                                 scale_test_x,
                                 train_y,
                                 test_y,
                                 model = "knn",
                                 k = 5) {

  # ==========================================
  # 1. VALIDATE INPUTS
  # ==========================================

  model <- tolower(model)

  valid_models <- c(
    "logistic",
    "knn",
    "decision_tree",
    "naive_bayes",
    "svm",
    "random_forest"
  )

  if (!model %in% valid_models) {
    stop(paste(
      "Choose model from:",
      paste(valid_models, collapse = ", ")
    ))
  }

  # Check dimensions
  if (nrow(scale_train_x) != length(train_y) ||
      nrow(scale_test_x) != length(test_y)) {
    stop("Feature rows and target lengths do not match.")
  }

  if (ncol(scale_train_x) != ncol(scale_test_x)) {
    stop("Training and testing features must have same columns.")
  }

  if (anyNA(scale_train_x) || anyNA(scale_test_x)) {
    stop("Missing values found in features.")
  }

  if (length(unique(train_y)) < 2) {
    stop("Training target must contain at least two classes.")
  }

  if (k < 1 || k > length(train_y)) {
    stop("Invalid K value.")
  }

  # Make target categorical
  classes <- if (is.factor(train_y)) {
    levels(train_y)
  } else {
    unique(as.character(train_y))
  }

  train_y <- factor(train_y, levels = classes)
  test_y <- factor(test_y, levels = classes)

  # Convert features to numeric matrices
  train_x <- as.matrix(scale_train_x)
  test_x <- as.matrix(scale_test_x)

  storage.mode(train_x) <- "double"
  storage.mode(test_x) <- "double"

  # Check infinite values
  if (any(!is.finite(train_x)) ||
      any(!is.finite(test_x))) {
    stop("Infinite or non-finite feature values found.")
  }

  # Convert to data frames for formula-based models
  train_df <- data.frame(
    train_x,
    check.names = FALSE
  )

  test_df <- data.frame(
    test_x,
    check.names = FALSE
  )

  train_df$target <- train_y

  # ==========================================
  # 2. TRAIN CLASSIFICATION MODEL
  # ==========================================

  cat("\n====================================\n")
  cat("MODEL:", toupper(model), "\n")
  cat("====================================\n")

  if (model == "knn") {

    # K-Nearest Neighbors

    fitted_model <- NULL

    predictions <- class::knn(
      train = train_x,
      test = test_x,
      cl = train_y,
      k = k
    )

  } else if (model == "logistic") {

    # Multinomial Logistic Regression

    fitted_model <- nnet::multinom(
      target ~ .,
      data = train_df,
      trace = FALSE
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df,
      type = "class"
    )

  } else if (model == "decision_tree") {

    # Decision Tree

    fitted_model <- rpart::rpart(
      target ~ .,
      data = train_df,
      method = "class"
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df,
      type = "class"
    )

  } else if (model == "naive_bayes") {

    # Naive Bayes

    fitted_model <- e1071::naiveBayes(
      target ~ .,
      data = train_df
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df,
      type = "class"
    )

  } else if (model == "svm") {

    # Support Vector Machine

    fitted_model <- e1071::svm(
      target ~ .,
      data = train_df,
      kernel = "radial",
      scale = FALSE
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df
    )

  } else if (model == "random_forest") {

    # Random Forest

    fitted_model <- randomForest::randomForest(
      target ~ .,
      data = train_df,
      ntree = 500,
      importance = TRUE
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df
    )
  }

  # ==========================================
  # 3. PREDICTION
  # ==========================================

  predictions <- factor(
    predictions,
    levels = classes
  )

  # ==========================================
  # 4. CONFUSION MATRIX
  # ==========================================

  cm <- table(
    Actual = test_y,
    Predicted = predictions
  )

  cat("\n========== CONFUSION MATRIX ==========\n")

  print(cm)

  # ==========================================
  # 5. CALCULATE METRICS
  # ==========================================

  TP <- diag(cm)

  actual_total <- rowSums(cm)
  predicted_total <- colSums(cm)

  precision <- ifelse(
    predicted_total > 0,
    TP / predicted_total,
    0
  )

  recall <- ifelse(
    actual_total > 0,
    TP / actual_total,
    0
  )

  f1 <- ifelse(
    precision + recall > 0,
    2 * precision * recall /
      (precision + recall),
    0
  )

  # Class-wise metrics
  class_metrics <- data.frame(
    Class = rownames(cm),
    Precision = as.numeric(precision),
    Recall = as.numeric(recall),
    F1_Score = as.numeric(f1),
    Support = as.numeric(actual_total)
  )

  # Overall accuracy
  accuracy <- sum(TP) / sum(cm)

  # Macro averages
  macro_precision <- mean(precision)
  macro_recall <- mean(recall)
  macro_f1 <- mean(f1)

  # Weighted averages
  weighted_precision <- sum(precision * actual_total) /
    sum(actual_total)

  weighted_recall <- sum(recall * actual_total) /
    sum(actual_total)

  weighted_f1 <- sum(f1 * actual_total) /
    sum(actual_total)

  overall_metrics <- data.frame(
    Metric = c(
      "Accuracy",
      "Macro Precision",
      "Macro Recall",
      "Macro F1 Score",
      "Weighted Precision",
      "Weighted Recall",
      "Weighted F1 Score"
    ),

    Score = c(
      accuracy,
      macro_precision,
      macro_recall,
      macro_f1,
      weighted_precision,
      weighted_recall,
      weighted_f1
    )
  )

  # ==========================================
  # 6. DISPLAY RESULTS
  # ==========================================

  cat("\n========== CLASS-WISE METRICS ==========\n")

  print(class_metrics, row.names = FALSE)

  cat("\n========== OVERALL METRICS ==========\n")

  print(overall_metrics, row.names = FALSE)

  cat("\n========== MODEL SUMMARY ==========\n")

  cat("Model:", model, "\n")
  cat("Accuracy:", round(accuracy * 100, 2), "%\n")

  cat("Macro F1 Score:", round(macro_f1, 4), "\n")

  # ==========================================
  # 7. RETURN EVERYTHING
  # ==========================================

  return(list(

    model = fitted_model,

    predictions = predictions,

    confusion_matrix = cm,

    class_metrics = class_metrics,

    overall_metrics = overall_metrics,

    accuracy = accuracy

  ))
}



regression_model <- function(scale_train_x,
                             scale_test_x,
                             train_y,
                             test_y,
                             model = "multiple_linear",
                             degree = 2,
                             alpha = 0.5,
                             k = 5) {

  # ==========================================
  # 1. VALIDATE INPUTS
  # ==========================================

  model <- tolower(model)

  valid_models <- c(
    "linear",
    "multiple_linear",
    "polynomial",
    "lasso",
    "ridge",
    "elastic_net",
    "decision_tree",
    "random_forest",
    "svm"
  )

  if (!model %in% valid_models) {
    stop(paste(
      "Choose from:",
      paste(valid_models, collapse = ", ")
    ))
  }

  # Convert targets to numeric
  train_y <- as.numeric(train_y)
  test_y <- as.numeric(test_y)

  if (length(train_y) != nrow(scale_train_x) ||
      length(test_y) != nrow(scale_test_x)) {
    stop("Feature rows and target lengths do not match.")
  }

  if (nrow(scale_train_x) < 2 ||
      nrow(scale_test_x) < 1) {
    stop("Insufficient training or testing observations.")
  }

  if (anyNA(train_y) || anyNA(test_y) ||
      any(!is.finite(train_y)) ||
      any(!is.finite(test_y))) {
    stop("Target contains missing or non-finite values.")
  }

  if (anyNA(scale_train_x) || anyNA(scale_test_x)) {
    stop("Missing values found in features.")
  }

  if (degree < 1 || degree != as.integer(degree)) {
    stop("Polynomial degree must be a positive integer.")
  }

  if (alpha < 0 || alpha > 1) {
    stop("Alpha must be between 0 and 1.")
  }

  if (model == "linear" && ncol(scale_train_x) != 1) {
    stop("Simple linear regression requires exactly one feature.")
  }

  if (model == "multiple_linear" && ncol(scale_train_x) < 2) {
    stop("Multiple linear regression requires at least two features.")
  }

  if (model %in% c("lasso", "ridge", "elastic_net") &&
      nrow(scale_train_x) <= k) {
    stop("Training rows must exceed the CV fold count.")
  }

  # Convert features to numeric matrices
  train_x <- as.matrix(scale_train_x)
  test_x <- as.matrix(scale_test_x)

  storage.mode(train_x) <- "double"
  storage.mode(test_x) <- "double"

  if (any(!is.finite(train_x)) ||
      any(!is.finite(test_x))) {
    stop("Features contain infinite or non-finite values.")
  }

  if (ncol(train_x) != ncol(test_x)) {
    stop("Train and test feature counts do not match.")
  }

  if (any(apply(train_x, 2, function(x) sd(x) == 0))) {
    warning("Some training features have zero variance.")
  }

  # Make safe feature names
  feature_names <- paste0("X", seq_len(ncol(train_x)))

  colnames(train_x) <- feature_names
  colnames(test_x) <- feature_names

  train_df <- data.frame(train_x, check.names = FALSE)
  test_df <- data.frame(test_x, check.names = FALSE)

  train_df$target <- train_y

  # ==========================================
  # 2. TRAIN MODEL
  # ==========================================

  set.seed(123)

  cat("\n====================================\n")
  cat("REGRESSION MODEL:", toupper(model), "\n")
  cat("====================================\n")

  fitted_model <- NULL

  if (model %in% c("linear", "multiple_linear")) {

    # Simple / Multiple Linear Regression

    fitted_model <- lm(
      target ~ .,
      data = train_df
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df
    )

  } else if (model == "polynomial") {

    # Polynomial Regression
    # Creates powers of each feature, without interactions

    make_poly <- function(x, degree) {

      result <- as.data.frame(x)

      for (j in seq_len(ncol(x))) {

        for (d in 2:degree) {

          result[[paste0("X", j, "_power_", d)]] <-
            x[, j]^d
        }
      }

      result
    }

    # Degree 1 terms are already present
    poly_train <- make_poly(train_x, degree)
    poly_test <- make_poly(test_x, degree)

    poly_train$target <- train_y

    fitted_model <- lm(
      target ~ .,
      data = poly_train
    )

    predictions <- predict(
      fitted_model,
      newdata = poly_test
    )

  } else if (model %in% c("lasso", "ridge", "elastic_net")) {

    # L1 / L2 / Elastic Net Regularization

    if (model == "lasso") {
      alpha_value <- 1
    } else if (model == "ridge") {
      alpha_value <- 0
    } else {
      alpha_value <- alpha
    }

    fitted_model <- glmnet::cv.glmnet(
      x = train_x,
      y = train_y,
      alpha = alpha_value,
      family = "gaussian",
      nfolds = k,
      standardize = FALSE
    )

    predictions <- predict(
      fitted_model,
      newx = test_x,
      s = "lambda.min"
    )

    predictions <- as.numeric(predictions)

  } else if (model == "decision_tree") {

    # Decision Tree Regression

    fitted_model <- rpart::rpart(
      target ~ .,
      data = train_df,
      method = "anova"
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df
    )

  } else if (model == "random_forest") {

    # Random Forest Regression

    fitted_model <- randomForest::randomForest(
      target ~ .,
      data = train_df,
      ntree = 500,
      importance = TRUE
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df
    )

  } else if (model == "svm") {

    # Support Vector Regression

    fitted_model <- e1071::svm(
      x = train_x,
      y = train_y,
      type = "eps-regression",
      kernel = "radial",
      scale = FALSE
    )

    predictions <- predict(
      fitted_model,
      newdata = test_x
    )
  }

  predictions <- as.numeric(predictions)

  if (length(predictions) != length(test_y) ||
      any(!is.finite(predictions))) {
    stop("Model returned invalid predictions.")
  }

  # ==========================================
  # 3. CALCULATE ERRORS
  # ==========================================

  residuals <- test_y - predictions

  n <- length(test_y)
  p <- ncol(train_x)

  MAE <- mean(abs(residuals))

  MSE <- mean(residuals^2)

  RMSE <- sqrt(MSE)

  # ==========================================
  # 4. R SQUARED
  # ==========================================

  SST <- sum((test_y - mean(test_y))^2)

  SSE <- sum(residuals^2)

  R2 <- if (SST == 0) {
    NA_real_
  } else {
    1 - SSE / SST
  }

  # Adjusted R2
  adjusted_R2 <- if (is.na(R2) || n <= p + 1) {
    NA_real_
  } else {
    1 - (1 - R2) * (n - 1) / (n - p - 1)
  }

  # ==========================================
  # 5. MAPE
  # ==========================================

  nonzero_indices <- which(test_y != 0)

  MAPE <- if (length(nonzero_indices) == 0) {
    NA_real_
  } else {
    mean(
      abs(
        residuals[nonzero_indices] /
          test_y[nonzero_indices]
      )
    ) * 100
  }

  # ==========================================
  # 6. EVALUATION TABLE
  # ==========================================

  metrics <- data.frame(
    Metric = c(
      "MAE",
      "MSE",
      "RMSE",
      "R2",
      "Adjusted R2",
      "MAPE (%)"
    ),

    Value = c(
      MAE,
      MSE,
      RMSE,
      R2,
      adjusted_R2,
      MAPE
    )
  )

  # ==========================================
  # 7. PREDICTION TABLE
  # ==========================================

  prediction_table <- data.frame(
    Actual = test_y,
    Predicted = predictions,
    Residual = residuals
  )

  # ==========================================
  # 8. DISPLAY RESULTS
  # ==========================================

  cat("\n========== REGRESSION METRICS ==========\n")

  print(metrics, row.names = FALSE)

  cat("\n========== PREDICTION DETAILS ==========\n")

  print(head(prediction_table, 10))

  cat("\n========== MODEL SUMMARY ==========\n")

  cat("Model:", model, "\n")
  cat("Training observations:", length(train_y), "\n")
  cat("Testing observations:", n, "\n")
  cat("Number of features:", p, "\n")

  # ==========================================
  # 9. RETURN EVERYTHING
  # ==========================================

  return(list(

    model = fitted_model,

    predictions = predictions,

    actual = test_y,

    residuals = residuals,

    prediction_table = prediction_table,

    metrics = metrics,

    MAE = MAE,

    MSE = MSE,

    RMSE = RMSE,

    R2 = R2,

    adjusted_R2 = adjusted_R2,

    MAPE = MAPE

  ))
}



# ==============================================================================
# PROBABILITY DISTRIBUTION ANALYSIS
# (Unified function from Assignment_05.R)
# ==============================================================================

#' Probability Distribution Analysis
#'
#' A unified function covering Binomial, Poisson, Normal, and Geometric
#' distributions.  Computes PMF/PDF, CDF, quantiles, probabilities,
#' fits observed frequencies, generates random samples, and produces
#' side-by-side comparison plots.
#'
#' @param dist      Distribution type: "binomial", "poisson", "normal",
#'                  or "geometric".
#' @param action    What to compute:
#'   \itemize{
#'     \item \code{"pmf"}       PMF/PDF values (returned + plotted).
#'     \item \code{"cdf"}       CDF values (returned + plotted).
#'     \item \code{"prob"}      Exact, cumulative, tail, or interval
#'                              probabilities.
#'     \item \code{"quantile"}  Quantile / percentile lookup.
#'     \item \code{"fit"}       Fit distribution to observed frequencies,
#'                              print expected counts, overlay plot.
#'     \item \code{"sample"}    Draw random sample, report empirical vs
#'                              theoretical moments, bar/histogram plot.
#'     \item \code{"compare"}   Side-by-side PMF comparison of two
#'                              distributions (\code{compare_dist},
#'                              \code{compare_params}).
#'     \item \code{"joint"}     Joint / marginal probability table from a
#'                              matrix (\code{joint_matrix}).
#'     \item \code{"density"}   Plot density curve(s) for Normal.
#'   }
#' @param size      Binomial \code{n} (number of trials).
#' @param prob      Success probability (Binomial / Geometric).
#' @param lambda    Poisson rate parameter.
#' @param mean      Normal mean (default 0).
#' @param sd        Normal standard deviation (default 1).
#' @param x         Integer vector of values for PMF/CDF.
#'                  For \code{"prob"}: single value or length-2 interval.
#' @param prob_type Probability query type (used with \code{action = "prob"}):
#'                  "exact", "leq", "geq", "between".
#' @param quantile_p  Probability for quantile lookup (e.g. 0.90).
#' @param observed  Numeric vector of observed frequencies
#'                  (used with \code{action = "fit"}).
#' @param n_sample  Sample size for random generation
#'                  (used with \code{action = "sample"}, default 1000).
#' @param seed      Random seed (default 123).
#' @param compare_dist  Second distribution for comparison ("binomial" or
#'                      "poisson").
#' @param compare_params Named list of params for the second distribution,
#'                       e.g. \code{list(lambda = 5)}.
#' @param joint_matrix  A numeric matrix for joint distribution analysis.
#' @param plot      Logical; produce plots? (default TRUE).
#'
#' @return A list containing computed values, data frames, and summary text
#'         (varies by action).
#' @export
probability_distribution <- function(dist      = "binomial",
                                      action    = "pmf",
                                      size      = NULL,
                                      prob      = NULL,
                                      lambda    = NULL,
                                      mean      = 0,
                                      sd        = 1,
                                      x         = NULL,
                                      prob_type = "exact",
                                      quantile_p = NULL,
                                      observed  = NULL,
                                      n_sample  = 1000,
                                      seed      = 123,
                                      compare_dist   = NULL,
                                      compare_params = NULL,
                                      joint_matrix   = NULL,
                                      plot      = TRUE) {

  dist      <- tolower(dist)
  action    <- tolower(action)
  prob_type <- tolower(prob_type)

  valid_dists   <- c("binomial", "poisson", "normal", "geometric")
  valid_actions <- c("pmf", "cdf", "prob", "quantile", "fit",
                     "sample", "compare", "joint", "density")

  if (!dist %in% valid_dists) {
    stop(paste("Choose distribution from:",
               paste(valid_dists, collapse = ", ")))
  }

  if (!action %in% valid_actions) {
    stop(paste("Choose action from:",
               paste(valid_actions, collapse = ", ")))
  }

  results <- list()

  cat("\n====================================\n")
  cat("DISTRIBUTION:", toupper(dist), "| ACTION:", toupper(action), "\n")
  cat("====================================\n")

  # -------------------------------------------------------
  # ACTION: PMF / PDF
  # -------------------------------------------------------
  if (action == "pmf") {

    if (dist == "binomial") {
      if (is.null(size) || is.null(prob))
        stop("Binomial requires 'size' and 'prob'.")
      if (is.null(x)) x <- 0:size
      vals <- dbinom(x, size = size, prob = prob)
      label <- paste0("PMF of Binomial(", size, ", ", prob, ")")
    } else if (dist == "poisson") {
      if (is.null(lambda))
        stop("Poisson requires 'lambda'.")
      if (is.null(x)) x <- 0:qpois(0.999, lambda)
      vals <- dpois(x, lambda = lambda)
      label <- paste0("PMF of Poisson(", lambda, ")")
    } else if (dist == "normal") {
      if (is.null(x))
        x <- seq(mean - 4 * sd, mean + 4 * sd, length.out = 500)
      vals <- dnorm(x, mean = mean, sd = sd)
      label <- paste0("PDF of Normal(", mean, ", ", sd, ")")
    } else if (dist == "geometric") {
      if (is.null(prob))
        stop("Geometric requires 'prob'.")
      if (is.null(x)) x <- 0:qgeom(0.999, prob)
      vals <- dgeom(x, prob = prob)
      label <- paste0("PMF of Geometric(", prob, ")")
    }

    results$x     <- x
    results$pmf   <- vals
    results$table <- data.frame(x = x, probability = round(vals, 6))

    if (plot) {
      if (dist == "normal") {
        plot(x, vals, type = "l", lwd = 2, col = "steelblue",
             main = label, xlab = "x", ylab = "Density")
        abline(v = mean, col = "red", lwd = 2, lty = 2)
      } else {
        plot(x, vals, type = "h", lwd = 2, col = "steelblue",
             main = label, xlab = "x", ylab = "P(X = x)")
        points(x, vals, pch = 16, col = "steelblue")
      }
    }

    cat("\n========== PMF / PDF VALUES ==========\n")
    print(results$table)

  # -------------------------------------------------------
  # ACTION: CDF
  # -------------------------------------------------------
  } else if (action == "cdf") {

    if (dist == "binomial") {
      if (is.null(size) || is.null(prob))
        stop("Binomial requires 'size' and 'prob'.")
      if (is.null(x)) x <- 0:size
      vals <- pbinom(x, size = size, prob = prob)
      label <- paste0("CDF of Binomial(", size, ", ", prob, ")")
    } else if (dist == "poisson") {
      if (is.null(lambda))
        stop("Poisson requires 'lambda'.")
      if (is.null(x)) x <- 0:qpois(0.999, lambda)
      vals <- ppois(x, lambda = lambda)
      label <- paste0("CDF of Poisson(", lambda, ")")
    } else if (dist == "normal") {
      if (is.null(x))
        x <- seq(mean - 4 * sd, mean + 4 * sd, length.out = 500)
      vals <- pnorm(x, mean = mean, sd = sd)
      label <- paste0("CDF of Normal(", mean, ", ", sd, ")")
    } else if (dist == "geometric") {
      if (is.null(prob))
        stop("Geometric requires 'prob'.")
      if (is.null(x)) x <- 0:qgeom(0.999, prob)
      vals <- pgeom(x, prob = prob)
      label <- paste0("CDF of Geometric(", prob, ")")
    }

    results$x     <- x
    results$cdf   <- vals
    results$table <- data.frame(x = x, cumulative_prob = round(vals, 6))

    if (plot) {
      plot(x, vals, type = "s", lwd = 2, col = "darkred",
           main = label, xlab = "x", ylab = "P(X <= x)")
    }

    cat("\n========== CDF VALUES ==========\n")
    print(results$table)

  # -------------------------------------------------------
  # ACTION: PROB (exact / leq / geq / between)
  # -------------------------------------------------------
  } else if (action == "prob") {

    if (!prob_type %in% c("exact", "leq", "geq", "between")) {
      stop("prob_type must be 'exact', 'leq', 'geq', or 'between'.")
    }

    if (is.null(x)) stop("Provide 'x' for probability calculation.")

    compute_prob <- function(d, val, pt) {
      if (d == "binomial") {
        if (is.null(size) || is.null(prob))
          stop("Binomial requires 'size' and 'prob'.")
        if (pt == "exact")   return(dbinom(val, size, prob))
        if (pt == "leq")     return(pbinom(val, size, prob))
        if (pt == "geq")     return(1 - pbinom(val - 1, size, prob))
        if (pt == "between") return(pbinom(val[2], size, prob) -
                                      pbinom(val[1] - 1, size, prob))
      } else if (d == "poisson") {
        if (is.null(lambda))
          stop("Poisson requires 'lambda'.")
        if (pt == "exact")   return(dpois(val, lambda))
        if (pt == "leq")     return(ppois(val, lambda))
        if (pt == "geq")     return(1 - ppois(val - 1, lambda))
        if (pt == "between") return(ppois(val[2], lambda) -
                                      ppois(val[1] - 1, lambda))
      } else if (d == "normal") {
        if (pt == "exact")
          stop("Normal is continuous; use 'leq', 'geq', or 'between'.")
        if (pt == "leq")     return(pnorm(val, mean, sd))
        if (pt == "geq")     return(1 - pnorm(val, mean, sd))
        if (pt == "between") return(pnorm(val[2], mean, sd) -
                                      pnorm(val[1], mean, sd))
      } else if (d == "geometric") {
        if (is.null(prob))
          stop("Geometric requires 'prob'.")
        if (pt == "exact")   return(dgeom(val, prob))
        if (pt == "leq")     return(pgeom(val, prob))
        if (pt == "geq")     return(1 - pgeom(val - 1, prob))
        if (pt == "between") return(pgeom(val[2], prob) -
                                      pgeom(val[1] - 1, prob))
      }
    }

    result_prob <- compute_prob(dist, x, prob_type)

    results$probability <- result_prob
    results$prob_type   <- prob_type
    results$x           <- x

    cat("\nP(", prob_type, ") =", round(result_prob, 6), "\n")

  # -------------------------------------------------------
  # ACTION: QUANTILE
  # -------------------------------------------------------
  } else if (action == "quantile") {

    if (is.null(quantile_p))
      stop("Provide 'quantile_p' (e.g. 0.90).")

    if (dist == "binomial") {
      if (is.null(size) || is.null(prob))
        stop("Binomial requires 'size' and 'prob'.")
      q_val <- qbinom(quantile_p, size, prob)
    } else if (dist == "poisson") {
      if (is.null(lambda))
        stop("Poisson requires 'lambda'.")
      q_val <- qpois(quantile_p, lambda)
    } else if (dist == "normal") {
      q_val <- qnorm(quantile_p, mean, sd)
    } else if (dist == "geometric") {
      if (is.null(prob))
        stop("Geometric requires 'prob'.")
      q_val <- qgeom(quantile_p, prob)
    }

    results$quantile   <- q_val
    results$quantile_p <- quantile_p

    cat("\n", quantile_p * 100, "th percentile =", q_val, "\n")

  # -------------------------------------------------------
  # ACTION: FIT observed frequencies
  # -------------------------------------------------------
  } else if (action == "fit") {

    if (is.null(observed))
      stop("Provide 'observed' frequency vector.")

    N <- sum(observed)
    x_vals <- 0:(length(observed) - 1)

    if (dist == "binomial") {
      n_trials <- length(observed) - 1
      mean_x   <- sum(x_vals * observed) / N
      p_hat    <- mean_x / n_trials
      fitted_p <- dbinom(x_vals, n_trials, p_hat)
      est_label <- paste("Estimated p =", round(p_hat, 4))
      results$estimated_p <- p_hat
    } else if (dist == "poisson") {
      lambda_hat <- sum(x_vals * observed) / N
      fitted_p   <- dpois(x_vals, lambda_hat)
      est_label  <- paste("Estimated lambda =", round(lambda_hat, 4))
      results$estimated_lambda <- lambda_hat
    } else {
      stop("Fit action is supported only for 'binomial' and 'poisson'.")
    }

    expected <- fitted_p * N

    fit_table <- data.frame(
      x            = x_vals,
      observed     = observed,
      fitted_prob  = round(fitted_p, 4),
      expected     = round(expected, 2)
    )

    results$fit_table <- fit_table

    cat("\n", est_label, "\n")
    cat("\n========== FIT TABLE ==========\n")
    print(fit_table)

    if (plot) {
      plot(x_vals, observed, type = "h", lwd = 5, col = "steelblue",
           main = paste("Observed vs Fitted", toupper(dist), "Frequencies"),
           xlab = "x", ylab = "Frequency",
           ylim = c(0, max(observed, expected) + 5))
      points(x_vals + 0.12, expected, type = "h", lwd = 5, col = "red")
      legend("topright",
             legend = c("Observed", "Fitted"),
             col    = c("steelblue", "red"), lwd = 3)
    }

  # -------------------------------------------------------
  # ACTION: SAMPLE
  # -------------------------------------------------------
  } else if (action == "sample") {

    set.seed(seed)

    if (dist == "binomial") {
      if (is.null(size) || is.null(prob))
        stop("Binomial requires 'size' and 'prob'.")
      samp   <- rbinom(n_sample, size, prob)
      t_mean <- size * prob
      t_sd   <- sqrt(size * prob * (1 - prob))
    } else if (dist == "poisson") {
      if (is.null(lambda))
        stop("Poisson requires 'lambda'.")
      samp   <- rpois(n_sample, lambda)
      t_mean <- lambda
      t_sd   <- sqrt(lambda)
    } else if (dist == "normal") {
      samp   <- rnorm(n_sample, mean, sd)
      t_mean <- mean
      t_sd   <- sd
    } else if (dist == "geometric") {
      if (is.null(prob))
        stop("Geometric requires 'prob'.")
      samp   <- rgeom(n_sample, prob)
      t_mean <- (1 - prob) / prob
      t_sd   <- sqrt((1 - prob) / prob^2)
    }

    emp_mean <- base::mean(samp)
    emp_sd   <- stats::sd(samp)

    results$sample           <- samp
    results$empirical_mean   <- emp_mean
    results$empirical_sd     <- emp_sd
    results$theoretical_mean <- t_mean
    results$theoretical_sd   <- t_sd

    cat("\nSample size:", n_sample, "\n")
    cat("Empirical  mean =", round(emp_mean, 3),
        " | Theoretical mean =", round(t_mean, 3), "\n")
    cat("Empirical  sd   =", round(emp_sd, 3),
        " | Theoretical sd   =", round(t_sd, 3), "\n")

    if (plot) {
      if (dist == "normal") {
        hist(samp, breaks = 30, col = "steelblue", border = "white",
             main = paste("Random Sample from", toupper(dist)),
             xlab = "Value", freq = FALSE)
        curve(dnorm(x, mean, sd), add = TRUE, col = "red", lwd = 2)
      } else {
        barplot(table(samp), col = "steelblue",
                main = paste("Random Sample from", toupper(dist)),
                xlab = "Value", ylab = "Frequency")
      }
    }

  # -------------------------------------------------------
  # ACTION: COMPARE two distributions
  # -------------------------------------------------------
  } else if (action == "compare") {

    if (is.null(compare_dist) || is.null(compare_params))
      stop("Provide 'compare_dist' and 'compare_params' for comparison.")

    # Primary distribution PMF
    if (dist == "binomial") {
      if (is.null(size) || is.null(prob))
        stop("Binomial requires 'size' and 'prob'.")
      if (is.null(x)) x <- 0:(size + 5)
      pmf1  <- dbinom(x, size, prob)
      lab1  <- paste0("Binomial(", size, ", ", prob, ")")
    } else if (dist == "poisson") {
      if (is.null(lambda))
        stop("Poisson requires 'lambda'.")
      if (is.null(x)) x <- 0:qpois(0.999, lambda)
      pmf1  <- dpois(x, lambda)
      lab1  <- paste0("Poisson(", lambda, ")")
    } else {
      stop("Compare action supports 'binomial' and 'poisson' only.")
    }

    # Second distribution PMF
    cd <- tolower(compare_dist)
    if (cd == "binomial") {
      pmf2 <- dbinom(x, compare_params$size, compare_params$prob)
      lab2 <- paste0("Binomial(", compare_params$size,
                     ", ", compare_params$prob, ")")
    } else if (cd == "poisson") {
      pmf2 <- dpois(x, compare_params$lambda)
      lab2 <- paste0("Poisson(", compare_params$lambda, ")")
    } else {
      stop("compare_dist must be 'binomial' or 'poisson'.")
    }

    results$x    <- x
    results$pmf1 <- pmf1
    results$pmf2 <- pmf2

    if (plot) {
      plot(x, pmf1, type = "h", lwd = 5, col = "steelblue",
           main = paste("PMF Comparison:", lab1, "vs", lab2),
           xlab = "x", ylab = "P(X = x)",
           ylim = c(0, max(pmf1, pmf2) + 0.03))
      points(x + 0.15, pmf2, type = "h", lwd = 5, col = "red")
      legend("topright", legend = c(lab1, lab2),
             col = c("steelblue", "red"), lwd = 3)
    }

    cat("\nComparison plotted.\n")

  # -------------------------------------------------------
  # ACTION: JOINT probability table
  # -------------------------------------------------------
  } else if (action == "joint") {

    if (is.null(joint_matrix) || !is.matrix(joint_matrix))
      stop("Provide a numeric 'joint_matrix'.")

    results$joint_freq     <- joint_matrix
    results$joint_prob     <- prop.table(joint_matrix)
    results$marginal_row   <- rowSums(joint_matrix)
    results$marginal_col   <- colSums(joint_matrix)

    cat("\n========== JOINT FREQUENCY TABLE ==========\n")
    print(joint_matrix)
    cat("\n========== JOINT PROBABILITY TABLE ==========\n")
    print(round(results$joint_prob, 4))
    cat("\nMarginal distribution of rows (row sums):\n")
    print(results$marginal_row)
    cat("\nMarginal distribution of columns (column sums):\n")
    print(results$marginal_col)

  # -------------------------------------------------------
  # ACTION: DENSITY curves (Normal)
  # -------------------------------------------------------
  } else if (action == "density") {

    if (dist != "normal")
      stop("Density curve action is for 'normal' distribution only.")

    if (is.null(x))
      x <- seq(mean - 4 * max(sd, 1), mean + 4 * max(sd, 1),
               length.out = 500)

    y <- dnorm(x, mean = mean, sd = sd)

    results$x <- x
    results$density <- y

    if (plot) {
      plot(x, y, type = "l", lwd = 2, col = "steelblue",
           main = paste0("Density of N(", mean, ", ", sd, "^2)"),
           xlab = "x", ylab = "Density")
      abline(v = mean, col = "red", lwd = 2, lty = 2)
    }

    cat("\nDensity curve plotted for N(", mean, ",", sd, "^2).\n")
  }

  cat("\n====================================\n")

  invisible(results)
}


# ==============================================================================
# CLUSTERING ANALYSIS
# (Unified function from assignment7_answers.R)
# ==============================================================================

#' Clustering Analysis
#'
#' Performs end-to-end unsupervised clustering: data exploration,
#' correlation analysis, PCA, K-Means (with Elbow and Silhouette),
#' Hierarchical Clustering (with dendrogram), cluster profiling,
#' and statistical tests (t-test, Wilcoxon, ANOVA, Tukey, Chi-squared).
#'
#' @param data          A data frame.
#' @param num_cols      Character vector of numeric column names to use
#'                      for clustering.  If NULL, all numeric columns are
#'                      used.
#' @param k             Number of clusters for K-Means and Hierarchical
#'                      (default 3).
#' @param max_k         Maximum k to evaluate in the Elbow method
#'                      (default 10).
#' @param hc_method     Linkage method for hierarchical clustering
#'                      (default "ward.D2").
#' @param profile_cols  Character vector of column names to profile /
#'                      aggregate by cluster.  Defaults to \code{num_cols}.
#' @param cat_cols      Character vector of categorical columns for
#'                      chi-squared tests against clusters (e.g.
#'                      \code{c("VisitorType", "Revenue")}).
#' @param test_var      Name of a numeric variable for statistical tests
#'                      (t-test, Wilcoxon, ANOVA) across clusters.
#' @param test_clusters Integer vector of length 2 — which two clusters to
#'                      use for pairwise t-test / Wilcoxon (default c(1,2)).
#' @param cor_threshold Threshold for flagging high correlations
#'                      (default 0.7).
#' @param seed          Random seed (default 123).
#' @param plot          Logical; produce plots? (default TRUE).
#'
#' @return A list containing:
#'   \itemize{
#'     \item \code{scaled_data}       Scaled numeric matrix.
#'     \item \code{cor_matrix}        Correlation matrix.
#'     \item \code{high_corr}         Pairs above \code{cor_threshold}.
#'     \item \code{pca}               PCA result (\code{prcomp} object).
#'     \item \code{variance_explained} Proportion of variance per PC.
#'     \item \code{wcss}              Within-cluster SS for k = 1..max_k.
#'     \item \code{kmeans}            K-Means result.
#'     \item \code{km_centroids}      Centroid means per cluster.
#'     \item \code{km_silhouette}     Mean silhouette width.
#'     \item \code{hclust}            Hierarchical clustering result.
#'     \item \code{hc_clusters}       Hierarchical cluster assignments.
#'     \item \code{cross_table}       KMeans vs Hierarchical cross-tab.
#'     \item \code{cluster_profile}   Aggregate means by cluster.
#'     \item \code{ttest}             Welch t-test result (if applicable).
#'     \item \code{wilcox}            Wilcoxon test result (if applicable).
#'     \item \code{anova}             ANOVA result (if applicable).
#'     \item \code{tukey}             Tukey HSD result (if applicable).
#'     \item \code{chisq_tests}       List of chi-squared test results.
#'   }
#' @export
clustering_analysis <- function(data,
                                 num_cols       = NULL,
                                 k              = 3,
                                 max_k          = 10,
                                 hc_method      = "ward.D2",
                                 profile_cols   = NULL,
                                 cat_cols       = NULL,
                                 test_var       = NULL,
                                 test_clusters  = c(1, 2),
                                 cor_threshold  = 0.7,
                                 seed           = 123,
                                 plot           = TRUE) {

  if (!is.data.frame(data) || nrow(data) == 0) {
    stop("Provide a non-empty data frame.")
  }

  results <- list()

  # ==========================================
  # 1. BASIC EXPLORATION
  # ==========================================

  cat("\n========== DATASET INFORMATION ==========\n")
  cat("Dimensions:", dim(data), "\n")
  cat("\nStructure:\n")
  str(data)
  cat("\nSummary:\n")
  print(summary(data))
  cat("\nMissing values per column:\n")
  print(colSums(is.na(data)))

  # ==========================================
  # 2. REMOVE DUPLICATES
  # ==========================================

  original_rows <- nrow(data)
  data <- data[!duplicated(data), ]
  dup_removed   <- original_rows - nrow(data)

  cat("\nOriginal rows:", original_rows,
      "| Cleaned rows:", nrow(data),
      "| Duplicates removed:", dup_removed, "\n")

  results$cleaned_data <- data

  # ==========================================
  # 3. SELECT NUMERIC COLUMNS
  # ==========================================

  if (is.null(num_cols)) {
    num_cols <- names(data)[sapply(data, is.numeric)]
    cat("\nAuto-detected numeric columns:",
        paste(num_cols, collapse = ", "), "\n")
  }

  if (length(num_cols) < 2)
    stop("Need at least 2 numeric columns for clustering.")

  df_num <- data[, num_cols, drop = FALSE]

  if (is.null(profile_cols)) profile_cols <- num_cols

  # ==========================================
  # 4. SCALING
  # ==========================================

  scaled_num <- scale(df_num)
  results$scaled_data <- scaled_num

  cat("\n========== SCALING DONE ==========\n")

  # ==========================================
  # 5. HISTOGRAMS
  # ==========================================

  if (plot) {
    n_cols <- length(num_cols)
    n_row  <- ceiling(n_cols / 5)
    par(mfrow = c(n_row, min(n_cols, 5)), mar = c(3, 3, 2, 1))
    for (col in num_cols) {
      hist(df_num[[col]], main = col, col = "skyblue",
           xlab = "", border = "white")
    }
    par(mfrow = c(1, 1))
  }

  # ==========================================
  # 6. CORRELATION ANALYSIS
  # ==========================================

  cor_matrix <- cor(df_num, use = "complete.obs")
  results$cor_matrix <- cor_matrix

  if (plot) {
    if (requireNamespace("corrplot", quietly = TRUE)) {
      corrplot::corrplot(cor_matrix, method = "circle",
                         type = "upper", tl.col = "black", tl.srt = 45)
    }
  }

  high_corr <- which(
    abs(cor_matrix) > cor_threshold & abs(cor_matrix) < 1,
    arr.ind = TRUE
  )

  results$high_corr <- high_corr

  cat("\n========== HIGH CORRELATIONS (>",
      cor_threshold, ") ==========\n")
  print(high_corr)

  # ==========================================
  # 7. PCA
  # ==========================================

  pca_res <- prcomp(scaled_num, center = TRUE, scale. = FALSE)
  results$pca <- pca_res

  var_explained <- (pca_res$sdev)^2 / sum((pca_res$sdev)^2)
  results$variance_explained <- var_explained

  cat("\n========== PCA SUMMARY ==========\n")
  print(summary(pca_res))

  if (plot) {
    # Scree plot
    plot(var_explained, type = "b", pch = 19, col = "blue",
         xlab = "Principal Component",
         ylab = "Proportion of Variance Explained",
         main = "Scree Plot")

    # PC1 vs PC2
    plot(pca_res$x[, 1], pca_res$x[, 2],
         col = rgb(0, 0.4, 0.8, 0.3), pch = 16,
         xlab = "PC1", ylab = "PC2",
         main = "PCA Observation Plot (PC1 vs PC2)")
  }

  # ==========================================
  # 8. ELBOW METHOD
  # ==========================================

  set.seed(seed)
  wcss <- numeric(max_k)
  for (i in 1:max_k) {
    wcss[i] <- sum(
      kmeans(scaled_num, centers = i, nstart = 10)$withinss
    )
  }
  results$wcss <- wcss

  if (plot) {
    plot(1:max_k, wcss, type = "b", pch = 19, col = "darkgreen",
         xlab = "Number of Clusters (k)",
         ylab = "Total Within-Cluster SS",
         main = "Elbow Method for Optimal k")
  }

  # ==========================================
  # 9. K-MEANS CLUSTERING
  # ==========================================

  set.seed(seed)
  km_res <- kmeans(scaled_num, centers = k, nstart = 25)
  results$kmeans <- km_res

  data$km_cluster <- as.factor(km_res$cluster)

  # Centroids
  centroids_km <- aggregate(
    df_num,
    by = list(Cluster = km_res$cluster),
    mean
  )
  results$km_centroids <- centroids_km

  cat("\n========== K-MEANS CENTROIDS (k =", k, ") ==========\n")
  print(round(centroids_km, 2))

  if (plot) {
    plot(pca_res$x[, 1], pca_res$x[, 2],
         col = km_res$cluster, pch = 16,
         xlab = "PC1", ylab = "PC2",
         main = paste("K-Means Clusters (k =", k, ")"))
    legend("topright",
           legend = paste("Cluster", 1:k),
           col = 1:k, pch = 16)
  }

  # ==========================================
  # 10. SILHOUETTE
  # ==========================================

  if (requireNamespace("cluster", quietly = TRUE)) {
    dist_matrix <- dist(scaled_num)
    sil_km <- cluster::silhouette(km_res$cluster, dist_matrix)
    mean_sil <- base::mean(sil_km[, 3])
    results$km_silhouette <- mean_sil

    cat("\nMean Silhouette Width:", round(mean_sil, 4), "\n")

    if (plot) {
      plot(sil_km, col = 1:k, border = NA,
           main = "Silhouette Plot for K-Means")
    }
  }

  # ==========================================
  # 11. HIERARCHICAL CLUSTERING
  # ==========================================

  dist_matrix <- dist(scaled_num)
  hc_res <- hclust(dist_matrix, method = hc_method)
  results$hclust <- hc_res

  hc_clusters <- cutree(hc_res, k = k)
  data$hc_cluster <- as.factor(hc_clusters)
  results$hc_clusters <- hc_clusters

  if (plot) {
    plot(hc_res,
         main = paste("Dendrogram (", hc_method, " Linkage)"),
         xlab = "", sub = "", labels = FALSE)
    rect.hclust(hc_res, k = k, border = "red")
  }

  # ==========================================
  # 12. CROSS-TABULATION
  # ==========================================

  cross_tab <- table(
    KMeans       = data$km_cluster,
    Hierarchical = data$hc_cluster
  )
  results$cross_table <- cross_tab

  cat("\n========== K-MEANS vs HIERARCHICAL ==========\n")
  print(cross_tab)

  # ==========================================
  # 13. CLUSTER PROFILING
  # ==========================================

  valid_profile <- intersect(profile_cols, names(data))
  if (length(valid_profile) > 0) {
    profile_formula <- as.formula(
      paste("cbind(",
            paste(valid_profile, collapse = ", "),
            ") ~ km_cluster")
    )
    cluster_profile <- aggregate(
      profile_formula,
      data = data,
      FUN  = base::mean
    )
    results$cluster_profile <- cluster_profile

    cat("\n========== CLUSTER PROFILE (MEANS) ==========\n")
    print(round(cluster_profile, 2))
  }

  # ==========================================
  # 14. STATISTICAL TESTS
  # ==========================================

  if (!is.null(test_var)) {

    if (!test_var %in% names(data))
      stop(paste("test_var", test_var, "not found in data."))

    cat("\n========== STATISTICAL TESTS on",
        test_var, "==========\n")

    # Confidence intervals per cluster
    calc_ci <- function(x) {
      res <- t.test(x)
      c(Mean     = base::mean(x),
        Std_Dev  = stats::sd(x),
        CI_Lower = res$conf.int[1],
        CI_Upper = res$conf.int[2])
    }
    ci_by_cluster <- by(data[[test_var]], data$km_cluster, calc_ci)

    cat("\nConfidence Intervals by Cluster:\n")
    print(ci_by_cluster)

    # Pairwise t-test
    c1 <- data[[test_var]][data$km_cluster == test_clusters[1]]
    c2 <- data[[test_var]][data$km_cluster == test_clusters[2]]

    if (length(c1) > 1 && length(c2) > 1) {
      ttest_res  <- t.test(c1, c2, var.equal = FALSE)
      wilcox_res <- wilcox.test(c1, c2)

      results$ttest  <- ttest_res
      results$wilcox <- wilcox_res

      cat("\n--- Welch t-test (Cluster", test_clusters[1],
          "vs", test_clusters[2], ") ---\n")
      print(ttest_res)

      cat("\n--- Wilcoxon Rank-Sum Test ---\n")
      print(wilcox_res)
    }

    # ANOVA + Tukey
    anova_res <- aov(
      as.formula(paste(test_var, "~ km_cluster")),
      data = data
    )
    results$anova <- anova_res

    cat("\n--- ANOVA ---\n")
    print(summary(anova_res))

    tukey_res <- TukeyHSD(anova_res)
    results$tukey <- tukey_res

    cat("\n--- Tukey HSD ---\n")
    print(tukey_res)

    # Descriptive by cluster
    if (requireNamespace("psych", quietly = TRUE)) {
      cat("\n--- Descriptive Statistics by Cluster ---\n")
      print(psych::describeBy(
        data[[test_var]],
        group = data$km_cluster,
        mat   = TRUE
      ))
    }
  }

  # ==========================================
  # 15. CHI-SQUARED TESTS
  # ==========================================

  if (!is.null(cat_cols)) {

    results$chisq_tests <- list()

    cat("\n========== CHI-SQUARED TESTS ==========\n")

    for (cc in cat_cols) {
      if (!cc %in% names(data)) {
        cat("Warning:", cc, "not found, skipping.\n")
        next
      }

      ct <- table(data$km_cluster, data[[cc]])
      chi_res <- chisq.test(ct)
      results$chisq_tests[[cc]] <- chi_res

      cat("\n--- Chi-squared test: km_cluster vs", cc, "---\n")
      print(chi_res)

      cat("\nContingency Table:\n")
      print(ct)
    }
  }

  # ==========================================
  # 16. RETURN
  # ==========================================

  cat("\n========== CLUSTERING ANALYSIS COMPLETE ==========\n")

  results$data <- data
  invisible(results)
}

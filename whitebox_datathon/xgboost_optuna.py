import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler


def load_data(train_path, test_path):
    """
    Load training and test data from CSV files.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the test CSV file.

    Returns:
        pd.DataFrame, pd.DataFrame: Training and test data.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_data(df, is_train=True):
    """
    Preprocess the data.

    Args:
        df (pd.DataFrame): Input data frame.
        is_train (bool): Flag to indicate if the data is training data.

    Returns:
        pd.DataFrame, pd.Series or pd.DataFrame: Processed features and target variable if training data.
    """


def objective(trial, X_train, y_train, X_valid, y_valid):
    """
    Objective function for Optuna.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_valid (pd.DataFrame): Validation features.
        y_valid (pd.Series): Validation target.

    Returns:
        float: RMSE of the model.
    """
    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    model = xgb.train(
        param,
        xgb.DMatrix(X_train, label=y_train),
        evals=[(xgb.DMatrix(X_valid, label=y_valid), "validation")],
        early_stopping_rounds=10,
    )

    preds = model.predict(xgb.DMatrix(X_valid))
    rmse = mean_squared_error(y_valid, preds, squared=False)
    return rmse


def train_and_predict(train_path, test_path, submission_path):
    """
    Train the XGBoost model with Optuna hyperparameter tuning and make predictions.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the test CSV file.
        submission_path (str): Path to save the submission CSV file.
    """
    # Load data
    train_df, test_df = load_data(train_path, test_path)

    # Preprocess data
    X_train, y_train = preprocess_data(train_df)
    X_test = preprocess_data(test_df, is_train=False)

    # Split training data for validation
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(
            trial, X_train_split, y_train_split, X_valid_split, y_valid_split
        ),
        n_trials=100,
    )

    # Train final model with best hyperparameters on all training data
    best_params = study.best_trial.params
    best_params["verbosity"] = 0
    best_params["objective"] = "reg:squarederror"
    best_params["eval_metric"] = "rmse"

    model = xgb.train(best_params, xgb.DMatrix(X_train, label=y_train))

    # Make predictions on the test set
    test_preds = model.predict(xgb.DMatrix(X_test))

    # Create the submission DataFrame
    submission_df = pd.DataFrame({"id": test_df["id"], "price": test_preds})

    # Save the submission file
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")


if __name__ == "__main__":
    # Path to your CSV file
    file_path = "path_to_your_csv_file.csv"
    train_model(file_path)

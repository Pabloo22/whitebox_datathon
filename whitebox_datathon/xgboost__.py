import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


from whitebox_datathon.pipelines import load_data, preprocess_data


def train_xgboost(
    df: pd.DataFrame, test_df: pd.DataFrame, submission_path="submission.csv"
):
    """
    Train an XGBoost model with default parameters and make predictions.

    Args:
        df (pd.DataFrame): Preprocessed training data.
        test_df (pd.DataFrame): Preprocessed test data.
        submission_path (str): Path to save the submission CSV file.
    """
    # Split features and target variable
    X = df.drop(columns=["price"])
    y = df["price"]
    X_test = test_df.drop(columns=["id"])  # Ensure 'id' is not used as a feature

    # Split the training data for validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.05, random_state=42
    )

    # Define XGBoost parameters
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 12,
        "eta": 0.3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # Train the model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test)
    evals = [(dtrain, "train"), (dvalid, "eval")]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=True,
    )

    # Make predictions on the test set
    test_preds = model.predict(dtest)

    # Create the submission DataFrame
    submission_df = pd.DataFrame({"id": test_df["id"], "price": test_preds})

    # Save the submission file
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")


if __name__ == "__main__":
    # Paths to your CSV files

    # Load data
    train_df, test_df = load_data()

    # Preprocess data
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df, is_train=False)

    # Train the model and create the submission file
    train_xgboost(train_df, test_df, "submission3.csv")


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error


# def train_random_forest(
#     df: pd.DataFrame, test_df: pd.DataFrame, submission_path="submission_2.csv"
# ):
#     """
#     Train a Random Forest model and make predictions.

#     Args:
#         df (pd.DataFrame): Preprocessed training data.
#         test_df (pd.DataFrame): Preprocessed test data.
#         submission_path (str): Path to save the submission CSV file.
#     """
#     # Split features and target variable
#     X = df.drop(columns=["price"])
#     y = df["price"]
#     X_test = test_df.drop(columns=["id"])  # Ensure 'id' is not used as a feature

#     # Split the training data for validation
#     X_train, X_valid, y_train, y_valid = train_test_split(
#         X, y, test_size=0.05, random_state=42
#     )

#     # Define Random Forest parameters
#     rf_params = {
#         "n_estimators": 200,
#         "max_depth": 10,
#         "min_samples_split": 2,
#         "min_samples_leaf": 1,
#         "random_state": 42,
#         "n_jobs": -1,
#     }

#     # Train the model
#     model = RandomForestRegressor(**rf_params)
#     model.fit(X_train, y_train)

#     # Validate the model
#     valid_preds = model.predict(X_valid)
#     rmse = mean_squared_error(y_valid, valid_preds, squared=False)
#     print(f"Validation RMSE: {rmse}")

#     # Make predictions on the test set
#     test_preds = model.predict(X_test)

#     # Create the submission DataFrame
#     submission_df = pd.DataFrame({"id": test_df["id"], "price": test_preds})

#     # Save the submission file
#     submission_df.to_csv(submission_path, index=False)
#     print(f"Submission file saved to {submission_path}")

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import warnings
import sys
 
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
 
    folder_name = 'Heart_Disease_Prediction_preprocessing'
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name)

    df_train = pd.read_csv(os.path.join(file_path, 'train.csv'))
    df_test = pd.read_csv(os.path.join(file_path, 'test.csv'))
 
    X_train = df_train.drop(columns=['Heart Disease'])
    X_test = df_test.drop(columns=['Heart Disease'])
    y_train = df_train['Heart Disease'].apply(lambda x: "presence" if x==1 else "absense")
    y_test = df_test['Heart Disease'].apply(lambda x: "presence" if x==1 else "absense")

    input_example = X_train[0:5]
    c = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
 
    with mlflow.start_run():
        model = LogisticRegression(max_iter=max_iter, C=c, random_state=42)
        model.fit(X_train, y_train)
 
        predicted_qualities = model.predict(X_test)
 
        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
        )
        # Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
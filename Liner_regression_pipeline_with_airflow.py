import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(target_column, **kwargs):
    
    task_instance = kwargs['ti']
    
    data = task_instance.xcom_pull(task_ids="read_csv")
    data = data.drop('id', axis = 1)
    data = data.drop('Unnamed: 32', axis = 1)
    
    le = LabelEncoder()

    data['diagnosis']= le.fit_transform(data['diagnosis'])
    
    corr = data.corr()
    
    corr_threshold = 0.6
    selected_features = corr.index[np.abs(corr['diagnosis']) >= corr_threshold]
    
    new_cancer_data = data[selected_features]
    
    
    X = new_cancer_data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train.to_dict(), X_test.to_dict(), y_train.to_dict(), y_test.to_dict()


def train_linear_regression( **kwargs):
    
    task_instance = kwargs['ti']
    
    X_train_dict, _, y_train_dict, _ = task_instance.xcom_pull(task_ids="preprocess_data")
    
    X_train = pd.DataFrame.from_dict(X_train_dict)
    y_train = pd.Series(y_train_dict)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    model_filepath = "/mnt/c/Users/Abdullah/Desktop/airflow_task/linear_regression_model.pkl"
    joblib.dump(model, model_filepath)
    
    return model_filepath


def test_model(**kwargs):
    
    task_instance = kwargs['ti']
    
    model_filepath = task_instance.xcom_pull(task_ids="train_linear_regression")
    
    model = joblib.load(model_filepath)
    
    _, X_test_dict, _, y_test_dict = task_instance.xcom_pull(task_ids="preprocess_data")
    
    X_test = pd.DataFrame.from_dict(X_test_dict)
    y_test = pd.Series(y_test_dict)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2023, 5, 3),
}


dag = DAG(
    "linear_regression_pipeline",
    default_args=default_args,
    description="A pipeline to read CSV, preprocess data, train and test a linear regression model",
    schedule_interval=timedelta(days=1),
    catchup=False,
)

file_path = "/mnt/c/Users/Abdullah/Desktop/airflow_task/data.csv"
target_column = "diagnosis"

t1 = PythonOperator(
    task_id="read_csv",
    python_callable=read_csv,
    op_args=[file_path],
    dag=dag,
)

t2 = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    op_args=[target_column],
    provide_context=True,
    dag=dag,
)

t3 = PythonOperator(
    task_id="train_linear_regression",
    python_callable=train_linear_regression,
    provide_context=True,
    dag=dag,
)

t4 = PythonOperator(
    task_id="test_model",
    python_callable=test_model,
    provide_context=True,
    dag=dag,
)


t1 >> t2 >> t3 >> t4
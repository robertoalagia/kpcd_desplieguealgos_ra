
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pyngrok import ngrok

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer, load_wine

import argparse
import subprocess
import time
import mlflow

def argumentos():
    print("entrando en argumentos")
    parser = argparse.ArgumentParser(description='__main__ de la aplicación con argumentos de entrada.')
    parser.add_argument('--nombre_job', type=str, help='Valor para el parámetro nombre_documento.')
    parser.add_argument('--c_list', nargs='+', type=float, help='List of c values.')
    return parser.parse_args()

def load_dataset():
    wine = load_wine()
    df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
    df['target'] = wine['target']
    return df

def data_treatment(df):
    # Split data into train and test sets
    train, test = train_test_split(df, test_size=0.2)
    test_target = test['target']
    test[['target']].to_csv('test-target.csv', index=False)
    del test['target']
    test.to_csv('test.csv', index=False)

    features = [x for x in list(train.columns) if x != 'target']
    x_raw = train[features]
    y_raw = train['target']
    x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw,
                                                        test_size=.20,
                                                        random_state=123,
                                                        stratify=y_raw)
    return x_train, x_test, y_train, y_test

def mlflow_tracking(nombre_job, x_train, x_test, y_train, y_test, c_list):
    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui', '--port', '5000'])
    print(mlflow_ui_process)
    time.sleep(5)
    mlflow.set_experiment(nombre_job)
    for i in c_list:
        with mlflow.start_run() as run:
            clf = SVC(C=1.5, degree= 3, gamma = 'scale',
              class_weight='balanced', verbose= True,
              random_state=42)

            preprocessor = Pipeline(steps=[('scaler', StandardScaler())])

            model = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('SVC', clf)])
            model.fit(x_train, y_train)
            accuracy_train = model.score(x_train, y_train)
            accuracy_test = model_svc.score(x_test,y_test)

            y_pred = model_svc.predict(x_test)

            cl_report = classification_report(y_test, y_pred, output_dict= True)
            cl_report_df = pd.DataFrame(cl_report).transpose()
            precision = cl_report_df['precision']
            recall = cl_report_df['recall']

            mlflow.log_metric('accuracy_train', accuracy_train)
            mlflow.log_metric('accuracy test', accuracy_test)
            mlflow.log_metric('recall para target 0', recall[0])
            mlflow.log_metric('recall para target 1', recall[1])
            mlflow.log_metric('recall para target 2', recall[2])
            mlflow.log_metric('precision para target 0', precision[0])
            mlflow.log_metric('precision para target 1', precision[1])
            mlflow.log_metric('precision para target 2', precision[2])
            mlflow.log_param('c_list', i)
            mlflow.sklearn.log_model(model, 'clf-model')

            ngrok.kill()

            NGROK_AUTH_TOKEN = '2iizevnFIykG1DSnz2KCri7WoUF_D1rww3iJZk4SR8hfxDzo'
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)
            ngrok_tunnel = ngrok.connect(addr='5000', proto= 'http', bind_tls = True)

    print("Se ha acabado el entrenamiento del modelo correctamente")

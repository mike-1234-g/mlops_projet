import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from dotenv import load_dotenv

load_dotenv()
wd = os.getenv("working_directory")
#sys.path.append(f'{wd}/src/fonctions')

from fonctions_modelisations import Grid_Search_RFR, Score, Create_Train_Test
from fonctions_traitement import data_v1


mlflow.set_experiment("House Price Prediction")
mlflow.set_tracking_uri(uri = f"file:./mlruns")

print(f"Tracking URI: {mlflow.get_tracking_uri()}")

def training_model_RFR(dataset_path):
    """
    Applique les transformations de la fonction data_v1
    df est split en données d'entrainement et de test (80/20)
    Train les 2 modeles LinearRegression et RandomForestRegressor sur les données d'entrainement
    Test sur les jeux de test
    Retourne les métriques de qualités et les valeurs des hyperparamètres utilisés
    """
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    print(dataset_name)
    df = pd.read_csv(dataset_path)
    data_V1 = data_v1(df)
    X_train, X_test, y_train, y_test = Create_Train_Test(data_V1)
    
    #RandomForestRegressor
    GS_RFR, best_RFR = Grid_Search_RFR(X_train, y_train)
    params_RFR = best_RFR.get_params()
    metrics_RFR, y_pred = Score(best_RFR, X_test, y_test)

    input_example = X_test.head(1)
    output_example = y_pred[0]

    signature = infer_signature(input_example, output_example)

    dico_RFR = {'model_name' : f'RFR_{dataset_name}','model': GS_RFR, 'params':params_RFR, 'metrics': metrics_RFR, 
                'input_example':input_example, 'signature':signature}
    
    return  dico_RFR

def test():

    dataset_path = f'{wd}/data/train_concat_year/DKHousing_1992_2022.csv'
    print(training_model_RFR(dataset_path))
    

def main():

    dataset_path = f'{wd}/data/train_concat_year/DKHousing_1992_2022.csv'

    list_datasets_paths = [f'{wd}/data/train_concat_year/DKHousing_1992_2020.csv',
                           f'{wd}/data/train_concat_year/DKHousing_1992_2021.csv',
                           f'{wd}/data/train_concat_year/DKHousing_1992_2022.csv']
    
    for dataset_path in list_datasets_paths:

        dico_info = training_model_RFR(dataset_path)

        model_name = dico_info['model_name']
        model = dico_info['model']
        params = dico_info['params']
        metrics = dico_info['metrics']
        input_example = dico_info['input_example']
        signature = dico_info['signature']

        print(model_name, '\n', model, '\n', params, '\n', metrics)

        if mlflow.active_run():
            print(f"Run actif détecté : {mlflow.active_run().info.run_id}")
            mlflow.end_run()
        
        with mlflow.start_run(run_name=model_name, nested=True) as run:

            print(f"Run ID: {run.info.run_id}")

            mlflow.log_param('criterion', params['criterion'])
            mlflow.log_param('n_estimators', params['n_estimators'])
            mlflow.log_param('max_depth', params['max_depth'])
            mlflow.log_param('max_features', params['max_features'])
            mlflow.log_param('random_state', params['random_state'])
            mlflow.log_param('bootstrap', params['bootstrap'])
            mlflow.log_param('ccp_alpha', params['ccp_alpha'])
            mlflow.log_param('max_leaf_nodes', params['max_leaf_nodes'])
            mlflow.log_param('min_impurity_decrease', params['min_impurity_decrease'])
            mlflow.log_param('min_samples_leaf', params['min_samples_leaf'])
            mlflow.log_param('min_samples_split', params['min_samples_split'])
            mlflow.log_param('min_weight_fraction_leaf', params['min_weight_fraction_leaf'])

            mlflow.log_metric('R square', metrics['R2'])
            mlflow.log_metric('RMSE', metrics['RMSE'])
            mlflow.log_metric('MAE', metrics['MAE'])
            mlflow.log_metric('MAPE', metrics['MAPE'])

            mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

#mlflow server --host localhost --port 5000 --backend-store-uri file:mlops_projet/src/mlruns
main()
    
import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn

from dotenv import load_dotenv
load_dotenv()
wd = os.getenv("working_directory")

from fonctions_traitement import data_v1, drop_variable_cible
from fonctions_modelisations import dummies

mlflow.set_tracking_uri(uri = f"file:{wd}/src/fonctions/mlruns")

def get_experiment(experiment_name = "House Price Prediction"):
    """
    Récupère l'experiment ID grâce au nom passé en argument
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    return experiment.experiment_id

def get_runs(experiment_id):
    """
    Récupère les runs de l'expérience identifié par son ID
    """
    return mlflow.search_runs(experiment_ids=[experiment_id])

def get_run_by_name(runs, run_name):
    """
    Récupère un run par son nom
    """
    matching_run = None
    for _, run in runs.iterrows():
        if run['tags.mlflow.runName'] == run_name: 
            matching_run = run
    return matching_run

def get_model_by_run(run):
    """
    Retourne le modèle associé au run
    """
    model = None
    if run is not None:
        model_uri = f"runs:/{run.run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Modèle (Run ID: {run.run_id}) chargé avec succès !")
    else:
        raise Exception("Pas de modèle associé au run")

    return model

def predict(year):
    """
    Predit la variable cible du dataset de l'année N en utilisant le modèle de l'année N-1
    """

    def load_dataset(year):
        """
        Charge le dataset associé à l'année
        """
        dataset_path = f"{wd}/data/predict_year/predict_DKHousing_{year}.csv"
        df = pd.read_csv(dataset_path)
        return df
    
    def load_model(run_name):
        """
        Charge le modele associé au run_name
        """
        experiment_id = get_experiment()
        runs = get_runs(experiment_id)
        run = get_run_by_name(runs, run_name)
        return get_model_by_run(run)

    #MODEL
    run_name = f"RFR_DKHousing_1992_{year-1}"
    model = load_model(run_name)
    
    #DATASET
    df_year_n_to_predict = load_dataset(year)
    df_year_n_to_predict_v1 = data_v1(df_year_n_to_predict)
    X_year_n_to_predict_v1 = drop_variable_cible(df_year_n_to_predict_v1)
    X_year_n_to_predict_v1_dummies = dummies(X_year_n_to_predict_v1)
    
    #PREDICTION
    Y_prediction_n = model.predict(X_year_n_to_predict_v1_dummies)
    df_prediction = pd.DataFrame(Y_prediction_n, columns=[f'Prediction_{year}'])
    df_prediction = pd.concat([X_year_n_to_predict_v1, df_prediction], axis=1)
    print(df_prediction.head().to_string())
    df_prediction.to_csv(f'{wd}/data/prediction_year/prediction_{year}.csv', index=False)


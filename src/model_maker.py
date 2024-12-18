import os
import sys
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()
wd = os.getenv("working_directory")
sys.path.append(f'{wd}/src/fonctions')
from fonctions.fonctions_traitement import data_v1
from fonctions.fonctions_modelisations import Create_Train_Test, random_forest_regressor, linear_regressor, Score



def RFR_fct(X_train, X_test, y_train, y_test):
    """
    Applique la modelisation pour RandomForestRegressor
    Retourne les paramètres et les métriques de qualités
    """
    params = {'bootstrap': False,
              'criterion': 'squared_error',
              'max_depth': 30,
              'max_features': 'sqrt',
              'min_samples_split': 4,
              'n_estimators': 200}
    
    RFR = random_forest_regressor(params)
    RFR_fit = RFR.fit(X_train, y_train)
    params_RFR = RFR.get_params()
    metrics_RFR = Score(RFR_fit, X_test, y_test)

    dico_RFR = {'model': RFR, 'params':params_RFR, 'metrics': metrics_RFR}

    return dico_RFR

def LR_fct(X_train, X_test, y_train, y_test):
    """
    Applique la modelisation pour LinearRegression
    Retourne les paramètres et les métriques de qualités
    """
    LR = linear_regressor()
    LR_fit = LR.fit(X_train, y_train)
    params_LR = LR.get_params()
    metrics_LR = Score(LR_fit, X_test, y_test)

    dico_LR = {'model': LR, 'params':params_LR, 'metrics': metrics_LR}

    return dico_LR

def training_models(dataset_path):
    """
    Applique les transformations de la fonction data_v1
    df est split en données d'entrainement et de test (80/20)
    Train les 2 modeles LinearRegression et RandomForestRegressor sur les données d'entrainement
    Test sur les jeux de test
    Retourne les métriques de qualités et les valeurs des hyperparamètres utilisés
    """
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    print(dataset_name)
    df = pd.read_parquet(dataset_path)
    data_V1 = data_v1(df)
    X_train, X_test, y_train, y_test = Create_Train_Test(data_V1)
    
    #RandomForestRegressor
    dico_RFR = RFR_fct(X_train, X_test, y_train, y_test)
    #LinearRegressor
    dico_LR = LR_fct(X_train, X_test, y_train, y_test)

    dico_dataset = {f'{dataset_name}_RFR':dico_RFR,
                    f'{dataset_name}_LR':dico_LR}
    
    return dico_dataset

def loop_training():
    """
    Applique la fonction training models sur plusieurs années
    """
    dico_master = {}
    for year in list(range(2010, 2022+1)):
        dataset_name = f'{wd}/data/concat_year/DKHousing_1992_{year}.parquet'
        dico = training_models(dataset_name)
        dico_master.update(dico)
        with open(f'{wd}/data/data_models/tracking_model_{year}.json', 'w', encoding='utf-8') as fichier:
            json.dump(dico, fichier, indent=4, ensure_ascii=False)
        

def main():
    loop_training()
    #print(training_models(f'{wd}/data/concat_year/DKHousing_1992_2021.parquet'))

main()





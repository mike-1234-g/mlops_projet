import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from fonctions_traitement import data_v1
load_dotenv()
wd = os.getenv("working_directory")

def Create_Train_Test(df):
    """
    Créer les dataframes de test et d'entraînement 
    """
    y = df.pop('purchase_price')
    X_non_numeric = df.select_dtypes(exclude=['number'])
    X_numeric = df.select_dtypes(include=['number'])
    X_dummies = pd.get_dummies(X_non_numeric)
    X = pd.concat([X_numeric, X_dummies], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def Grid_Search_RFR(X_train, y_train):
    """
    Compute le RandomForestRegressor en appliquant un GridSearchCV
    """
    param_grid = {
    'criterion' : ["squared_error", "friedman_mse"],
    'n_estimators': [200],
    'max_depth': [20, 30],
    'min_samples_split': [3,4,5],
    'max_features': ['sqrt'],
    'bootstrap': [False]
    }

    random_forest = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(random_forest,param_grid,cv=4,scoring='r2',n_jobs=6)
    grid_search_fit = grid_search.fit(X_train, y_train)
    best_model = grid_search_fit.best_estimator_

    return grid_search, best_model

def Grid_Search_LR(X_train, y_train):
    """
    Compute le RandomForestRegressor en appliquant un GridSearchCV
    """
    param_grid = {
    'fit_intercept': [True, False]
    }

    linear_regressor = LinearRegression()
    grid_search = GridSearchCV(linear_regressor,param_grid,cv=5,scoring='r2',n_jobs=5)
    grid_search_fit = grid_search.fit(X_train, y_train)
    best_model = grid_search_fit.best_estimator_

    return grid_search, best_model

def random_forest_regressor(params=None):
    """
    Compute Random Forest Regressor et retourne le modèle entraîné
    """
    if params is None:
        params = {}
    model = RandomForestRegressor(random_state=42, **params)
    
    return model

def linear_regressor():
    """
    Compute Linear Regressor et retourne le modèle entrainé 
    """
    model = LinearRegression(fit_intercept=True)

    return model

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

def Score(model, X_test, y_test):
    """
    Calcul les métriques de qualité des modeles de régression
    R², RMSE, MAE, MAPE
    """
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred) 
    mape = MAPE(y_test, y_pred)

    dico_metrics={"R2":r2, "RMSE":float(rmse), "MAE": mae, "MAPE": float(mape)}
    
    return dico_metrics 


def main():
    """
    """
    df = pd.read_csv(f'{wd}/data/train_concat_year/DKHousing_1992_2022.csv')
    df_v1 = data_v1(df)
    X_train, X_test, y_train, y_test = Create_Train_Test(df_v1)
    GS_RFR, best_RFR = Grid_Search_RFR(X_train, y_train)
    dico_metrics = Score(best_RFR, X_test, y_test)
    print(best_RFR.get_params())
    print(dico_metrics)




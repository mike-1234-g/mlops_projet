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

def mod_random_forest(df):
    X = df.drop(columns=["purchase_price"])
    y = df['purchase_price']

    # Création des dummies pour les variables catégorielles
    X_non_numeric = X.select_dtypes(exclude=['number'])
    X_numeric = X.select_dtypes(include=['number'])
    X_dummies = pd.get_dummies(X_non_numeric)
    X = pd.concat([X_numeric, X_dummies], axis=1)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Grille de paramètres pour GridSearchCV
    param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
    }

    random_forest = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(random_forest,param_grid,cv=5,scoring='r2',n_jobs=1
                               ) # verbose => niveau d'affichage des messages pendant l'execution

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    # score 
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error : {mse}")
    print(f"R² Score :{r2}")


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
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['log2', 'sqrt'],
    'bootstrap': [True, False]
    }

    random_forest = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(random_forest,param_grid,cv=5,scoring='r2',n_jobs=7)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    results = pd.DataFrame(grid_search.cv_results_)

    # Trier les résultats par la meilleure performance (MSE le plus élevé)
    best_results = results.sort_values(by='mean_test_score', ascending=False)

    # Afficher les 10 meilleures combinaisons d'hyperparamètres
    top_10_params = best_results[['params', 'mean_test_score']].head(10)
    print(top_10_params)

    return best_model

def Grid_Search_LR(X_train, y_train):
    """
    Compute le RandomForestRegressor en appliquant un GridSearchCV
    """
    param_grid = {
    'fit_intercept': [True, False]
    }

    linear_regressor = LinearRegression()
    grid_search = GridSearchCV(linear_regressor,param_grid,cv=5,scoring='r2',n_jobs=7)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    return best_model

def random_forest_regressor(X_train, y_train, params=None):
    """
    Compute Random Forest Regressor et retourne le modèle entraîné
    """
    if params is None:
        params = {}
    model = RandomForestRegressor(random_state=42, **params).fit(X_train, y_train)
    
    return model

def linear_regressor(X_train, y_train, params=None):
    """
    Compute Linear Regressor et retourne le modèle entrainé 
    """
    if params is None:
        params = {}
    model = LinearRegression(**params).fit(X_train, y_train)

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
    df = pd.read_csv(f'{wd}/data/data_year/DKHousing_1999.csv')
    df_v1 = data_v1(df)
    X_train, X_test, y_train, y_test = Create_Train_Test(df_v1)
    best_RFR = Grid_Search_RFR(X_train, y_train)
    dico_metrics = Score(best_RFR, X_test, y_test)
    print(dico_metrics)


main()
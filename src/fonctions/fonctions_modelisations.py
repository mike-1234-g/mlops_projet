import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
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

def random_forest_regressor(X_train, y_train, params=None):
    """
    Compute Random Forest Regressor et retourne le modèle entraîné
    """
    if params is None:
        params = {}
    model = RandomForestRegressor(random_state=42, **params).fit(X_train, y_train)
    
    return model

def main():
    """
    """
    df = pd.read_csv(f'{wd}/data/data_year/DKHousing_1999.csv')
    df_v1 = data_v1(df)
    X_train, X_test, y_train, y_test = Create_Train_Test(df_v1)
    params = {'max_depth': 10}
    RFR_fit = random_forest_regressor(X_train, y_train, params=params)
    print(RFR_fit.predict(X_test))
    
main()
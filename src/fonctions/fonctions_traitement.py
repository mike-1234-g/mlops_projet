import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
# variables d'environnement
import os
from dotenv import load_dotenv
load_dotenv()
wd = os.getenv("working_directory")

def data_v1(bdd):
    """
    Transformation du df au format data_v1
    """
    df = bdd.copy()
    df.dropna(inplace=True)
    df["Mois"] = df["date"].astype(str).str[5:7]
    df['Mois'] = pd.to_numeric(df['Mois'], errors='coerce')
    df["Annee"] = df["date"].astype(str).str[:4]
    df['Annee'] = pd.to_numeric(df['Annee'], errors='coerce')
    df['year_build'] = pd.to_numeric(df['year_build'], errors='coerce')
    df["Quarter"] = df["quarter"].astype(str).str[4:6]
    df["Anciennete"] = df["Annee"]-df["year_build"]

    df_new = df.drop(columns=["house_id","date","quarter","%_change_between_offer_and_purchase","city","address","zip_code","sqm_price"])
    
    return df_new

def drop_variable_cible(df):
    """
    Drop la variable cible purchase_price de df pour préparer les prédictions
    """
    df.drop('purchase_price', axis=1, inplace=True)
    return df

### Fonctions d'analyse de la base de données

#données manquantes

def pct_mq(df):
    taille_bdd = len(df)
    nb_lignes_avec_na = df.isna().any(axis=1).sum()
    pct_na = round(nb_lignes_avec_na/taille_bdd*100,2)
    print(f"Pct avec données manquantes : {pct_na}%")

    # données manquantes représentent 0.08% de la database

def pct_mq_col(df):
    taille_bdd = len(df)
    nb_na = df.isna().sum()
    print(nb_na/taille_bdd*100)

    # données mq présentes dans les colonnes city, df_ann_inf_rate et yield_on_mortgage 

def vis_manquantes(df):
    bdd_na_vis = df[df.isnull().any(axis=1)]
    print(bdd_na_vis.head(10))

# données abérrantes

## méthode z-score

def detect_outliers_zscore(df, threshold=3):
    outlier_info = {}

    # Sélectionner uniquement les colonnes numériques
    numeric_columns = df.select_dtypes(include=[float, int]).columns

    # Calculer les Z-Scores pour chaque colonne numérique
    for column in numeric_columns:
        z_scores = zscore(df[column])
        outliers = np.abs(z_scores) > threshold
        
        # Ajouter les résultats au dictionnaire
        outlier_info[column] = {
            'z_scores': z_scores,
            'outliers': outliers
        }
        
        print(f"{column}:")
        #print(f"  Z-scores: {z_scores}")
        print(f"  Outliers: {outliers.sum()}")
        print("")
        # df.drop(outliers.loc[outliers==True].index, inplace=True)

    return outlier_info

# regroupements

def distinct_par_var(df):
    for colonne in df.columns:
        if colonne in ["house_type","sales_type","area","region"]: # pas mis city pcq plus de 800 modalités
            valeurs_uniques = df[colonne].unique()
            print(f"Modalités '{colonne}': {valeurs_uniques}")
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

def col_modifs(df):
    df["Annee"] = df["date"].astype(str).str[:4]
    df["Annee"] = df["Annee"].astype(object)
    df["Quarter"] = df["quarter"].astype(str).str[4:6]
    #df["Anciennete"] = df["Annee"]-df["year_build"]

    df_new = df.drop(columns=["house_id","date","quarter","%_change_between_offer_and_purchase","address"])
    return df_new

# données manquantes

def pct_mq(df):
    taille_bdd = len(df)
    nb_lignes_avec_na = df.isna().any(axis=1).sum()
    pct_na = round(nb_lignes_avec_na/taille_bdd*100,2)
    print(f"Pct avec données manquantes : {pct_na}%")

    # données manquantes représentent 0.09% de la database

def pct_mq_col(df):
    taille_bdd = len(df)
    nb_na = df.isna().sum()
    print(nb_na/taille_bdd*100)

    # données mq présentes dans les colonnes city, df_ann_inf_rate et yield_on_mortgage 

def mq_heatmap(df):
    sns.heatmap(df.isnull())
    plt.show()
    # marche pas

def vis_manquantes(df):
    bdd_na_vis = df[df.isnull().any(axis=1)]
    print(bdd_na_vis.head(10))

def supp_mq(df):
    df.dropna(inplace=True)
    df.to_csv(f"{wd}/data/versions_data/data_sans_na.csv", index=False)

# données abérrantes

## méthode IQR

def box_plot(df): # pk ca marche pas ?
    df.plot(kind='box', subplots=True, figsize=(22, 5))
    plt.show()

def detect_outliers_all_numeric(df): # détermination des bornes
        outlier_bounds = {}

    # Sélectionner uniquement les colonnes numériques
        numeric_columns = df.select_dtypes(include=[float, int]).columns

    # Calculer les limites pour chaque colonne numérique
        for column in numeric_columns:
            Q3 = df[column].quantile(0.75)
            Q1 = df[column].quantile(0.25)
            IQR = Q3 - Q1
            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)

            lower_outliers = (df[column] < lower_bound).sum()
            upper_outliers = (df[column] > upper_bound).sum()
            total_outliers = lower_outliers + upper_outliers
        
        # Ajouter les résultats au dictionnaire
            outlier_bounds[column] = {'lower_bound': lower_bound, 'upper_bound': upper_bound, 'total_outliers': total_outliers}
        
            print(f"{column}:")
            print(f"  lower_bound = {lower_bound}")
            print(f"  upper_bound = {upper_bound}")
            print(f"  Total outliers = {total_outliers}")
            print("")

        return outlier_bounds

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from pandas.plotting import table

# variables d'environnement
import os
from dotenv import load_dotenv
load_dotenv()
wd = os.getenv("working_directory")

bdd = pd.read_parquet(f"{wd}/data/versions_data/data_v2.parquet")

# histogramme des variables qualitatives

def distinct_par_var(df):
    for colonne in df.columns:
        if colonne in ["house_type","sales_type","area","region"]: # pas mis city pcq plus de 800 modalités
            valeurs_uniques = df[colonne].unique()
            print(f"Modalités '{colonne}': {valeurs_uniques}")

def stats_graphs(df): # pour toute les variables quali de la base
    for colonne in df:
        if colonne in ["house_type","sales_type","area","region"]:
            modalites_workclass = df[colonne].value_counts()
            modalites_workclass = modalites_workclass/len(df)*100
            modalites_workclass = modalites_workclass.rename('number').reset_index()
            modalites_workclass.columns = ['Modalite', 'number']  # Renommer les colonnes

            plt.figure(figsize=(10,6))  # Taille du graphique
            plt.bar(modalites_workclass['Modalite'], modalites_workclass['number'], color='skyblue', edgecolor='black')
            plt.xticks(rotation=45, ha='right')  # 'ha' ajuste l'alignement horizontal ('right' pour aligner à droite)
            plt.title(colonne)
            plt.savefig(f"../stats/histogram_stats/{colonne}.pdf")

stats_graphs(bdd)

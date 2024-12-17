import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from pandas.plotting import table

bdd = pd.read_csv("../Datas/bases/DKHousingPricesSample100k.csv")

# histogramme des variables qualitatives

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

# Faire des graphiques (nuages de points) entre les variables explicatives quantitatives et la variable expliquée pour les regroupements

bdd = pd.read_csv("../Datas/bases/DKHousingPricesSample100k.csv")

def scatter_plot(df):
    for colonne in df:
        if colonne not in ["mettre les quantis"]:
            plt.figure(figsize=(8, 6))  
            plt.scatter(bdd['sqm'], bdd[colonne], color='blue', marker='o')
            plt.xlabel('sqm') 
            plt.ylabel(colonne)
            plt.title(f'Scatter Plot between sqm and {colonne}') 
            plt.grid(True) 
            plt.show()

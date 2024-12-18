import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
load_dotenv()
wd = os.getenv("working_directory")

def load_parquet():
    """
    Charge le fichier DKHousePrices.parquet
    """
    df = pd.read_parquet(f'{wd}/data/bases/DKHousingPrices.parquet')
    return df

def dico_generate_dataframe_year(df):
    """
    Génère un dico year:df_year pour chaque année de 1992 à 2024
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df_per_year = {year: df_year.drop(columns=['year']) for year, df_year in df.groupby('year')}
    return df_per_year

def load_to_csv(dico):
    """
    Charge chaque dataframe en csv
    """
    for year, df_year in dico.items():
        df_year.to_csv(f'{wd}/data/data_year/DKHousing_{year}.csv', index=False)

def main():
    """
    """
    df = load_parquet()
    dico_year = dico_generate_dataframe_year(df)
    del df
    load_to_csv(dico_year)

main()

import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
wd = os.getenv("working_directory")

def load_parquet():
    """
    Charge le fichier DKHousePrices.parquet
    """
    df = pd.read_parquet(f'{wd}/data/bases/DKHousingPrices.parquet')
    return df

def load_csv():
    """
    Charge le fichier DKHousePrices.csv
    """
    df = pd.read_csv(f'{wd}/data/bases/DKHousingPricesSample100k.csv')
    return df

def dico_generate_dataframe_year(df):
    """
    Génère un dico year:df_year pour chaque année de 1992 à 2024
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df_per_year = {year: df_year.drop(columns=['year']) for year, df_year in df.groupby('year')}
    return df_per_year

def load_train_by_year():
    """
    Charge les dataframes d'entrainement par année en csv dans /data/train_data_year
    """
    def load_train_by_year_to_csv(dico):
        """
        Charge chaque dataframe (du dico de la fonction dico_generate_dataframe_year) en csv
        """
        for year, df_year in dico.items():
            df_year.to_csv(f'{wd}/data/train_data_year/DKHousing_{year}.csv', index=False)

    df = load_csv()
    dico_year = dico_generate_dataframe_year(df)
    load_train_by_year_to_csv(dico_year)

def load_predict_for_year(years):
    """
    Charge les dataframes qui vont servier aux prédictions pour les années passées en argument
    """
    def keep_years(dico):
        """
        Supprime les clés qui ne sont pas dans la liste years
        """
        dico_keep_years = {cle: dico[cle] for cle in years if cle in dico}
        return dico_keep_years
    
    def load_predict_to_csv(dico):
        """
        Charge les dataframes du dico en csv dans data/predict_year
        """
        for year, df_year in dico.items():
            df_year.to_csv(f'{wd}/data/predict_year/predict_DKHousing_{year}.csv', index=False)

    df = load_parquet()
    dico = dico_generate_dataframe_year(df)
    dico_keep_years = keep_years(dico)
    load_predict_to_csv(dico_keep_years)


def load_train_concat_year():
    """
    Charge les fichiers concat en csv
    """

    for year in list(range(1992, 2022 + 1)):
        df_year_name = f'{wd}/data/train_data_year/DKHousing_{year}.csv'
        if year == 1992:
            df_master = pd.read_csv(df_year_name)
            
        else : 
            df_master = pd.read_csv(f'{wd}/data/train_concat_year/DKHousing_1992_{year-1}.csv')
            df = pd.read_csv(df_year_name)
            df_master = pd.concat((df_master, df),axis=0)
        
        df_master.to_csv(f'{wd}/data/train_concat_year/DKHousing_1992_{year}.csv', index=False)

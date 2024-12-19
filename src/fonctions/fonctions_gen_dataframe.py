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

def load_concat_year():
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

def load_to_csv(dico):
    """
    Charge chaque dataframe en csv
    """
    for year, df_year in dico.items():
        df_year.to_csv(f'{wd}/data/train_data_year/DKHousing_{year}.csv', index=False)

def GEN_DF_BY_YEAR():
    """
    """
    df = load_csv()
    dico_year = dico_generate_dataframe_year(df)
    del df
    load_to_csv(dico_year)

def main():
    load_concat_year()
main()

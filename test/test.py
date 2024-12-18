import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
wd = os.getenv("working_directory")

df = pd.read_parquet(f'{wd}/data/concat_year/DKHousing_1992_1996.parquet')
print(df.shape)
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from fonctions.fonctions_traitement import data_v1
from fonctions.fonctions_modelisations import Create_Train_Test, Grid_Search_RFR, Grid_Search_LR, Score
load_dotenv()
wd = os.getenv("working_directory")


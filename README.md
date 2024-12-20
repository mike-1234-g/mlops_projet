# MLOps Project: Predicting House Prices

## Objective
The goal of this project is to predict house purchase prices for the years 2021 to 2023 based on a database containing house characteristics and their purchase prices between 1992 and 2022.
With this project, we are simulating a production environnement by updating our models based on the newer data.
Starting with the data between 1992 and 2020, we are training a model that will predict houses prices for 2021. We repeat the operations to train a new model on data between 1992 and 2021 to predict 2022 and so on for the prediction of year 2023.
We are using mlflow to track and store our models over the year. It also allow us to get back the exact models we trained to make the predictions.

## Installation and Usage
1. **Set Up Environment**: create a *.env* file where you instantiate a variable called "working_directory" that specifies the path to the mlops project
for example : *working_directory = "<absolute/path/to/the/project>/mlops_projet"*
2. **Create a virtual environment and install dependencies**: 
   ```bash
   # create virutal environment
   python -m venv <venv_name>
   # activate virtual environment
   ## on mac
   source venv_name/bin/activate
   ## on windows
   venv_name\Scripts\activate
   # install the dependencies
   pip install -r requirements.txt
   ```
3. **GIT LFS**: Git LFS (Large File Storage) is used to efficiently manage large files like `.plk` files, which are binary files often generated in machine learning workflows. In our case, we are using it because MLflow generates `.plk` files which are too big to handle by a classic git repository. This step is mandatory to make the whole workflows works. Indeed, the `.plk` files defines models and if you dont have those in your project, you'll not be able to predict the price over the years. Make sure you have those files!!!
To do it through mac (Homebrew):
   ```bash
   brew install git-lfs
   ```
For both windows and mac users:
   ```bash
   git lfs install
   ```
Once these steps are completed, you should be able to see models `.plk` in the mlrun folder. For example : 
"<absolute/path/to/the/project>/mlops_projet/src/fonctions/mlruns/711528715738709649/88133a887d4348efbf5b09ffcde7841d/artifacts/model/model.pkl"
If you still don't see them after pulling the repo, you might want to delete the project and git clone it back.

4. **Run the Pipeline**:
   - Execute the `DEMO` notebook to run the full workflow or use individual scripts in `src/fonctions` for specific steps.

## Project Structure
The Git repository is organized into the following main directories:

### 1. `datas`
This directory contains the datasets used and processed during the study. It includes the following subdirectories:
- **`bases`**: Contains the raw databases provided for the study. 
  - `DKHousingPrices.parquet`: 1,5 million rows (used for predictions)
  - `DKHousingPricesSample100k.csv`: 100K rows (used for training)
- **`predict_year`**: Contains the datasets for the years 2021 to 2023, where the predictions will be done.
- **`prediction_year`**: Contains the datasets for the years 2021 to 2023 with the predictions.
- **`train_concat_year`**: Contains the datasets used to train the models.
- **`train_data_year`**: Contains the yearly data sampled from `DKHousingPricesSample100k.csv`.

### 2. `src`
This directory contains all the Python scripts and notebooks for the study:
#### - **Functions Subdirectory**: Includes all Python functions categorized as follows:
  - **`fonctions_traitement`**: Processes data (e.g., handling missing or outlier values).
  - **`fonctions_stats`**: Performs statistical analyses to identify potential groupings. Results are saved in `stats/histogramme_stats`.
  - **`fonctions_modelisations`**: Includes functions for:
    - Preparing data for modeling (e.g., creating dummy variables).
    - Defining parameters for GridSearchCV.
    - Training models using `RandomForestRegressor`.
    - Evaluating model performance using metrics like R2, RMSE, MAE, and MAPE.
  - **`fonctions_gen_dataframe`**: Generates training datasets and the dataframes used for predictions.
##### MLflow
  - **`mlruns`**: Directory used to save models that are tracked using *mlflow tracking*.
  - **`fonctions_mlflow_tracking`**: Train models and saves them in mlruns thanks to the mlflow module.
  - **`fonctions_mlflow_prediction`**: With the models that are trained through *mlflow_tracking*, predictions are made for the years 2021 to 2023 (cf. data/predict_year).
- **`DEMO` Notebook**: Demonstrates the execution of all functions.

### 3. `stats`
Contains visualizations and analyses of qualitative variables from the database, including histograms and other statistical plots.

## Additional Files
- **`.env`**: Stores environment variables. In our case, it stores the project path in a variable called *working_directory*. It is mandatory for every person in the project to start the project by creating their *.env*. 
- **`.gitattributes`**: Configures Git handling for text and binary files, especially using *git lfs*. For this project, it handles the *.pkl* files generated through mlflow tracking. This is done because the *.pkl* are too bulky to be pushed on the classic github repo.
- **`.gitignore`**: Specifies files and directories to exclude from Git tracking (e.g., `.env`).
- **`launch_server.sh`**: A script to launch the server or application associated with the project.

## Key Features
- **Data Preprocessing**:
  - Cleaning raw data.
  - Handling missing and outlier values.
  - Generating processed versions of the dataset.
- **Statistical Analysis**:
  - Provides insights into data characteristics and guides modeling decisions.
- **Modeling**:
  - Trains machine learning models for regression.
  - Evaluates model performance using multiple metrics.
- **Scalability**:
  - Efficient handling of large datasets using parquet format.

# MLOps Project: Predicting House Prices

## Objective
The goal of this project is to predict house purchase prices for the years 2011 to 2020 based on a database containing house characteristics and their purchase prices between 1992 and 2010.

## Project Structure
The Git repository is organized into the following main directories:

### 1. `datas`
This directory contains the datasets used and processed during the study. It includes the following subdirectories:
- **`bases`**: Contains the raw databases provided for the study.
- **`predict_year`**: Contains the datasets with predictions for the years 2011 to 2020.
- **`versions_data`**: Contains the processed versions of the datasets after cleaning and transformation.

### 2. `src`
This directory contains all the Python scripts and notebooks for the study:
- **Functions Subdirectory**: Includes all Python functions categorized as follows:
  - **`fonctions_traitement`**: Loads the raw database, processes data (e.g., handling missing or outlier values), and saves versions in `datas/versions_data`.
  - **`fonctions_stats`**: Performs statistical analyses to identify potential groupings. Results are saved in `stats/histogramme_stats`.
  - **`fonctions_modelisations`**: Includes functions for:
    - Preparing data for modeling (e.g., creating dummy variables).
    - Defining parameters for GridSearchCV.
    - Training models using `RandomForestRegressor` and `DecisionTreeRegressor`.
    - Evaluating model performance using metrics like R2, RMSE, MAE, and MAPE.
  - **`fonctions_gen_dataframe`**: Generates training and test dataframes.
- **`DEMO` Notebook**: Demonstrates the execution of all functions.

### 3. `stats`
Contains visualizations and analyses of qualitative variables from the database, including histograms and other statistical plots.

### 4. `test`
This directory includes:
- Python scripts to test whether environment variables are properly set.
- Code to verify that large parquet files (over a million rows) can be successfully loaded.

## Additional Files
- **`.env`**: Stores environment variables.
- **`.gitattributes`**: Configures Git handling for text and binary files. This ensures consistent line endings across systems and defines attributes for large file support if required.
- **`.gitignore`**: Specifies files and directories to exclude from Git tracking (e.g., `.env`).
- **`launch_server.sh`**: A script to launch the server or application associated with the project.

## Installation and Usage
1. **Install Dependencies**: Ensure all necessary Python libraries are installed using:
   ```bash
   pip install -r requirements.txt
   ```
2. **Set Up Environment**:
   - Create a `.env` file to define variables like database paths, API keys, etc.
3. **Run the Pipeline**:
   - Execute the `DEMO` notebook to run the full workflow or use individual scripts in `src` for specific steps.

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

#!/bin/bash

# Charger les variables d'environnement à partir du fichier .env
source .env
mlflow server --host localhost --port 5000 --backend-store-uri file:$working_directory/src/fonctions/mlruns

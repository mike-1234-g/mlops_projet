import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


mlflow.set_experiment("Premier Cycle")

mlflow.set_tracking_uri(uri = 'https://localhost:5000')

with mlflow.start_run():
    pass
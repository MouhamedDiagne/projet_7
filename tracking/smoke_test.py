"""Run an MLflow smoke test locally or from a Kaggle notebook.

Example:
    %run /kaggle/working/smoke_test.py

Or copy this folder to the Kaggle working directory and run:
    !python /kaggle/working/tracking/smoke_test.py
"""

from mlflow_tracking import run_smoke_test


if __name__ == "__main__":
    result = run_smoke_test()
    print("MLflow smoke test completed.")
    print(f"Backend: {result.backend}")
    print(f"Experiment ID: {result.experiment_id}")
    print(f"Run ID: {result.run_id}")
    print(f"Verified artifact: {result.artifact_path}")

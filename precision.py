import numpy as np
import os
from utils import LinearRegressionModel, MinMaxScaler

def main():
  data_file_path = "data.csv"
  model_file_path = "model_0.json"

  model = LinearRegressionModel()
  data = np.loadtxt(data_file_path, delimiter=',', skiprows=1)
  data_X = data[:, 0]
  data_y = data[:, 1]
  X_min_max_scaler = MinMaxScaler(data_X)
  y_min_max_scaler = MinMaxScaler(data_y)

  X = X_min_max_scaler.normalize(data_X)
  y = y_min_max_scaler.normalize(data_y)

  # load the saved model if it exist
  if os.path.isfile(model_file_path):
    model.load(model_file_path)

  mean_squared_error = model.mean_squared_error(X, y)
  r2_score = model.r2_score(X, y)
  print(f"Mean Squared Error (MSE) : {mean_squared_error}")
  print(f"R-squared = {r2_score}")

if __name__ == "__main__":
  main()

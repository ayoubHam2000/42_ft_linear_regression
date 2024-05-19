import numpy as np
import os
from utils import LinearRegressionModel, MinMaxScaler

def read_mileage_value():
  while True:
    try:
      mileage = int(input("Enter the car's mileage: 🚗: "))
      assert(mileage >= 0), "The mileage should be greater than or equal to 0"
      return mileage
    except Exception as e:
      print(f"An error occurred: {e}")

def main():
  data_file_path = "data.csv"
  model_file_path = "model_0.json"

  model = LinearRegressionModel()
  data = np.loadtxt(data_file_path, delimiter=',', skiprows=1)
  data_X = data[:, 0]
  data_y = data[:, 1]
  X_min_max_scaler = MinMaxScaler(data_X)
  y_min_max_scaler = MinMaxScaler(data_y)

  # load the saved model if it exist
  if os.path.isfile(model_file_path):
    model.load(model_file_path)

  mileage = read_mileage_value()
  n_mileage = X_min_max_scaler.normalize(mileage)
  pred = model.hypothesis(n_mileage)
  real_pred = y_min_max_scaler.inverse_normalize(pred)
  print(f"The estimation of the price of a car based on its mileage 🚗📈💰 is : {real_pred}$")

if __name__ == "__main__":
  main()

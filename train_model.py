import numpy as np
from utils import LinearRegressionModel, MinMaxScaler
import matplotlib.pyplot as plt

def plot_data(X, y, pred, epochs, train_cost):
  plt.figure(figsize=(14, 6))
  plt.subplot(1, 2, 1)
  plt.title('Price over mileage')
  plt.scatter(X, y, c="b", s=8, label="data")
  plt.xlabel('mileage')
  plt.ylabel('price')
  plt.plot(X, pred, color='g', label="predictions")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.title("Training progress")
  plt.plot(epochs, train_cost, label="training_cost")
  plt.legend()
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.show()


def main():
  data_file_path = "data.csv"
  model_file_path = "model_0.json"

  model = LinearRegressionModel()
  data = np.loadtxt(data_file_path, delimiter=',', skiprows=1)
  data_X = data[:, 0]
  data_y = data[:, 1]
  X_min_max_scaler = MinMaxScaler(data_X)
  y_min_max_scaler = MinMaxScaler(data_y)
  
  X_train = X_min_max_scaler.normalize(data_X)
  y_train = y_min_max_scaler.normalize(data_y)
  lr = 0.1

  arr_train_cost = []
  arr_epoch = []
  epochs = 10000
  for epoch in range(epochs):
    model.gradientDescent(X_train, y_train, lr)
    if epoch % 1000 == 0:
      train_cost = model.cost(X_train, y_train)
      arr_train_cost.append(train_cost)
      arr_epoch.append(epoch)
      print(f"Epoch: {epoch} | Train loss: {train_cost}")

  model.save(model_file_path)
  
  pred = model(X_train)
  pred = y_min_max_scaler.inverse_normalize(pred)

  plot_data(data_X, data_y, pred, arr_epoch, arr_train_cost)

if __name__ == "__main__":
  main()

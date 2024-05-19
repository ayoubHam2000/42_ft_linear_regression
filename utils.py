import numpy as np
import json

class MinMaxScaler():
  def __init__(self, data):
    data = np.asarray(data)
    self.d_max = np.max(data)
    self.d_min = np.min(data)
    self.d_mean = np.mean(data)
  
  def normalize(self, data):
    return ((data - self.d_mean) / (self.d_max - self.d_min))

  def inverse_normalize(self, data):
    return (data * (self.d_max - self.d_min) + self.d_mean)


class LinearRegressionModel():
  def __init__(self):
    self.theta_0 = 0
    self.theta_1 = 0

  def hypothesis(self, x):
    return x * self.theta_1 + self.theta_0

  def mean_squared_error(self, X, y):
    return self.cost(X, y)
  
  def r2_score(self, X, y):
    y_pred = self.hypothesis(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

  def cost(self, X, y):
    m = y.shape[0]
    temp = (self.hypothesis(X) - y) ** 2
    return (1 / (2 * m)) * (np.sum(temp))

  def gradientDescent(self, X, y, lr):
    m = y.shape[0]
    y_pred = self.hypothesis(X)
    temp = y_pred - y
    self.theta_0 = self.theta_0 - lr * np.sum(temp) / m
    self.theta_1  = self.theta_1 - lr * np.dot(temp, X) / m
  
  def get_pramas(self):
    return (self.theta_0, self.theta_1)
  
  def __call__(self, X):
    return self.hypothesis(X)
  
  def load(self, path):
    with open(path, 'r') as file:
      data = json.load(file)
      self.theta_0 = data["theta_0"]
      self.theta_1 = data["theta_1"]
  
  def save(self, path):
    data = {
      "theta_0" : self.theta_0,
      "theta_1" : self.theta_1
    }
    with open(path, 'w') as file:
      json.dump(data, file)
  
  

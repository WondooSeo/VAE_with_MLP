# # univariate mlp example
# import numpy as np
# from numpy import array
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# # define dataset
# X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
# y = array([40, 50, 60, 70])
# print(np.shape(X))
# print(np.shape(y))
# # define model
# model = Sequential()
# model.add(Dense(100, activation='relu', input_dim=3))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# # fit model
# model.fit(X, y, epochs=2000, verbose=0)
# # demonstrate prediction
# x_input = array([50, 60, 70])
# x_input = x_input.reshape((1, 3))
# yhat = model.predict(x_input, verbose=1)
# print(yhat)


import numpy as np

z_sample = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

print(np.shape(z_sample))
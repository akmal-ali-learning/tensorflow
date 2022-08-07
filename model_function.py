# Use tensorflow to learn a 1-1 function

import math
import sys
import tensorflow as tf
import numpy as np

def test_fn(x):
  return x*x + 2*x + 1


def main():
  x_data = range(1, 1000)
  y_data = []

  for x in x_data:
    y_data.append( test_fn(x) )

  x_data = np.asarray(x_data).astype('float32').reshape(-1, 1)
  y_data = np.asarray(y_data).astype('float32').reshape(-1, 1)

  sample = 700
  idx = np.random.permutation(x_data.shape[0])  # can also use random.shuffle
  train_idx, test_idx = idx[:sample], idx[sample:]
  x_train, x_test, y_train, y_test = x_data[train_idx,:], x_data[test_idx,:], y_data[train_idx,], y_data[test_idx,]


  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1, 1)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1,)
  ])

  print("Compile model")
  model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError(), metrics=['accuracy'])
  print("Train model")
  model.fit(x_train, y_train, epochs=1000)
  print("Evaluate model")
  model.evaluate(x_test,  y_test, verbose=2)

  print("Use model")
  # probability_model = tf.keras.Sequential([ model, tf.keras.layers.Softmax() ])
  # probability_model(x_test[:5])
  prediction = model.predict(x_test[:5])
  print(x_test[:5])
  print(prediction)


if __name__ == '__main__':
  sys.exit(main())  # next section explains the use of sys.exit
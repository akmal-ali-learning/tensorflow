# Use tensorflow to learn doubling function.

import tensorflow as tf
import numpy as np
print("TensorFlow version:", tf.__version__)


x_train = []
y_train = []
x_test  = []
y_test  = [] 

print("Create training data ")

for i in range(1,1000):
    x_train.append(i * 1.0)
    y_train.append(2*i * 1.0)
    x_test.append(2*i * 1.0)
    y_test.append(4*i * 1.0)

x_train = np.asarray(x_train).astype('float32').reshape(-1, 1)
y_train = np.asarray(y_train).astype('float32').reshape(-1, 1)
x_test  = np.asarray(x_test).astype('float32').reshape(-1, 1)
y_test  = np.asarray(y_test).astype('float32').reshape(-1, 1)


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1, input_shape=(1, 1)),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(1,)
])

print("Compile model")
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
print("Train model")
model.fit(x_train, y_train, epochs=250)
print("Evaluate model")
model.evaluate(x_test,  y_test, verbose=2)

print("Use model")
# probability_model = tf.keras.Sequential([ model, tf.keras.layers.Softmax() ])
# probability_model(x_test[:5])
prediction = model.predict(x_test[:5])
print(x_test[:5])
print(prediction)
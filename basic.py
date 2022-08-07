import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Load DATA
mnist = tf.keras.datasets.mnist
print("Load data")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert data for floating point
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build machine-learning model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# For each example, the model returns a vector of logits (log-odd scores). 1 for each class.
predictions = model(x_train[:1]).numpy()

# Convert logits to probabilities for each class.
tf.nn.softmax(predictions).numpy()

# Define a loss function for training - Takes a vector of logits and a True index and returns scalar loss for each example.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.
# This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.
loss_fn(y_train[:1], predictions).numpy()


# Before you start training, configure and compile the model using Keras Model.compile. 
# Set the optimizer class to adam, set the loss to the loss_fn function you defined earlier, 
# and specify a metric to be evaluated for the model by setting the metrics parameter to accuracy.
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Train and evaluate your model

## Train the model
model.fit(x_train, y_train, epochs=5)

## Evaluate the model's performance 
print("Evaluate model's performance")
model.evaluate(x_test,  y_test, verbose=2)

## Probability Model
probability_model = tf.keras.Sequential([ model, tf.keras.layers.Softmax() ])
probability_model(x_test[:5])
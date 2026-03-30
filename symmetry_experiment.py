import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

# ANN with sigmoid in hidden layer
model = Sequential()
model.add(Dense(2, activation='sigmoid', input_dim=2))
model.add(Dense(1, activation='sigmoid'))

# Initialize all weights and biases to zero
weights = model.get_weights()
for i in range(len(weights)):
    weights[i] = np.zeros_like(weights[i])
model.set_weights(weights)

# model compilation
model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])

# Train
model.fit(X, y, epochs=100, batch_size=1, verbose=0)

# Print learned weights
w = model.get_weights()
print("Hidden weights:\n", w[0])
print("Hidden biases:\n", w[1])
print("Output weights:\n", w[2])
print("Output bias:\n", w[3])

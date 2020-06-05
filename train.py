import numpy
from tensorflow import keras

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = numpy.array([-1.0, 11.0], dtype=float)
ys = numpy.array([10.0, 15.0], dtype=float)

model.fit(xs, ys, epochs=1000)
print(model.predict([3.0]))

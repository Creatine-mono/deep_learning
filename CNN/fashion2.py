import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(trainX, trainY),(testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

#print(trainX.shape)
#print(trainY.shape)
#print(trainX[0])

# plt.imshow(trainX[1])
# plt.gray()
# plt.colorbar()
# plt.show()

trianX = trainX / 255.0
testX = testX / 255.0


trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

class_names = ["T-shirt/top",  # index 0
        "Trouser",      # index 1
        "Pullover",     # index 2 
        "Dress",        # index 3 
        "Coat",         # index 4
        "Sandal",       # index 5
        "Shirt",        # index 6 
        "Sneaker",      # index 7 
        "Bag",          # index 8 
        "Ankle boot"]   # index 9

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D( 32, (3,3) , padding = "same", activation='relu', input_shape=(28,28,1) ),
  tf.keras.layers.MaxPooling2D( (2,2) ),
  tf.keras.layers.Conv2D( 32, (3,3) , padding = "same", activation='relu', input_shape=(28,28,1) ),
  tf.keras.layers.MaxPooling2D( (2,2) ),
  # tf.keras.layers.Dense(128, input_shape=(28,28) ,activation='relu'),
  tf.keras.layers.Flatten(),
  # tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax'),
])

model.summary()

# exit()


model.compile( loss='sparse_categorical_crossentropy',
               optimizer='adam', metrics=['accuracy'] )

model.fit(trainX, trainY, validation_data = (testX, testY), epochs=5)

# score = model.evaluate(testX, testY)
# print(score)
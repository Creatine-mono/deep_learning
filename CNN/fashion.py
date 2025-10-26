import tensorflow as tf
import matplotlib.pyplot as plt

(trainX, trainY),(testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

#print(trainX.shape)
#print(trainY.shape)
#print(trainX[0])

# plt.imshow(trainX[1])
# plt.gray()
# plt.colorbar()
# plt.show()

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
  tf.keras.layers.Dense(128, input_shape=(28,28) ,activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10, activation='softmax'),
])

# model.summary()

# exit()


model.compile( loss='sparse_categorical_crossentropy',
               optimizer='adam', metrics=['accuracy'] )

model.fit(trainX, trainY, epochs=5)




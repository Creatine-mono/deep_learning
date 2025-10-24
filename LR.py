import tensorflow as tf

키 = [150,160,170,180]
신발 = [152,162,172,182]

a = tf.Variable(0.1)
b = tf.Variable(0.5)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

for i in range(1000):

  with tf.GradientTape() as tape:
    예측신발 = 키 * a + b
    loss = (예측신발 - 신발)**2
    loss = tf.reduce_mean(loss)

  gradient = tape.gradient(loss, [a,b])
  opt.apply_gradients([[gradient[0], a],[gradient[1], b]])
  print(a.numpy(), b.numpy())

print(예측신발)
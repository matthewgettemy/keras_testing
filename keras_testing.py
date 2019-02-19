import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# neural networks try to minimize loss, not maximize accuracy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# netural networks are great at fitting, but we want to make sure
# that they don't overfit - you want the model to work generally...
model.fit(x_train, y_train, epochs=3)

validation_loss, validation_accuracy = model.evaluate(x_test, y_test)
print(validation_loss, validation_accuracy)

model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')

predictions = new_model.predict([x_test])
print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()

#plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()
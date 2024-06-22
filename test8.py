from turtle import shape
import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.list_physical_devices('GPU')
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()

print(train_labels.max())
number_of_classes = train_labels.max() + 1
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(number_of_classes)
])
image_height = 28
image_width = 28

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    verbose=True,
    validation_data=(valid_images, valid_labels)
)
model.predict(train_images[0:10])
data_idx = 8675 # The question number to study with. Feel free to change up to 59999.

# plt.figure()
# plt.imshow(train_images[data_idx], cmap='gray')
# plt.colorbar()
# plt.grid(False)
# plt.show()
print(model.predict(train_images[data_idx:data_idx+1]))
x_values = range(number_of_classes)
plt.figure()
plt.bar(x_values, model.predict(train_images[data_idx:data_idx+1]).flatten())
plt.xticks(range(10))
plt.show()

print("correct answer:", train_labels[data_idx])

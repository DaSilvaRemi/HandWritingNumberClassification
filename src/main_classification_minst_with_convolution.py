import network3
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU
import numpy as np
import matplotlib.pyplot as plt

training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10

net = network3.Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 1, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)

test_image = test_data[0][0].eval().reshape((1, 1, 28, 28))
predictions = net.test_mb_predictions(test_image, batch_size=mini_batch_size)
predicted_labels = np.argmax(predictions, axis=1)

# calculate the accuracy of the model
plt.bar(np.arange(10), predictions)
plt.xticks(np.arange(10))
plt.show()
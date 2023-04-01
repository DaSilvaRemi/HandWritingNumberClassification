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

test_x, test_y = test_data
test_y = test_y.eval()
test_y

test_accuracies = []
for i in range (0, mini_batch_size):
    predictions = net.test_mb_predictions(i)
    predictions = np.array(predictions)
    test_accuracy = []
    
    for j in range(0, len(predictions)):
        test_accuracy.append(test_y[j] == predictions[j])

    test_accuracy = np.mean(test_accuracy)
    test_accuracies.append(test_accuracy * 100)

plt.plot(test_accuracies)
plt.xlabel('Batch')
plt.ylabel('Test Accuracy')
plt.show()
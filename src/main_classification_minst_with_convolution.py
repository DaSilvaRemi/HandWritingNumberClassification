import network2
import numpy as np
import mnist_loader
import matplotlib.pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = Network([
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
net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)

validation_data = list(validation_data)
test_data = list(test_data)

# Test the trained network on a single image
test_image = test_data[0][0]
test_label = test_data[0][1]
prediction = net.feedforward(test_image)
predicted_label = np.argmax(prediction)

# Display the image and the predicted label
plt.imshow(test_image.reshape(28, 28), cmap='gray')
plt.title(f'Predicted label: {predicted_label}')
plt.show()
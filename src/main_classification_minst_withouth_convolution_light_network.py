import network
import numpy as np
import matplotlib.pyplot as plt
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

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
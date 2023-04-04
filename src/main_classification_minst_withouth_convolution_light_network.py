import network2
import numpy as np
import mnist_loader
import matplotlib.pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = network2.Network([784], cost=network2.CrossEntropyCost)
net.SGD(training_data, 
    epochs=30, 
    mini_batch_size=10, 
    eta=0.1,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)

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
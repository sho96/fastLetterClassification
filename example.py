import imgClassifier
import numpy as np
import pickle
from keras.datasets import mnist

#create a model with an input shape of 28x28 and 10 outputs
model = imgClassifier.Classifier((28, 28), 10)

#load mnist data
(trainX, trainY),(testX, testY) = mnist.load_data()

#train a model with input and its label (index of an output neuron)
for img, label in zip(trainX, trainY):
    model.train(img, label)

#test a model for its accuracy
corrects = 0
for img, label in zip(testX, testY):
    answer = np.argmax(model.classify(img))
    if label == answer:
        corrects += 1

#print out accuracy
accuracy = corrects / len(testX)
print(f"accuracy: {accuracy * 100}%")

#you can use pickle library to save your model and use it again on another projects
data = pickle.dumps(model)
with open("~file path~", "rb") as f:
    f.write(data)

#loading model from a file and rebuilding an model
with open("~file path~", "rb") as f:
    model = pickle.dumps(f.read())
print(model)


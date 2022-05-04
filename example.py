import imgClassifier
import numpy as np
import pickle
from tensorflow.keras.datasets import mnist

#create a model
model = imgClassifier.Classifier((28, 28), 10)

#load mnist data
(trainX, trainY),(testX, testY) = mnist.load_data()

#train a model
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

#you can use pickle library to save your model and use it again on another projects (this works on the same project, of course)
data = pickle.dumps(model)
f = open("~file path~", "wb")
f.write(data)
f.flush()
f.close()

#loading model from a file
f = open("~file path~", "rb")
nodel = pickle.loads(f.read())
f.close()

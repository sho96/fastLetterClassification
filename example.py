import fastImgClassifier as imgClassifier
import numpy as np
import pickle
from keras.datasets import mnist
from tqdm import tqdm

#create a model with an input shape of 28x28 and 10 outputs
model = imgClassifier.Classifier((28, 28), 10)

#load mnist data
(trainX, trainY),(testX, testY) = mnist.load_data()

#train a model with input and its label (index of an output neuron)
for img, label in tqdm(zip(trainX, trainY)):
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

file_path = "digitClassifier"
#use pickle library to save the model
data = pickle.dumps(model)
with open(file_path, "wb") as f:
    f.write(data)

#load model from a file
with open(file_path, "rb") as f:
    model = pickle.dumps(f.read())



#doing the same thing with fastImgClassifier
import fastImgClassifier

model = fastImgClassifier.Classifier((28,28), 10)

model.trainAll(trainX, trainY)

accuracy = model.evaluate(testX, testY)

print(f"accuracy: {accuracy * 100}%")

file_path = "digitClassifier"

data = pickle.dumps(model)

with open(file_path, "wb") as f:
    f.write(data)
    
with open(file_path, "rb") as f:
    model = pickle.loads(f.read())

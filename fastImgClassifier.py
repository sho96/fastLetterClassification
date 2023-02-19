import numpy as np
import random
import time
from keras.datasets import mnist
from tqdm import tqdm
import cv2
 
class Classifier:
    def __init__(self, imgSize, outputs, randomWeights = True, initVal = 1):
        self.imgSize = imgSize
        if randomWeights:
            self.weights = np.array([[[random.uniform(-1.0, 1.0)] * imgSize[0] for i in range(imgSize[1])] for i in range(outputs)], dtype = np.float32)
        else:
            self.weights = np.array([[[initVal] * imgSize[0] for i in range(imgSize[1])] for i in range(outputs)], dtype = np.float32)
        self.outputs = np.array([0] * outputs, dtype = np.float32)
        self.biases = np.array([0] * outputs , dtype = np.float32)
        
    def train(self, img, outputIndex, step = 1): #use trainAll method for faster speed
        for i in range(len(self.outputs)):
            self.outputs[i] = np.sum(img * self.weights[i])
            
        for i, output in enumerate(self.outputs):
            if i == outputIndex:
                if output < self.biases[i]:
                    self.weights[i] = self.weights[i] + img * step
            else:
                if output > self.biases[i]:
                    self.weights[i] = self.weights[i] - img * step
    def trainAll(self, imgs, labels, step=1):
        multiplied = imgs * step
        for q, a, m in tqdm(zip(imgs, labels, multiplied)):
            outputs = np.sum(self.weights * q, axis=(1,2))
            for i, output in enumerate(outputs):
                if i == a and output < 0:
                    self.weights[i] += m
                if i != a and output > 0:
                    self.weights[i] -= m
    def evaluate(self, imgs, labels):
        numOfOutputs = len(self.outputs)
        result = np.zeros((numOfOutputs, len(imgs)))
        for i in range(numOfOutputs):
            result[i] = np.sum(imgs * self.weights[i], axis=(1,2))
        result = np.rot90(result)
        return np.count_nonzero(np.argmax(result, axis=1) == labels[::-1])/len(labels)
    def classify(self, img):
        for i in range(len(self.outputs)):
            self.outputs[i] = np.sum(img * self.weights[i])
        return self.outputs

from numba.experimental import jitclass
from numba import int32, float32
import numpy as np
import random
 
class imgClassifier:
    def __init__(self, imgSize, outputs):
        self.imgSize = imgSize
        self.weights = np.array([[[random.uniform(-1.0, 1.0)] * imgSize[0] for i in range(imgSize[1])] for i in range(outputs)], dtype = np.float32)
        self.outputs = np.array([0] * outputs, dtype = np.float32)
        self.biases = np.array([0] * outputs , dtype = np.float32)
        
    def train(self, img, outputIndex, step = 1):
        for i in range(len(self.outputs)):
            self.outputs[i] = np.sum(img * self.weights[i])
            
        for i, output in enumerate(self.outputs):
            if i == outputIndex:
                if output < self.biases[i]:
                    self.weights[i] = self.weights[i] + img * step
            else:
                if output > self.biases[i]:
                    self.weights[i] = self.weights[i] - img * step

    def classify(self, img):
        for i in range(len(self.outputs)):
            self.outputs[i] = np.sum(img * self.weights[i])
        return self.outputs

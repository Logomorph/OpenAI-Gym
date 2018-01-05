import gym
import random

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import optimizers
import numpy as np

env = gym.make('CartPole-v0')
env.reset()

learningRate = 1e-3
trainingSets = 1000
trainingEpochs = 100

def generateTrainingData():
    trainingInput = []
    trainingOutput = []
    dataSets = 0
    for i in range(trainingSets):
        dataIn = []
        dataOut = []
        oldObservation = []
        totalReward = 0
        env.reset()
        for _ in range(200):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            if len(oldObservation) > 0:
                # store the old observation and the action we did. the action we did is based on the old observation
                inputData = []
                for value in oldObservation:
                    inputData.append(value)
                dataIn.append(inputData)
                dataOut.append([1,0] if action == 0 else [0,1])
            oldObservation = observation
            totalReward += reward

            if done:
                break
        if totalReward > 50:
            print totalReward
            trainingInput += dataIn
            trainingOutput += dataOut
            dataSets += 1

    return dataSets, trainingInput, trainingOutput
        
def buildNeuralNet(inputSize):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_dim=inputSize))
    model.add(Dropout(0.5))
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.summary()

    adam = Adam(lr=0.001)
    model.compile(adam, 'mse')
    return model

def testModel(model):
    results = []
    for i in range(100):
        env.reset()
        oldObservation = []
        score = 0
        for _ in range(200):
            action = 0
            if len(oldObservation) == 0:
                # start with a random action, as we can't predict anything yet
                action = random.randrange(0,2)
            else:
                prediction = model.predict(np.array([oldObservation]))
                action = np.argmax(prediction[0])
            observation, reward, done, info = env.step(action)
            
            oldObservation = observation
            score += reward

            if done:
                break
        results.append(score)

    return sum(results)/len(results)

dataSets, x_train, y_train = generateTrainingData()
model = buildNeuralNet(4)
print("%i data sets" % (dataSets))
model.fit(x_train, y_train,
          epochs=trainingEpochs,
          batch_size=128,
          verbose=1)
score = model.evaluate(x_train, y_train, batch_size=128)
print score

print testModel(model)
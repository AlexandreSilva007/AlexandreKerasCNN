import os
import matplotlib.pylab as plt
import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Cropping2D

from AlexandreKerasCNN.Functions import ActivationFunction, LossFunction, OptimizerFunction
from AlexandreKerasCNN.Kernel import Kernel
from AlexandreKerasCNN.DeepNeuralNetwork import DeepNeuralNetwork
from AlexandreKerasCNN.Inception import Inception
#from sklearn.utils import shuffle

class CNN(DeepNeuralNetwork):
    
  def __init__(self, num_classes, name):
    super(CNN, self).__init__(num_classes, name)
  
  #override adding augment_data
  def configureLearning(self, loss_function, optr_function, batch_size, epochs, augment_data=True):
    super(CNN,self).configureLearning(loss_function, optr_function, batch_size, epochs)
    if(augment_data):
      self.augmentInputData()
  
  def augmentInputData(self):
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=True,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(self.input_train)
    # fits the model on batches with real-time data augmentation:
    self._model.fit_generator(datagen.flow(self.input_train, self.output_train, self._batch_size), steps_per_epoch=len(self.input_train) / self._batch_size, epochs=self._epochs)
  
  #number of filters (int), kernel object, input shape = shape
  def add2DConvolutionLayer(self, num_kernels, kernel):
    if((not hasattr(self, 'input_train')) or (self.input_train is None)):
      raise ValueError('Carregue os dados de entrada primeiro! Sem input_shape')
    self._model.add(Conv2D(num_kernels, kernel.size, padding=kernel.padding, input_shape=self.input_train.shape[1:]))
    self._model.add( ActivationFunction.ReLU() )
    
  def add2DMaxPoolingLayer(self, size):
    self._model.add(MaxPooling2D(pool_size=size))
    
  def add2DAveragePoolingLayer(self, size):
    self._model.add(AveragePooling2D(pool_size=size))
    
  def addDropoutLayer(self, rate=0.2):
    self._model.add(Dropout(rate))    
    
  def addCropLayer(self, top=0,left=0, bottom=0, right=0):
    self._model.add(Cropping2D(cropping=((top,left), (bottom,right))))
    
  def addInceptionLayer(self, inceptionlayer):
    self._model.add(inceptionlayer.concatenateLayers())

  def createInception(self):
    return Inception(self)
    
  def printImageSamples(self, img_data_array, size=(10,5),columns=8,rows=4):
    if(len(self._model.layers)>0):
      newimg = self._model.predict(img_data_array)
      print('predicted shape: ', newimg.shape)
      fig=plt.figure(figsize=size)
      for i in range(0, rows):
        for j in range(0, columns):
          img = newimg[(i*columns)+j]
          x = img[:,:,j]
          fig.add_subplot(rows, columns, (i*columns)+j+1) #+1, subplot starts in 1
          plt.grid(False)
          plt.imshow(x)
      plt.show()
      if(img_data_array.shape[0] > (columns*rows+columns+3)):
        fig2 = plt.figure(figsize=(size[0], size[1]/rows))# 1 line
        for i in range(0, columns):
          img = newimg[rows*columns+1]
          x = img[:,:,i:i+3]
          fig2.add_subplot(1, columns, i+1) #+1, subplot starts in 1
          plt.grid(False)
          plt.imshow(x)
        plt.show()
    else:
      fig=plt.figure(figsize=size)
      newimg = img_data_array
      for i in range(1, columns*rows +1):
        img = newimg[i]#self.input_train[i]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
      plt.show()  

  def dataDistribution(self):
    print('Distribuição dos Dados')
    objects = np.zeros(self._NUM_CLASSES)
    for i in range(0,self.output_train.shape[0]):
      out  = np.argmax(self.output_train[i])
      objects[out] += 1
    fig=plt.figure(figsize=(10,5))
    fig.add_subplot(1, 2, 1)
    for i in range(0,self._NUM_CLASSES):
      plt.bar(i, objects[i], 0.8)
      plt.xlabel('Classes')
      plt.ylabel('Elementos')
    fig.add_subplot(1, 2, 2)
    for i in range(0,self._NUM_CLASSES):
      plt.pie(objects )
      plt.xlabel('Classes x Elementos')
    plt.show()

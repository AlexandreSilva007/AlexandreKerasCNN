import os
import matplotlib.pylab as plt
import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from google.colab import files
#import matplotlib.pylab as plt
#import numpy as np
from AlexandreKerasCNN.DeepNeuralNetwork import DeepNeuralNetwork

class CNN(DeepNeuralNetwork):
  __AUGMENTATION = True
  __NUM_PREDICTIONS = 20
    
  def __init__(self, num_classes, name):
    super(CNN, self).__init__(num_classes, name)
        
  def loadCIFAR10Data(self):
    CNN.__TMP_DIRECTORY = os.path.join(os.getcwd(), 'CIFAR10/saved_models')
    print('Download to: ', CNN.__TMP_DIRECTORY)
    # The data, split between train and test sets:
    (self.input_train, self.output_train), (self.input_test, self.output_test) = cifar10.load_data()
    print('out_train shape:', self.output_train.shape)
    print(self.input_train.shape[0], 'train samples')
    print(self.input_test.shape[0], 'test samples')
    print('converting to binary matrices...')
    self.output_train = self.convertYVector2BinaryMatrix(self.output_train)
    self.output_test = self.convertYVector2BinaryMatrix(self.output_test)
    print('new output_train shape:', self.output_train.shape)
    print('new output_test shape:', self.output_test.shape)
    
    self.input_train = self.input_train.astype('float32')
    self.input_test = self.input_test.astype('float32')
    self.input_train /= 255
    self.input_test /= 255
    #self.input_train = np.dot(self.input_train[...,:3], [0.2125, 0.7154, 0.0721])
    #self.input_test = np.dot(self.input_test[...,:3], [0.2125, 0.7154, 0.0721])
   
    self.printImageSamples(size=(10,10), columns=8,rows=4, img_data_array=self.input_train)
    self.augmentInputData()
    #self.input_train = np.expand_dims(self.input_train,axis=3)
    #self.input_test = np.expand_dims(self.input_test,axis=3)
  
  def augmentInputData(self):
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(self.input_train)
  
  #number of filters (int), kernel object, input shape = shape
  def add2DConvolutionLayer(self, num_kernels, kernel):
    if((not hasattr(self, 'input_train')) or (self.input_train is None)):
      raise ValueError('Carregue os dados de entrada primeiro! Sem input_shape')
    self._model.add(Conv2D(num_kernels, kernel.size, padding=kernel.padding, input_shape=self.input_train.shape[1:]))
    self._model.add( ActivationFunction.ReLU() )
  
  def add2DMaxPoolingLayer(self, size):
    self._model.add(MaxPooling2D(pool_size=size))
    
  def add2DMinPoolingLayer(self, size):
    self._model.add(MinPooling2D(pool_size=size))
    
  def addDropoutLayer(self, rate=0.2):
    self._model.add(Dropout(rate))    
    
  def printImageSamples(self, img_data_array, size=(8,8),columns=8,rows=2):
    if(len(self._model.layers)>0):
      newimg = self._model.predict(img_data_array)
      for j in range(0,3,1): #exibir 3 dos filtros apenas...
        fig=plt.figure(figsize=size)
        for i in range(1, columns*rows +1):
          img = newimg[i]
          fig.add_subplot(rows, columns, i)
          plt.imshow(img[:,:,j:(j+3)])
        plt.show()
    else:
      fig=plt.figure(figsize=size)
      newimg = img_data_array
      for i in range(1, columns*rows +1):
        img = newimg[i-1]#self.input_train[i]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
      plt.show()  

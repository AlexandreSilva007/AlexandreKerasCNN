import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Cropping2D

from AlexandreKerasCNN.Functions import ActivationFunction, LossFunction, OptimizerFunction
from AlexandreKerasCNN.Kernel import Kernel
from AlexandreKerasCNN.CNN import CNN

def addInceptionLayer(self, inceptionlayer):
	self._model.add(inceptionlayer.concatenateLayers())

def createInception(self):
	return Inception(self)

	
class Inception():
	
	def __init__(self, model):
		self.operations = []
		self.model = model
	
	def add2DConvolutionLayer(self, num_kernels, kernel):
		if((not hasattr(self, 'input_train')) or (self.input_train is None)):
			raise ValueError('Carregue os dados de entrada primeiro! Sem input_shape')
		self.operations.add(Conv2D(num_kernels, kernel.size,  padding='same', input_shape=model.input_train.shape[1:]))
  
	def add2DMaxPoolingLayer(self, size):
		self.operatons.add(MaxPooling2D(pool_size=size, padding='same'))
    
	def add2DAveragePoolingLayer(self, size):
		self.operations.add(AveragePooling2D(pool_size=size, padding='same'))
    
	def concatenateLayers(self):
		return keras.layers.concatenate(operations, axis = 3)
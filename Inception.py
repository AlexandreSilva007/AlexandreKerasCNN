import numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Cropping2D

from AlexandreKerasCNN.Functions import ActivationFunction, LossFunction, OptimizerFunction
from AlexandreKerasCNN.Kernel import Kernel

class Inception():
	
	def __init__(self, model):
		self.operations = []
		self.model = model
		self.last_layer = None
	
	def add2DConvolutionLayer(self, num_kernels, kernel):
		if((not hasattr(self.model, 'input_train')) or (self.model.input_train is None)):
			raise ValueError('Carregue os dados de entrada primeiro! Sem input_shape')
			
		if self.last_layer is None:
			input_obj = Input(shape = self.model.input_train.shape[1:])
			self.last_layer = Conv2D(num_kernels, kernel.size,  padding='same')(input_obj)
		else:
			self.last_layer = Conv2D(num_kernels, kernel.size,  padding='same')(self.last_layer)  
			
		self.operations.append(self.last_layer)
  

	def add2DMaxPoolingLayer(self, size):
		self.operations.append(MaxPooling2D(pool_size=size, padding='same'))
    
	def add2DAveragePoolingLayer(self, size):
		self.operations.append(AveragePooling2D(pool_size=size, padding='same'))
    
	def concatenateLayers(self):
		return keras.layers.concatenate(self.operations, axis = 3)

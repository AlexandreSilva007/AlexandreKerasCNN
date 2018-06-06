import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from google.colab import files
import matplotlib.pylab as plt
import numpy as np
from AlexandreKerasCNN.CustomCallback import CustomCallback

class DeepNeuralNetwork(CustomCallback):
	SAVE_DIRECTORY = 'KerasSavedModels/'
	_configured = False
	_flattened_data = False
 
	def __init__(self, num_classes, name):
		self._model = Sequential()
		self._NUM_CLASSES = num_classes
		self._CONFUSION_LABELS = "ABCDEFGHIJKLMNOPQRSTWUXYZ" #to be set in specific classes
		self._NAME = name
		self.input_train, self.output_train = None, None
		self.input_test, self.output_test = None, None
 
	def convertYVector2BinaryMatrix(self, y_vector):
		y_binary_matrix = keras.utils.to_categorical(y_vector, self._NUM_CLASSES)
		return y_binary_matrix

	def addFullyConnectedLayer(self, num_neurons, activation_function):
		if(not self._flattened_data):
			self._model.add(Flatten())
			self._flattened_data  = True
		self._model.add(Dense(num_neurons))
		self._model.add(activation_function)
    
    
	def addOutputLayer(self, activation_function):
		if(not self._flattened_data):
		  self._model.add(Flatten())
		  self._flattened_data  = True
		self._model.add(Dense(self._NUM_CLASSES))
		self._model.add(activation_function)
    
	def addNormalizationLayer(self):
		self._model.add(BatchNormalization())

	def configureLearning(self, loss_function, optr_function, batch_size, epochs):
		self._model.compile(loss=loss_function, optimizer=optr_function, metrics=['accuracy'])
		self._batch_size = batch_size
		self._epochs = epochs
		self._configured = True    
		self._batch_step = batch_size / (self.input_train.shape[0]*0.1)
		self._epoch_step = epochs / 10
		print(self._model.summary())
    
	def train(self, shuffle = True, save=True, verbose=0):
		if(not self._configured):
		  raise ValueError('É necessário informar as funcoes de erro,otimizador,lote e épocas. Invoque configureLearning primeiro.')
		self.history = self._model.fit(self.input_train, self.output_train, batch_size = self._batch_size, epochs = self._epochs, validation_data=(self.input_test, self.output_test), shuffle=shuffle, verbose=verbose, callbacks=[self]) 
		if(save):
		  self.save()
		self.showAccuracyGraph()
		return self.history
		
	def save(self):
		if not os.path.isdir(DeepNeuralNetwork.SAVE_DIRECTORY):
		  os.makedirs(DeepNeuralNetwork.SAVE_DIRECTORY)
		self._model_path = os.path.join(os.getcwd(), DeepNeuralNetwork.SAVE_DIRECTORY , self._NAME)
		self._model.save(self._model_path)
		print('Saved trained model at %s ' % self._model_path)
	    
	def download(self):
		print('Downloading trained model at %s ' % self._model_path)
		files.download( self._model_path )
    
	def evaluate(self):
		scores = self._model.evaluate(self.input_test, self.output_test, verbose=1)
		print('Test loss:', scores[0])
		print('Test accuracy:', scores[1])
        
	def showAccuracyGraph(self):
		plt.title('Precisão')
		plt.plot(self.history.history['acc'])
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_acc'])
		plt.xlabel('Epocas')
		plt.ylabel('Valor')
		plt.legend(['Treino', 'Erro', 'Validacao'], loc='upper right')
		plt.show()

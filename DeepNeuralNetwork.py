import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from google.colab import files
import matplotlib.pylab as plt
import numpy as np
from AlexandreKerasCNN.CustomCallback import CustomCallback

class DeepNeuralNetwork(CustomCallback):
	SAVE_DIRECTORY = 'KerasSavedModels/'
	__configured = False
 
	def __init__(self, num_classes, name):
		self.__model = Sequential()
		self.__NUM_CLASSES = num_classes
		self.__NAME = name
		self.input_train, self.output_train = None, None
		self.input_test, self.output_test = None, None
 
	def convertYVector2BinaryMatrix(self, y_vector):
		y_binary_matrix = keras.utils.to_categorical(y_vector, self.__NUM_CLASSES)
		return y_binary_matrix

	def addFullyConnectedLayer(self, num_neurons, activation_function):
		if(not self.__flattened_data):
			self.__model.add(Flatten())
			self.__flattened_data  = True
		self.__model.add(Dense(num_neurons))
		self.__model.add(activation_function)
    
    
	def addOutputLayer(self, activation_function):
		if(not self.__flattened_data):
		  self.__model.add(Flatten())
		  self.__flattened_data  = True
		self.__model.add(Dense(self.__NUM_CLASSES))
		self.__model.add(activation_function)
    
	def configureLearning(self, loss_function, optr_function, batch_size, epochs):
		self.__model.compile(loss=loss_function, optimizer=optr_function, metrics=['accuracy'])
		self.__batch_size = batch_size
		self.__epochs = epochs
		self.__configured = True    
		self.__batch_step = batch_size / (self.input_train.shape[0]*0.01)
    
	def train(self, shuffle = True, save=True, verbose=0):
		if(not self.__configured):
		  raise ValueError('É necessário informar as funcoes de erro,otimizador,lote e épocas. Invoque configureLearning primeiro.')
		self.history = self.__model.fit(self.input_train, self.output_train, batch_size = self.__batch_size, epochs = self.__epochs, validation_data=(self.input_test, self.output_test), shuffle=shuffle, verbose=verbose, callbacks=[self]) 
		if(save):
		  self.save()
		self.showAccuracyGraph()
		return self.history
		
	def saveCNN(self):
		if not os.path.isdir(CNN.SAVE_DIRECTORY):
		  os.makedirs(CNN.SAVE_DIRECTORY)
		self.__model_path = os.path.join(os.getcwd(), CNN.SAVE_DIRECTORY , self.__NAME)
		self.__model.save(self.__model_path)
		print('Saved trained model at %s ' % self.__model_path)
	    
	def downloadCNN(self):
		print('Downloading trained model at %s ' % self.__model_path)
		files.download( self.__model_path )
    
	def evaluate(self):
		scores = self.__model.evaluate(self.input_test, self.output_test, verbose=1)
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
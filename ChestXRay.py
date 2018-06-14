import os
import matplotlib.pylab as plt
import numpy as np
import sys
import urllib.request
import zipfile
from PIL import Image
import skimage
from skimage.transform import resize
from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils.np_utils import to_categorical

from AlexandreKerasCNN.Functions import ActivationFunction, LossFunction, OptimizerFunction
from AlexandreKerasCNN.Kernel import Kernel
from AlexandreKerasCNN.DeepNeuralNetwork import DeepNeuralNetwork
from AlexandreKerasCNN.CNN import CNN
from AlexandreKerasCNN.CustomCallback import CustomImageAugmentationCallback

class ChestXRay(CNN):
	def __init__(self, num_classes, name):
		super(ChestXRay, self).__init__(num_classes, name)
		self._CONFUSION_LABELS = "NP"
		
	def downloadDriveZip(self, local_download_path, drive_file, file_name):
		# 1. Authenticate and create the PyDrive client.
		auth.authenticate_user()
		gauth = GoogleAuth()
		gauth.credentials = GoogleCredentials.get_application_default()
		drive = GoogleDrive(gauth)
		try:
			os.makedirs(local_download_path)
		except: pass
		fname = os.path.join(local_download_path, file_name)
		if(os.path.isfile(fname)):
			print('skipping, file already exists', end="")
		else:
			fileId = drive.CreateFile({'id': drive_file}) #DRIVE_FILE_ID is file id example: 1iytA1n2z4go3uVCwE_vIKouTKyIDjEq
			#print('filename: ',fileId['title'])
			fileId.GetContentFile(fileId['title'])
			#print(fileId)
			print("wait..extracting zip...")
			zip_ref = zipfile.ZipFile(fileId['title'], 'r')
			zip_ref.extractall(local_download_path)
			zip_ref.close()
			print('Xtracted to: ',local_download_path)

	def loadChestXRayFromDrive(self):
		local_download_path = os.path.join(os.getcwd())
		drive_file = '1BMdv-PRDZwI91IVdHsNFuNbKX2AgDGa6'
		file_name = 'chest_xray.zip'
		self.downloadDriveZip(local_download_path, drive_file, file_name)
		
		print('Carregando e redimensionando imagens na memória...')
		self.input_train, self.output_train = self.get_data(local_download_path+'/chest_xray/train/')
		self.input_test, self.output_test = self.get_data(local_download_path+'/chest_xray/test/')
		print('Imagens carregadas\r\n')
		
		self.input_train, self.output_train = shuffle(self.input_train, self.output_train, random_state=1)
		self.input_test, self.output_test = shuffle(self.input_test, self.output_test, random_state=1)
		
		self.input_train = self.input_train.astype('float32')
		self.input_test = self.input_test.astype('float32')
		
		print('ORIGINAL TRAINNING SAMPLES: ', self.input_train.shape[0])
		print('Test samples: ', self.input_test.shape[0])
		not_hot_output_train = self.output_train
		self.output_train = to_categorical(self.output_train, num_classes = self._NUM_CLASSES)
		self.output_test = to_categorical(self.output_test, num_classes = self._NUM_CLASSES)
		self.dataDistribution()
		self.output_train = not_hot_output_train
	
		print('Balancing TRAINNING data...')
		self.balanceInputData()
		print('BALLANCED TRAINNING SAMPLES: ', self.input_train.shape[0])
		
		self.dataDistribution()
		self.printImageSamples(size=(12,6), columns=6,rows=3, img_data_array=self.input_train)

	def balanceInputData(self):
		# Deal with imbalanced class sizes below
		# Make Data 1D for compatability upsampling methods
		X_trainShape = self.input_train.shape[1]*self.input_train.shape[2]*self.input_train.shape[3]
		#X_testShape = self.input_test.shape[1]*self.input_test.shape[2]*self.input_test.shape[3]
		X_trainFlat = self.input_train.reshape(self.input_train.shape[0], X_trainShape)
		#X_testFlat = self.input_test.reshape(self.input_test.shape[0], X_testShape)
		Y_train = self.output_train
		#Y_test = self.output_test

		ros = RandomUnderSampler(ratio='auto')
		X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)
		#X_testRos, Y_testRos = ros.fit_sample(X_testFlat, Y_test)
		# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
		#print(Y_trainRos.shape)
		#print(Y_trainRos)
		Y_trainRosHot = to_categorical(Y_trainRos, num_classes = self._NUM_CLASSES)
		#Y_testRosHot = to_categorical(Y_testRos, num_classes = self._NUM_CLASSES)
		# Make Data 2D again
		for i in range(len(X_trainRos)):
		    height, width, channels = 100,150,3
		    X_trainRosReshaped = X_trainRos.reshape(len(X_trainRos),height,width,channels)
		#for i in range(len(X_testRos)):
		    #height, width, channels = 100,150,3
		    #X_testRosReshaped = X_testRos.reshape(len(X_testRos),height,width,channels)
		
		self.input_train = X_trainRosReshaped
		self.output_train =  Y_trainRosHot
		#self.input_test = X_testRosReshaped
		#self.output_test =  Y_testRosHot
		
		
	def get_data(self,folder):
		X = []
		y = []
		count=0
		for folderName in os.listdir(folder):
			if not folderName.startswith('.'):
				if folderName in ['NORMAL']:
					label = 0#[0,1]
				elif folderName in ['PNEUMONIA']:
					label = 1#[1,0]
				else:
					label = 0
					print('past nao esperada')
				for image_filename in (os.listdir(folder + folderName)):
					count += 1
					if(count%3==0):
						label=1
					else:
						label=0
					if (count>20): 
						break
					if( (os.path.splitext(image_filename.upper())[1] == '.JPG') or (os.path.splitext(image_filename.upper())[1] == '.JPEG') ):
						fpath = folder + folderName + '/' + image_filename
						print('\rLoading File: ', fpath, end="")
						img_file = Image.open(fpath).convert("RGB")
						img_file.load()
						img_file = np.asarray(img_file)
						if img_file is not None:
							img_file = skimage.transform.resize(img_file, (150, 150, 3))
							img_file = img_file[30:130,0:150]#.crop((7, 12, 163, 138)) #crop
							X.append(img_file)
							y.append(label)
						else:
							print('ops')
		X = np.asarray(X)
		y = np.asarray(y)
		return X,y
		
		
	def listfolders():
	  for dirname, dirnames, filenames in os.walk('.'):
		  # print path to all subdirectories first.
		  for subdirname in dirnames:
			  print(os.path.join(dirname, subdirname))

		  # print path to all filenames.
		  for filename in filenames:
			  print(os.path.join(dirname, filename))

		  # Advanced usage:
		  # editing the 'dirnames' list will stop os.walk() from recursing into there.
		  if '.git' in dirnames:
			  # don't go into any .git directories.
			  dirnames.remove('.git')
		  
		
	def augmentInputData(self):
		datagen = ImageDataGenerator(
			featurewise_center=False,  # set input mean to 0 over the dataset
			samplewise_center=False,  # set each sample mean to 0
			featurewise_std_normalization=False,  # divide inputs by std of the dataset
			samplewise_std_normalization=True,  # divide each input by its std
			zca_whitening=False,  # apply ZCA whitening
			rotation_range=7,  # randomly rotate images in the range (degrees, 0 to 180)
			width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
			height_shift_range=0,  # randomly shift images vertically (fraction of total height)
			horizontal_flip=False,  # randomly flip images
			vertical_flip=False)  # randomly flip images
		# Compute quantities required for feature-wise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(self.input_train)
		daCallback = CustomImageAugmentationCallback(self._epochs)
		# fits the model on batches with real-time data augmentation:
		self._model.fit_generator(datagen.flow(self.input_train, self.output_train, self._batch_size), steps_per_epoch=len(self.input_train) / self._batch_size, epochs=self._epochs, verbose=0, callbacks=[daCallback])

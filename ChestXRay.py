import os
import matplotlib.pylab as plt
import numpy as np
import sys
import urllib.request
import zipfile
from PIL import Image
import skimage
from skimage.transform import resize

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

class ChestXRay(CNN):
	def __init__(self, num_classes, name):
		super(ChestXRay, self).__init__(num_classes, name)
		
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
			print('filename: ',fileId['title'])
			fileId.GetContentFile(fileId['title'])
			print(fileId)
			print("wait..extracting...")
			zip_ref = zipfile.ZipFile(fileId['title'], 'r')
			zip_ref.extractall(local_download_path)
			zip_ref.close()
			print('Xtracted to: ',local_download_path)

	def loadChestXRayFromDrive(self):
		local_download_path = os.path.join(os.getcwd())
		drive_file = '1BMdv-PRDZwI91IVdHsNFuNbKX2AgDGa6'
		file_name = 'chest_xray.zip'
		self.downloadDriveZip(local_download_path, drive_file, file_name)
		self.input_train, self.output_train = self.get_data(local_download_path+'/chest_xray/train/')
		self.input_test, self.output_test = self.get_data(local_download_path+'/chest_xray/test/')
		self.output_train = self.convertYVector2BinaryMatrix(self.output_train)
		self.output_test = self.convertYVector2BinaryMatrix(self.output_test)
		print('Train: ', self.input_train.shape)
		print('Test: ', self.input_test.shape)
		self.input_train = self.input_train.astype('float32')
		self.input_test = self.input_test.astype('float32')
		#self.input_train /= 255
		#self.input_test /= 255
		self.printImageSamples(size=(12,6), columns=6,rows=3, img_data_array=self.input_train)
		self.dataDistribution()
		
	def get_data(self,folder):
		X = []
		y = []
		for folderName in os.listdir(folder):
			if not folderName.startswith('.'):
				if folderName in ['NORMAL']:
					label = 0
				elif folderName in ['PNEUMONIA']:
					label = 1
				else:
					label = 2
				for image_filename in (os.listdir(folder + folderName)):#tdqm
					if( (os.path.splitext(image_filename.upper())[1] == '.JPG') or (os.path.splitext(image_filename.upper())[1] == '.JPEG') ):
						img_file = Image.open(folder + folderName + '/' + image_filename).convert("RGB")
						img_file.load()
						img_file = np.asarray(img_file)
						if img_file is not None:
							img_file = skimage.transform.resize(img_file, (150, 150, 3))
							img_arr = np.asarray(img_file)
							X.append(img_arr)
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
			samplewise_std_normalization=False,  # divide each input by its std
			zca_whitening=False,  # apply ZCA whitening
			rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
			width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
			height_shift_range=0,  # randomly shift images vertically (fraction of total height)
			horizontal_flip=True,  # randomly flip images
			vertical_flip=False)  # randomly flip images
		# Compute quantities required for feature-wise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(self.input_train)
		# fits the model on batches with real-time data augmentation:
		self._model.fit_generator(datagen.flow(self.input_train, self.output_train, self._batch_size), steps_per_epoch=len(self.input_train) / self._batch_size, epochs=self._epochs)

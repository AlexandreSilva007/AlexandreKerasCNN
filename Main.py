# -*- coding: utf-8 -*-
"""CIFAR10.ipynb
Automatically generated by Colaboratory.
"""
#download das classes necessárias do repositório
!rm -rf AlexandreKerasCNN
!git clone https://github.com/AlexandreSilva007/AlexandreKerasCNN.git
!ls AlexandreKerasCNN

import time
import numpy as np
from AlexandreKerasCNN.Functions import ActivationFunction, LossFunction, OptimizerFunction
from AlexandreKerasCNN.Kernel import Kernel
from AlexandreKerasCNN.CNN import CNN
#salva um modelo a cada treinamento
name = 'KerasCNNModel_' +  str(time.time()).replace('.','') + '.h5'

#Carrega Dados
modelo = CNN(num_classes=10, name=name )
modelo.loadCIFAR10Data()

#Camada Convolucional + MaxPooling 
modelo.add2DConvolutionLayer(32, Kernel((3,3), has_padding=False))
modelo.add2DMaxPoolingLayer((2,2))
modelo.addDropoutLayer(0.2)
print("Amostra CL 1")
# imprime "amostra" dos filtros iniciais. 6x2 filtros + 1 linha de filtros combinados 3 a 3
modelo.printImageSamples(size=(12,4), columns=6,rows=2, img_data_array=modelo.input_train)

#Camada Convolucional + MaxPooling 
modelo.add2DConvolutionLayer(32, Kernel((2,2), has_padding=False))
modelo.add2DMaxPoolingLayer((2,2))
modelo.addDropoutLayer(0.2)
print("Amostra CL 2")
# imprime "amostra" dos filtros iniciais. 6x2 filtros + 1 linha de filtros combinados 3 a 3
modelo.printImageSamples(size=(12,4), columns=6,rows=2,img_data_array=modelo.input_train)

#Camada Convolucional + MaxPooling 
modelo.add2DConvolutionLayer(32, Kernel((2,2), has_padding=False))
modelo.add2DMaxPoolingLayer((2,2))
#modelo.add2DAveragePoolingLayer((3,3))
modelo.addDropoutLayer(0.2)
print("Amostra CL 3")
# imprime "amostra" dos filtros iniciais. 6x2 filtros + 1 linha de filtros combinados 3 a 3
modelo.printImageSamples(size=(12,4), columns=6,rows=2,img_data_array=modelo.input_train)

#modelo.addNormalizationLayer()
 
#Camada Totalmente Conectada (2)
modelo.addFullyConnectedLayer(512, ActivationFunction.ReLU() )
modelo.addFullyConnectedLayer(32, ActivationFunction.Sigmoid())
modelo.addFullyConnectedLayer(512, ActivationFunction.ReLU() )

#Camada de Saída
modelo.addOutputLayer(ActivationFunction.SoftMax() )

#Run
#Para utilizar apenas o dataset original apenas(50k p treino) (e não demorar na geração de dados aumentados dinâmicamente) modifique o parâmetro augment_data = False
modelo.configureLearning(LossFunction.CATEGORICAL_CROSS_ENTROPY, OptimizerFunction.Adam(0.0001), batch_size=50, epochs=150, augment_data=True)
modelo.train(verbose=0) # treina e salva modelo+pesos
modelo.evaluate()

#baixa modelo treinado. No google colaboratory as vezes é necessário aguardar para invocar este método (o arquivo ainda pode estar sendo salvo pelo fim do treinamento)
#modelo.download()
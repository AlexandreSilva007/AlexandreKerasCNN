import keras
import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class CustomImageAugmentationCallback(keras.callbacks.Callback):
  def __init__(self, max_epochs):
    super(CustomImageAugmentationCallback, self).__init__()
    self.__max_epochs = max_epochs
    
  def on_train_begin(self, logs={}):
    self.symbols = ['—','\\','|','/','—','\\','/']
    print('Começando data augmentation...')
  def on_train_end(self, logs = None):
    print('Fim da data augmentation')
  def on_epoch_begin(self, epoch, logs = None):
    self.__epoch = epoch
    print("\rComecando DA ",epoch, end="", flush=True)
  def on_batch_begin(self, batch, logs = None):
    index = batch%7 
    print( "\rProcessando época %s de %s: %s " % (self.__epoch+1, self.__max_epochs, self.symbols[index]), end="", flush=True)

class CustomCallback(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.hist_train = []
    self.hist_test = []
    self._epoch_percentage_count = 0
    self.val_f1s = []
    self.val_recalls = []
    self.val_precisions = []
    
  def on_epoch_begin(self, epoch, logs = None):
    print("\rComecando época ",epoch, end="", flush=True)
    self._batch_percentage_count = 0
    self.acc = []
    self.val_acc = []
    self._epoch_percentage_count += 1
        
  def on_batch_begin(self, batch, logs = None):
    self._batch_percentage_count += self._batch_step
    self.n = int(self._batch_percentage_count)
    self.sqr = "■"*self.n
    print( "\rProgresso época %s: %s %s%s" % (self._epoch_percentage_count, self.sqr,self.n*10,'%'), end="\t", flush=True)
  
  def on_batch_end(self, batch, logs = None):
    self.acc.append(logs.get('acc'))
    self.val_acc.append(logs.get('val_acc'))
  
  def on_epoch_end(self, epoch, logs={}):
    self.hist_train.append(logs.get('acc'))
    self.hist_test.append(logs.get('val_acc'))
    
    val_predict = (np.asarray(self._model.predict(self.input_test))).round()
    val_targ = self.output_test

    _val_f1 = f1_score(val_targ, val_predict, average=None)
    _val_recall = recall_score(val_targ, val_predict,average=None)
    _val_precision = precision_score(val_targ, val_predict,average=None)
    self.val_f1s.append(_val_f1)
    self.val_recalls.append(_val_recall)
    self.val_precisions.append(_val_precision)
    #print ("\r val_f1: %s val_precision: %s val_recall %s" %(_val_f1, _val_precision, _val_recall))
    print ("\r f1_score: %s" % (_val_f1), end="", flush=True)
        
    if(self._epoch_percentage_count%self._epoch_step==0):
      print("\r\n******************** RESULTADO PARCIAL ***********************")
      #confusion matrix
      cm = confusion_matrix(val_targ.argmax(axis=1), val_predict.argmax(axis=1))
      CustomCallback.plot_confusionmatrix2(cm, self._CONFUSION_LABELS)
      
      #Gráficos de precisão
      print("\rEpoca ",epoch, "\tacc: ", logs.get('acc'), "\ttest_acc: ", logs.get('val_acc'), flush=True)#, "\terro:", logs.get('loss'))
      fig=plt.figure(figsize=(15,5))
      fig.add_subplot(1, 3, 1)
      plt.grid(True)
      plt.plot(np.arange(len(self.acc)), self.acc)
      #plt.plot(np.arange(len(self.val_acc)), self.val_acc)
      plt.xlabel('Batch')
      plt.ylabel('Valor')
      plt.legend(['Treino Batch'], loc='upper left')
      fig.add_subplot(1, 3, 2)
      plt.bar(1, logs.get('acc'), 0.8)
      plt.bar(2, logs.get('val_acc'), 0.8)
      plt.bar(3, logs.get('loss'), 0.8)
      plt.xlabel('Elementos')
      plt.ylabel('Precisao')
      plt.legend(['Treino', 'Validacao','Erro'], loc='upper left')
      fig.add_subplot(1, 3, 3)
      plt.grid(True)
      plt.plot(np.arange(len(self.hist_train)), self.hist_train)
      plt.plot(np.arange(len(self.hist_test)), self.hist_test)
      plt.xlabel('Epoca')
      plt.ylabel('Precisao')
      plt.legend(['Treino', 'Validacao'], loc='upper left')
      plt.show()
    
  def on_train_end(self, logs = None):
    print("Finish!!!")
    print('F1s: ',self.val_f1s)
    print('Recalls: ',self.val_recalls)
    print('Precisions: ',self.val_precisions)
    
    
  def plot_confusionmatrix2(conf_arr, alphabet):
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    plt.grid(False)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')
    width, height = conf_arr.shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    cb = fig.colorbar(res)
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    #plt.show()

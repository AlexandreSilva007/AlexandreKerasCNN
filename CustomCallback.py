import keras
import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class CustomCallback(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.hist_train = []
    self.hist_test = []
    self._epoch_percentage_count = 0
    self.val_f1s = []
    self.val_recalls = []
    self.val_precisions = []
    
  def on_epoch_begin(self, epoch, logs = None):
    print("\rComecando época ",epoch, end=" ", flush=True)
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
    print ("\r f1_score: %s" % (_val_f1))
    
    print('confusion: ', self._NUM_CLASSES)
    #confusion matrix
    cm = confusion_matrix(val_targ.argmax(axis=1), val_predict.argmax(axis=1))
    print(cm)
    CustomCallback.plot_confusion_matrix(cm, self._NUM_CLASSES)
        
    if(self._epoch_percentage_count%self._epoch_step==0):
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
    
  def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusão', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusão Normalizada")
    else:
        print('Matriz de Confusão, sem normalização')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

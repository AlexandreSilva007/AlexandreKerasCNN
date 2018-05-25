import keras
import matplotlib.pylab as plt

class CustomCallback(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    print("Go!!!!!!")
    self.hist_train = []
    self.hist_test = []
    
  def on_epoch_begin(self, epoch, logs = None):
    print("\rComecando época ",epoch, end=" ", flush=True)
    self._batch_percentage_count = 0
    self.acc = []
    self.val_acc = []
        
  def on_batch_begin(self, batch, logs = None):
    self._batch_percentage_count += self._batch_step
    self.n = int(self._batch_percentage_count)
    self.sqr = "■"*self.n
    print( "\rProgresso época: %s %s%s" % (self.sqr,self.n*10,'%'), end="\t", flush=True)
  
  def on_batch_end(self, batch, logs = None):
    self.acc.append(logs.get('acc'))
    self.val_acc.append(logs.get('val_acc'))
  
  def on_epoch_end(self, epoch, logs={}):
    print("\rEpoca ",epoch, "\tacc: ", logs.get('acc'), "\ttest_acc: ", logs.get('val_acc'), flush=True)#, "\terro:", logs.get('loss'))
    self.hist_train.append(logs.get('val'))
    self.hist_test.append(logs.get('val_acc'))
    fig=plt.figure(figsize=(15,5))
    fig.add_subplot(1, 3, 1)
    plt.grid(True)
    plt.plot(np.arrange(len(self.acc)), self.acc)
    plt.plot(np.arrange(len(self.val_acc)), self.val_acc)
    plt.xlabel('Batch')
    plt.ylabel('Valor')
    plt.legend(['Treino', 'Validacao'], loc='upper left')
    fig.add_subplot(1, 3, 2)
    plt.bar(1, logs.get('acc'), 0.8)
    plt.bar(2, logs.get('val_acc'), 0.8)
    plt.bar(3, logs.get('loss'), 0.8)
    plt.xlabel('Elementos')
    plt.ylabel('Precisao')
    plt.legend(['Treino', 'Validacao','Erro'], loc='upper left')
    fig.add_subplot(1, 3, 3)
    plt.grid(True)
    plt.plot(np.arrange(len(self.hist_train)), self.hist_train)
    plt.plot(np.arrange(len(self.hist_test)), self.hist_test)
    plt.xlabel('Epoca')
    plt.ylabel('Precisao')
    plt.legend(['Treino', 'Validacao'], loc='upper left')
    plt.show()
    
  def on_train_end(self, logs = None):
    print("Finish!!!")

import keras
import matplotlib.pylab as plt

class CustomCallback(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    print("Go!!!!!!")
    
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
    plt.grid(True)
    plt.plot(self.acc,'b')
    plt.plot(self.val_acc,'r')
    plt.xlabel('Batch')
    plt.ylabel('Valor')
    plt.legend(['Treino', 'Validacao'], loc='upper right')
    plt.show()
       
  def on_train_end(self, logs = None):
    print("Finish!!!")

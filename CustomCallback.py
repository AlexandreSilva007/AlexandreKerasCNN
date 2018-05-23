import keras

class CustomCallback(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    print("Go!!!!!!")
    
  def progressBar(self, n):
    print("\rProgresso: %s %s%s" % ("■"*n,n,'%'), end=" ", flush=False)

  def on_epoch_begin(self, epoch, logs = None):
    print("\rComecando época ",epoch, end=" ")
    self.__batch_percentage_count = 0
        
  def on_batch_begin(self, batch, logs = None):
    self.__batch_percentage_count += self.__batch_step
    self.progressBar( int(self.__batch_percentage_count) )
  
  def on_batch_end(self, batch, logs = None):
    pass
  
  def on_epoch_end(self, epoch, logs={}):
    print("\rFim da época ",epoch, "\tacc: ", logs.get('acc'), "\ttest_acc: ", logs.get('val_acc'), "\terro:", logs.get('loss'))
   
  def on_train_end(self, logs = None):
    print("Finish!!!")
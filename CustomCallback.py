import keras

class CustomCallback(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    print("Go!!!!!!")
    
  def on_epoch_begin(self, epoch, logs = None):
    print("\rComecando época ",epoch, end=" ", flush=True)
    self._batch_percentage_count = 0
        
  def on_batch_begin(self, batch, logs = None):
    self._batch_percentage_count += self._batch_step
    self.n = int(self._batch_percentage_count)
    self.sqr = "■"*n
    print( "\rProgresso época: %s %s%s" % (self.sqr,self.n,'%'), end="", flush=True)
  
  def on_batch_end(self, batch, logs = None):
    print( "\rProgresso época: %s %s%s done" % (self.sqr,self.n,'%'), end="", flush=True)
  
  def on_epoch_end(self, epoch, logs={}):
    print("\rEpoca ",epoch, "\tacc: ", logs.get('acc'), "\ttest_acc: ", logs.get('val_acc'), flush=True)#, "\terro:", logs.get('loss'))
   
  def on_train_end(self, logs = None):
    print("Finish!!!")

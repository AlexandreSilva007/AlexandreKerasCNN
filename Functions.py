class ActivationFunction():
  def SoftMax(naxis=-1):
    return Activation('softmax')
  
  def Elu(nalpha=1.0):
    return Activation('elu', alpha=nalpha)
  
  def SeLU():
    return Activation('selu')
  
  def SoftPlus():
    return Activation('softplus')
  
  def SoftSign():
    return Activation('softsign')
  
  def ReLU(nalpha=0.0, nmax_value=None):
    return Activation('relu')
  
  def Tangent():
    return Activation('tanh')
  
  def Sigmoid():
    return Activation('sigmoid')
  
  def HardSigmoid():
    return Activation('hard_sigmoid')
  
  def Linear():
    return Activation('linear')
  
  
class LossFunction():
  MEAN_SQUARED_ERRO = 'mean_squared_error'
  MEAN_ABSOLUTE_ERROR = 'mean_absolute_error'
  MEAN_ABSOLUTE_PERCENTAGE_ERROR = 'mean_absolute_percentage_error'
  MEAN_SQUARED_LOGARITHIMIC_ERROR = 'mean_squared_logarithmic_error'
  SQUARE_HINGE = 'squared_hinge'
  HINGE='hinge'
  CATEGORICAL_HINGE = 'categorical_hinge'
  LOG_COSH = 'logcosh'
  CATEGORICAL_CROSS_ENTROPY = 'categorical_crossentropy'
  SPARSE_CATEGORICAL_CROSS_ENTROPY = 'sparse_categorical_crossentropy'
  BINARY_CROSS_ENTROPY = 'binary_crossentropy'
  KULLBACK_LEIBLER_DIVERGENCE = 'kullback_leibler_divergence'
  POISSON = 'poisson'
  COSINE_PROXIMITY = 'cosine_proximity'

class OptimizerFunction():
  def RMSprop(lr=0.001):
    return keras.optimizers.RMSprop(lr=lr)
  
  def Adagrad(lr=0.01):
    return keras.optimizers.Adagrad(lr=lr)
  
  def Adadelta(lr=1.0):
    return keras.optimizers.Adadelta(lr=lr)
  
  def Adam(lr=0.001):
    return keras.optimizers.Adam(lr=lr)
  
  def Adamax(lr=0.002):
    return keras.optimizers.Adamax(lr=lr)
  
  def Nadam(lr=0.002):
    return keras.optimizers.Nadam(lr=lr)
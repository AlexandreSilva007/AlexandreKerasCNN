class Kernel():
  #size (x,y), padding = True or False
  def __init__(self, size, has_padding):
    self.size = size
    self.padding = ('same' if has_padding==True else 'valid')
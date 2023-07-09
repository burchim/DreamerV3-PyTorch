class AttrDict(dict):

  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

  def __getstate__(self):
    return self.copy()
  
  def __setstate__(self, mapping):
    self.update(mapping)
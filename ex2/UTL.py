import numpy as np
import functools
import io
import sys
from scipy import stats
'''
this function converts np.genfromtxt to python 3 np.genfromtxt (helper method)
'''
genfromtxt_old = np.genfromtxt
@functools.wraps(genfromtxt_old)
def genfromtxt_py3_fixed(f, encoding="utf-8", *args, **kwargs):
  if isinstance(f, io.TextIOBase):
    if hasattr(f, "buffer") and hasattr(f.buffer, "raw") and \
    isinstance(f.buffer.raw, io.FileIO):
      fb = f.buffer.raw
      fb.seek(f.tell())
      result = genfromtxt_old(fb, *args, **kwargs)
      f.seek(fb.tell())
    else:
      old_cursor_pos = f.tell()
      fb = io.BytesIO(bytes(f.read(), encoding=encoding))
      result = genfromtxt_old(fb, *args, **kwargs)
      f.seek(old_cursor_pos + fb.tell())
  else:
    result = genfromtxt_old(f, *args, **kwargs)
  return result

if sys.version_info >= (3,):
  np.genfromtxt = genfromtxt_py3_fixed
'''
convert sex to float value
'''
def convertSexToFloat(sex):
    sex = str(sex)
    if (sex == "b'M'"):
        return float(1)
    elif (sex == "b'F'"):
        return float(2)
    return float(0)

'''
read data and create numpy array
'''
def readTrainData(train_x_file):
    with open(train_x_file) as fp:
        train_x = np.genfromtxt(fp, delimiter=',', dtype='f8', converters={0: convertSexToFloat})
    return train_x

'''
read data and create numpy array
'''
def readTrainY(train_y_file):
    with open(train_y_file) as fp:
        train_y = np.genfromtxt(fp, dtype=float)
    return train_y

'''
normalize by min max method
'''
def maxMixNormalize_train(data):
    maxVal=[]
    minVal=[]
    data= np.transpose(data)
    for i, line in enumerate(data):
        minVal.append(line.min())
        maxVal.append(line.max())
        if line.max() != line.min():
            data[i] = (line - line.min()) / (line.max() - line.min())
    return np.transpose(data) , minVal,maxVal

'''
normalize by min max method
'''
def maxMixNormalize_test(data,maxVal,minVal):
    data= np.transpose(data)
    for i, line in enumerate(data):
        if maxVal[i] != minVal[i]:
            data[i] = (line - minVal[i]) / (maxVal[i] - minVal[i])
    return np.transpose(data)

'''
normalize by z score method
'''
def normalize_z_train(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = ((data - mean) / std)
    return data,mean,std

'''
normalize by z score method
'''
def normalize_z_test(data,mean,std):
    data = ((data - mean) / std)
    return data

'''
shuffle function
'''
def shuffleData(data_x, data_y):
    idex = np.arange(data_x.shape[0])
    np.random.shuffle(idex)
    data_x = data_x[idex]
    data_y = data_y[idex]
    return data_x,data_y


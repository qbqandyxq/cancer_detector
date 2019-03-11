import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
import pydicom
import pandas as pd
def get_test_num(filename):
    #lines = open(filename, 'r')
    df=pd.read_csv(filename)
    return len(df)

def get_frames_data( filename, num_frames_per_clip = 16 ):  #read ndarray
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  s_index = 0
  
  npyreader = np.load(filename)
  #print(np.shape(npyreader))
  if( np.shape(npyreader)[0] < 2 * num_frames_per_clip + 1 ):
    return [], s_index
  # filenames = sorted(filenames)

  s_index = 1
  #for i in range(s_index, s_index + 2*int(num_frames_per_clip))[ : : 2 ] : # frameskip
  #  image_name = str(filename) + '/' + str(filenames[i])
    #img = Image.open(image_name)
  #  ds = pydicom.read_file(image_name)
  #  img = ds.pixel_array
  #  img_data = np.array(img)
  #  ret_arr.append(img_data)
  ret_arr= npyreader[1:num_frames_per_clip+1,:,:]
  return ret_arr, s_index

def read_clip_and_label(dataset_dir, filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False):
  #lines = open(filename,'r')
  csv_file_data = pd.read_csv(filename)
  dataset_file_path = filename.split('label')[0]+'processed/'
  #print(dataset_file_path)
  lines = csv_file_data['id']
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  next_batch_start = -1
  lines = lines.tolist()
  #np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    shuffle = True
  if shuffle:
    video_indices = list(range(len(lines)))
    random.seed(time.time())
    random.shuffle(video_indices)
  else:
    # Process videos sequentially
    video_indices = list(range(start_pos, len(lines)))
  for index in video_indices:
    if(batch_index>=batch_size):
      next_batch_start = index
      break
    line = lines[index].strip('\n').split()
    dirname = line[0]
    # dirname = os.path.join(dataset_file_path, dirname)
    dirname = os.path.join(dataset_file_path, dirname + '.npy')
    tmp_label = csv_file_data['ret'][index]
    #if not shuffle:
      #print("Loading a video clip from {}...".format(dirname))
    tmp_data, _ = get_frames_data(dirname, num_frames_per_clip)
    
    img_datas = []
    if(len(tmp_data)!=0):
    #if(tmp_data.shape[0]!=0):
      for j in range(len(tmp_data)):
      #for j in range(tmp_data.shape[0]):
        #img = Image.fromarray(tmp_data[j].astype(np.uint8))
        #img =np.array( img.resize((crop_size, crop_size),Image.ANTIALIAS)).reshape(crop_size,crop_size,1)
        #max_pixel=img.max()
        #min_pixel=img.min()
        #img = (img-min_pixel)/(max_pixel-min_pixel)
        #img_datas.append(img)
        img = tmp_data[j].reshape(crop_size,crop_size,1)
        img_datas.append(img)
      data.append(img_datas)
      label.append(int(tmp_label))
      
      batch_index = batch_index + 1
      read_dirnames.append(dirname)

  # pad (duplicate) data/label if less than batch_size
  valid_len = len(data)
  pad_len = batch_size - valid_len
  if pad_len:
    for i in range(pad_len):
      data.append(img_datas)
      label.append(int(tmp_label))

  np_arr_data = np.array(data).astype(np.float32)
  np_arr_label = np.array(label).astype(np.int64)
  #print("over")
  return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len

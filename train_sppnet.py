from spp_net import *
from spp_layer import *
import os
import time
import tensorflow as tf
import numpy as np



train_size = 3060
batch_size = 50
max_epochs =10
num_class = 10
eval_frequency = 100
max_steps = 100000

     
class path_batch():
    
    def __init__(self, path):
        self.path = path
        
    def next_batch(self, batch_size):
        label_len = len(self.path)
        shuffle_index = np.arange(label_len)
        np.random.shuffle(shuffle_index)
        batch_index = shuffle_index[:batch_size]
        path_batch = self.path[batch_index]
        return path_batch

print('load path')
path_list = np.load('/home/user001/spp_net/T-0_pre_data/path_list.npy')
path_list = path_batch(path_list)
if __name__ == '__main__':
    spp = SPPnet(30, path_list, 10000)
    spp.begin('c')



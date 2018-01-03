
# coding: utf-8

# In[ ]:


from __future__ import absolute_import                                                                         
from __future__ import division
from __future__ import print_function

import os
import logging
import math 
import sys
from spp_layer_f import SPPLayer

import numpy as np
import tensorflow as tf
SEED = 1356
stddev = 0.05
class SPPnet:
    def __init__(self):
        self.random_weight= False
        self.wd = 5e-4
        self.stddev = 0.05


    def _conv_layer(self, bottom, name, shape=None):
        with tf.variable_scope(name) as scope:
        
            initW = tf.truncated_normal_initializer(stddev = self.stddev)
            filter = tf.get_variable(name='filter', shape=shape, initializer=initW)  
            initB = tf.constant_initializer(0.0)
            conv_bias = tf.get_variable(name='bias',shape=shape[3], initializer=initB)
            conv = tf.nn.conv2d(bottom, filter, strides=[1 ,1 ,1 ,1], padding='SAME')
            relu = tf.nn.relu( tf.nn.bias_add(conv, conv_bias) )            
            
            return relu
                
    def _fc_layer(self, bottom, name, shape=None):
        with tf.variable_scope(name) as scope:
    
            weight =self._variable_with_weight_decay(shape, self.stddev, self.wd)
            initB = tf.constant_initializer(0.0)
            bias = tf.get_variable(name='bias',shape=shape[1], initializer=initB)
            fc = tf.nn.bias_add(tf.matmul(bottom, weight), bias)
            if name == 'output' :
                return fc   
            else:
                relu = tf.nn.relu(fc)
                return relu
       
        

    def inference(self, data, train=True, num_class=10, tp=None):
        if tp is not None:
            with tf.name_scope('SPP'):
                print('**********SPP*************')
                self.conv1 = self._conv_layer(data, 'conv1', [5, 5, 1, 6])
                self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],
                    padding='SAME',name='pool1')
                print ('pool1.shape', self.pool1.shape)

                self.conv2 = self._conv_layer(self.pool1, 'conv2', [5, 5, 6, 16])
                self.conv3 = self._conv_layer(self.conv2, 'conv3', [5, 5, 16, 16])
      
                bins = [ 3, 2, 1]
                map_size = self.conv3.get_shape().as_list()[2]
                print('conv3.shape', self.conv3.get_shape())
                sppLayer = SPPLayer(bins, map_size)
                self.sppool = sppLayer.spatial_pyramid_pooling(self.conv3)
            
                numH = self.sppool.get_shape().as_list()[1]
                print('numH', numH)
                self.fc7 = self._fc_layer(self.sppool, 'fc6', shape=[numH, 48])
                if train:
                    self.fc7 = tf.nn.dropout(self.fc7, 0.5, seed=SEED)
                self.output = self._fc_layer(self.fc7, 'output', shape=[48,num_class])
                print('inference')
                return self.output
        else:
            with tf.name_scope('cnn'):
                print('**********CNN*************')
                self.conv1 = self._conv_layer(data, 'conv1', [5,5,1,6])
                self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1,2,2,1],strides=[1,2,2,1],
                    padding='SAME',name='pool1')
                print ('pool1.shape', self.pool1.shape)
                 
                self.conv2 = self._conv_layer(self.pool1, 'conv2', [5,5,6,16])
                self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1,2,2,1],strides=[1,2,2,1],
                    padding='SAME',name='pool2')
                print ('pool2.shape', self.pool2.shape)
                
                self.conv3 = self._conv_layer(self.pool2, 'conv3', [5,5,16,16])
                self.pool3 = tf.nn.max_pool(self.conv3, ksize=[1,2,2,1],strides=[1,2,2,1],
                    padding='SAME',name='pool3')
                print ('pool3.shape', self.pool3.shape)
                
 
                temp1, temp2, temp3 = self.pool3.get_shape().as_list()[1:4]
                self.fc = tf.reshape(self.pool3, [-1, temp1*temp2*temp3])
                print ('fc.shape', self.fc.shape)
                
                self.fc1 = self._fc_layer(self.fc, 'fc1',shape= [temp1*temp2*temp3,32])
                
                if train:
                    self.fc1 = tf.nn.dropout(self.fc1, 0.5, seed=SEED)
                self.output = self._fc_layer(self.fc1, 'output', shape=[32,num_class])
                print('inference')
                return self.output
    
    
    
    def train(self, logits, global_step, label=None):
            self.pred = tf.nn.softmax(logits)
            if label is not None:
                label = tf.cast(label, tf.float32)
                self.entropy_loss = -tf.reduce_mean(label * tf.log(tf.clip_by_value(self.pred,1e-5,1)))  
                self.lr = tf.train.exponential_decay(self.lr, global_step*self.batch_size, self.train_size*self.decay_epochs, 0.95, staircase=True)
                #self.optimizer = tf.train.MomentumOptimizer(self.lr, 0.1).minimize(self.entropy_loss,global_step = global_step)
                self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.entropy_loss)
                correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(label,1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                return (self.entropy_loss, self.accuracy, self.optimizer)
            else:
                return self.pred
    

    def set_lr(self, lr, batch_size, train_size, decay_epochs = 10):
        self.lr = lr
        self.batch_size = batch_size
        self.train_size = train_size
        self.decay_epochs = decay_epochs

    def _variable_with_weight_decay(self, shape, stddev, wd):

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        return var




# coding: utf-8

# In[ ]:


from __future__ import absolute_import                                                                         
from __future__ import division
from __future__ import print_function
from spp_layer import SPPLayer
from packaging_data import *
import numpy as np
import tensorflow as tf
import os
import logging
import math 
import sys


SEED = 1356
stddev = 0.05

class SPPnet:
    def __init__(self, batch_size, path_list, max_steps, num_class=10):
        self.random_weight= False
        self.wd = 5e-4
        self.stddev = 0.05
        self.batch_size = batch_size
        self.path_list = path_list
        self.num_class = num_class
        self.max_steps = max_steps
        self.eval_frequency = 10

    def _conv_layer(self, bottom, name, shape=None):
        with tf.variable_scope(name) as scope:
        
            initW = tf.truncated_normal_initializer(stddev = self.stddev)
            filter = tf.get_variable(name='filter', shape=shape, initializer=initW)  
            initB = tf.constant_initializer(0.0)
            conv_bias = tf.get_variable(name='bias',shape=shape[3], initializer=initB)
            conv = tf.nn.conv2d(bottom, filter, strides=[1 ,1 ,1 ,1], padding='SAME')
            relu = tf.nn.relu( tf.nn.bias_add(conv, conv_bias) )            
            
            return relu, filter, conv_bias
                
    def _fc_layer(self, bottom, name, shape=None):
        with tf.variable_scope(name) as scope:
           
            weight =self._variable_with_weight_decay(shape, self.stddev, self.wd)
            initB = tf.constant_initializer(0.0)
            bias = tf.get_variable(name='bias',shape=shape[1], initializer=initB)
            fc = tf.nn.bias_add(tf.matmul(bottom, weight), bias)
            if name == 'output' :
                return fc, weight, bias  
            else:
                relu = tf.nn.relu(fc)
                return relu, weight, bias
       
    def train_save(self, logits, label):

            if label == 'c':
                self.pred = tf.nn.softmax(logits)
                label = tf.cast(label, tf.float32)
                self.entropy_loss = -tf.reduce_mean(label * tf.log(tf.clip_by_value(self.pred,1e-5,1)))  
                self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.entropy_loss)
                correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(label,1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            else:
                self.loss = tf.reduce_mean(tf.square(logits - self.train_label)) 
                self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
                
            self.saver = tf.train.Saver({v.op.name: v for v in 
                                            [ 
                                              self.conv_w1, self.conv_b1,
                                              self.conv_w2, self.conv_b2,
                                              self.conv_w3, self.conv_b3,
                                              self.conv_w4, self.conv_b4,
                                              self.fc_w5, self.fc_b5,
                                              self.fc_w6, self.fc_b6,
                                              self.fc_wout, self.bout
                                             ]})
    

    def inference(self, rc):
        
        self.g = tf.Graph()
        
        with self.g.as_default():
            
            path_batch = self.path_list.next_batch(self.batch_size)
            self.tickn_len = np.random.randint(200,3500)
            self.data_batch = []
            self.label_batch = []
            for path in path_batch:
                temp1, temp2 = packaging_data(self.tickn_len, path, rc)
                self.data_batch.append(temp1)
                self.label_batch.append(temp2)
            self.data_batch = np.array(self.data_batch)
            self.label_batch = np.array(self.label_batch)
           # print ('data_batch.shape', self.data_batch.shape)
           # print ('label_batch.shape', self.label_batch.shape)
            
            self.train_data = tf.placeholder("float", shape=[None,self.tickn_len ,410 ,1])
            if rc == 'r':
                self.train_label = tf.placeholder("float", shape=[None])
            if re == 'c':
                self.train_label = tf.placeholder("float", shape=[None,1])
                
            self.keep_prob = tf.placeholder("float")
            self.num_class = 10

            with tf.name_scope('SPP'):
                #print('**********SPP*************')
                #print ('train_data.shape', self.train_data.shape)
                self.conv1, self.conv_w1, self.conv_b1 = self._conv_layer(self.train_data, 'conv1', [5, 5, 1, 6])
                self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],
                    padding='SAME',name='pool1')
                #print ('pool1.shape', self.pool1.shape)

                self.conv2, self.conv_w2, self.conv_b2 = self._conv_layer(self.pool1, 'conv2', [5, 5, 6, 16])
                self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],
                    padding='SAME',name='pool2')
                #print ('pool2.shape', self.pool2.shape)
                
                self.conv3, self.conv_w3, self.conv_b3 = self._conv_layer(self.pool2, 'conv3', [5, 5, 16, 16])
                self.pool3 = tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],
                    padding='SAME',name='pool3')
                #print ('pool3.shape', self.pool3.shape)
                
                self.conv4, self.conv_w4, self.conv_b4 = self._conv_layer(self.pool3, 'conv4', [5, 5, 16, 16])
                #print('conv4.shape', self.conv4.get_shape())
      
                bins = [ 8, 6, 4]
                map_size = self.conv4.get_shape().as_list()[1:3]
                sppLayer = SPPLayer(bins, map_size)
                self.sppool = sppLayer.spatial_pyramid_pooling(self.conv4)
                #print ('sppool.shape', self.sppool.shape)
            
                numH = self.sppool.get_shape().as_list()[1]
                #print('numH', numH)
                self.fc5, self.fc_w5, self.fc_b5 = self._fc_layer(self.sppool, 'fc5', shape=[numH, 256])
                self.fc5 = tf.nn.dropout(self.fc5, self.keep_prob, seed=SEED)
                #print ('fc5.shape', self.fc5.shape)
                
                self.fc6, self.fc_w6, self.fc_b6 = self._fc_layer(self.fc5, 'fc6', shape=[256, 64])
                #print ('fc6.shape', self.fc6.shape)
                self.output, self.fc_wout, self.bout = self._fc_layer(self.fc6, 'output', shape=[64, 1])
                
                # train
                self.train_save(self.output, rc)
                #print ('output.shape', self.output.shape)

    
    
    

    

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
    
    def begin(self, rc):
        
        if rc=='c':
            global_step = tf.Variable(0, trainable=False)
            for step in range(self.max_steps):
            #inference
                self.inference(rc)
                #print('*****inference over*******')
                with tf.Session(graph = self.g) as sess:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    if os.path.exists('./spp_model.ckpt'):
                        saver.restore(sess, './spp_model.ckpt')
                
                    self.optimizer.run( feed_dict={ self.train_data:self.data_batch, self.train_label:self.label_batch,
                                                   self.keep_prob: 0.5 } )
                    if step % self.eval_frequency ==0:
                        loss_value=self.entropy_loss.eval(feed_dict={ self.train_data:self.data_batch, self.train_label:self.label_batch,
                                                                     self.keep_prob: 1.0})
                        accu = accuracy.eval(feed_dict={ self.train_data:self.data_batch, self.train_label:self.label_batch,
                                                        self.keep_prob: 1.0})
                        print('train loss: ', loss_value) 
                        print('train accu: ', accu)
                    self.saver.save(sess, './spp_model.ckpt')
                sess.close()
                del(sess)
                
        if rc=='r':
            global_step = tf.Variable(0, trainable=False)
            for step in range(self.max_steps):
            #inference
                self.inference(rc)
                #print('*****inference over*******')
                with tf.Session(graph = self.g) as sess:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    if os.path.exists('./spp_model.ckpt'):
                        saver.restore(sess, './spp_model.ckpt')
                    self.optimizer.run( feed_dict={ self.train_data:self.data_batch, self.train_label:self.label_batch,
                                                   self.keep_prob:0.5 } )
                    print('step:',step)
                    if step % self.eval_frequency ==0:
                        loss_value=self.loss.eval(feed_dict={ self.train_data:self.data_batch, self.train_label:self.label_batch,
                                                             self.keep_prob: 1.0})
                        print('train loss: ', loss_value) 
                    self.saver.save(sess, './spp_model.ckpt')
                sess.close()
                del(sess)





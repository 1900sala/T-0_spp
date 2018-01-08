from spp_net import *
from spp_layer import *
import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# path_list = np.load('/home/user001/spp_net/T-0_pre_data/path_list.npy')



def packaging_data(tick_len, path, rc):
    print (tick_len, path)
    data = pd.read_pickle(path)
    start_ticktime = datetime.datetime.timestamp(data.iloc[1].exchangeTime + datetime.timedelta(minutes = 2))
    threshold_ticktime = datetime.datetime.timestamp(data.iloc[1].exchangeTime + datetime.timedelta(minutes = 10))
    open_price = data.iloc[0].latestPrice
    data['exchangeTime'] = data['exchangeTime'].apply(lambda x: int(datetime.datetime.timestamp(x)))
    str_l=['ask', 'bid']
    
    for col in str_l:
        for i in range(1,6):
            temp_col = col + 'Px' + str(i)
            temp_vol = col + 'Vol' + str(i)
            data[temp_col] = (data[temp_col]/open_price-1)*2000
            data[temp_col] = data[temp_col].apply(lambda x: 999 if int(x)<-205 else int(x))
            data[temp_col] = data[temp_col].apply(lambda x: 999 if x>205 else x)
            #print (data[temp_col].min())
            #print (data[temp_col].max())  
            
    images = np.zeros((tick_len,410))
    tag = 0
    temp = 0
    l_time = 0
    for index,row in data.iterrows():

        if row['exchangeTime']<start_ticktime:
            continue
            
        elif row['exchangeTime']>= start_ticktime and tag<tick_len :
            last_price = row.latestPrice
            if temp == 0:
                l_time = row['exchangeTime']
                temp = 1
#             print ('00000000', l_time, row.exchangeTime, start_ticktime, start_ticktime+tick_len*3)
            if l_time == row.exchangeTime:    
                for col in str_l:
                    for i in range(1,6):
                        temp_col = col + 'Px' + str(i)
                        temp_vol = col + 'Vol' + str(i)
                        #print (index, row[temp_col])
                        if row[temp_col]==999:
                            continue
                        y = row[temp_col]+205
                        images[tag, y] = row[temp_vol]
#                         print ('11111111',tag, images[tag, y])
                tag = tag + 1
                l_time = l_time+3
#                 print ('11111111', tag, l_time)
            else:
                while l_time < row.exchangeTime and tag < tick_len:
                    l_time = l_time+3
                    images[tag, :] = images[tag-1, :]
                    tag = tag + 1
                if tag == tick_len:
                    continue
                
                for col in str_l:
                    for i in range(1,6):
                        temp_col = col + 'Px' + str(i)
                        temp_vol = col + 'Vol' + str(i)
                        if row[temp_col]==999:
                            continue
                        y = row[temp_col]+205
                        images[tag, y] = row[temp_vol]
#                         print ('22222222222',tag, images[tag, y])
                tag = tag + 1
                l_time = l_time+3
#                 print ('22222222222',tag, l_time)
            
        elif row['exchangeTime']>start_ticktime+tick_len*3+180:
            label = (row.latestPrice - last_price) / last_price
            if rc = 'c':
                if label>0.003:
                    label = [1]
                else:
                    label = [0]
            temp = np.array(images)
            
            break
         
    images = np.reshape (temp, (temp.shape[0],temp.shape[1],1))
    return images,label
            
    
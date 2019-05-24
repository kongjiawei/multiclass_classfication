# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:17:51 2019

@author: 孔嘉伟
"""

from keras.models import load_model
import pandas as pd
import numpy as np
#from Classify_Training import 

Prefix = 'exception_all'
FileName = Prefix + '.csv'

'''读取测试集的数据'''
DataX = []
DataY = []
DataX = pd.read_csv(FileName, usecols = [1,2,3,4,5])
DataY = pd.read_csv(FileName, usecols = [6])
DataX = DataX.values.tolist() 
DataY = DataY.values.tolist()
DataX = np.array(DataX)
DataY = np.array(DataY)

'''载入模型并预测'''
model = load_model(Prefix + '.h5')
predict = model.predict(DataX)

'''计算精确度'''
predict_to_value = np.argmax(predict,axis = 1)
count_right = 0
count_sum = len(predict_to_value)
for i in range(count_sum):
    if(predict_to_value[i] == DataY[i][0]):#DataY是嵌套的
        count_right = count_right + 1

accuracy = count_right / count_sum
print('accuracy:',accuracy)







 
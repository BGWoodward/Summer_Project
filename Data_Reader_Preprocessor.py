# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 16:55:54 2014

@author: Ben Woodward
"""

import csv
import numpy as np
from sklearn import svm, grid_search, cross_validation, preprocessing

training_path = 'S:/Repositories/Kaggle_Data/training.csv'
ifile = open(training_path,'r')
reader = csv.reader(ifile)
A = reader.next()

class Training_Element:
    def __init__(self,row,label):
        self.ID = row[0]
        self.features = row[1:31]
        self.weight = row[31]
        self.label = label

'''
Class to make a list of training points
'''        
class Training_Cache:
    def __init__(self):
        self.element_list = []
    def add_element(self,values,label):
        self.element_list.append(Training_Element(values,label))

Training_Data = Training_Cache()

for row in reader: #iterate through every row
    B = []
    for i in row[:-1]:
        B = np.append(B,float(i))
    Training_Data.add_element(B,row[-1])

Weights = []
Weights.append(Training_Data.element_list[0].weight)
ID = []
ID.append(Training_Data.element_list[0].ID)
Signal_Indices = []
Background_Indices =[]
Features = Training_Data.element_list[0].features
if Training_Data.element_list[0].label == 's':
    Signal_Indices.append(0)
else:
    Background_Indices.append(0)

for i,element in enumerate(Training_Data.element_list[1:]):
    Features = np.vstack([Features,element.features])
    Weights.append(element.weight)
    ID.append(element.ID)
    if element.label == 's':
        Signal_Indices.append(i)
    else:
        Background_Indices.append(i)

scaler = preprocessing.StandardScaler().fit(Features)
Scaled_Data = scaler.transform(Features)

#Save off the files so we don't have to recalculate them again, as it is a
#time consuming process

np.savetxt('Scaled_Features.csv',Scaled_Data,delimiter=',')
np.save('Scaled_Features_npy',Scaled_Data)
np.save('Signal_Indices',Signal_Indices)
np.save('Background_Indices',Background_Indices)
np.save('Weights',Weights)
np.save('IDs',ID)
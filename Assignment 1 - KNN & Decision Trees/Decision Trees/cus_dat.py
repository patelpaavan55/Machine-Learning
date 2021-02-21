# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 22:44:12 2019

@author: Paavan Patel
"""

import numpy as np
import csv

data_path = 'C:/Users/Paavan Patel/Desktop/customer_labels.csv'
with open(data_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    # get header from first row
    headers = next(reader)
    # get all the rows as a list
    data = list(reader)
#    print("DT: ",data)
    data = np.array(data)

print(headers)
print(data.shape)
print(data[:3])
print("----------------------------------")
new=data.tolist()
print(new)
#for k in new:
#    print(k)


#print(data)

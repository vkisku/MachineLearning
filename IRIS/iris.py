# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:26:01 2017

@author: vkisku
"""
from sklearn import tree
import numpy as np
fo = open("IRIS.csv", "r")
print(fo.name)

str = fo.readlines()
l=len(str);
i=0;     
features=["node"]*l
#$features[1]=str[0].split(',');   
#print(str[0][1]); 
temp=[];   
labels=["node"]*l;
#print(temp);     
while(l>0):
    temp=str[i].split(',')
    labels[i]=temp[8].strip("\n")
    features[i]=[float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4]), float(temp[5]),float(temp[6]),float(temp[7])];
    l-=1
    i+=1
#print(len(str));
fo.close()
print("Training of the data Completed Now time to test it")
f=open("samples.csv", "r")
new_str=f.readlines()
l=len(new_str)
l1=1
#print(new_str)
testing_set=["node"]*l
#new_temp=[]
i=0            
while(l>0):
    new_temp=new_str[i].split(',');
                    
    n=[float(new_temp[0]), float(new_temp[1]), float(new_temp[2]), float(new_temp[3]), float(new_temp[4]), float(new_temp[5]),float(new_temp[6]),float(new_temp[7])]                
    #n=np.array(n).reshape((-1,1))
    n = np.array(n).reshape((1, -1))
    #print(n)
    testing_set[i]=n
    #print(testing_set)
    l-=1
    i+=1            
f.close()        
#print(features[0]);
#labels=["setosa","versicolor","virginica"]
#features = features.reshape(1,-1) 

clf=tree.DecisionTreeClassifier()

clf=clf.fit(features,labels)
#print(labels)
#print(testing_set[1]) 
  
#temp=[6.5,0.611111111,2.8,0.333333333,4.6,0.610169492,1.5,0.583333333]
#temp = np.array(temp).reshape((1, -1))
#temp =np.reshape(temp, (4, 2)).T
#print(temp)
j=0
l1=10
while(l1>0):
    #print(testing_set[j])
    print(clf.predict(testing_set[j])[0]);
    j+=1
    l1-=1                             
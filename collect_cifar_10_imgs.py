import os           
import csv          
import pandas as pd  

path = os.getcwd()  

train_path = "/Users/maburidi/Downloads/cifar-10/train"
train_2_path ="/Users/maburidi/Downloads/cifar-10/"
labels_path = "/Users/maburidi/Downloads/cifar-10/trainLabels.csv"

imgs_ids = os.listdir(train_path)

df = pd.read_csv(labels_path)
classes = ["airplane", "automobile", "bird", "cat", "deer" , "dog", "frog", "horse" ,"ship" , "truck"]

labels_grs = [] 
for i in range(len(classes)):
    labels_grs.append([str(classes[i]) ,list( df[df.label == str(classes[i])].id)])
   

# Copy files 

import shutil
for i in range(len(classes)):
    for img in range(5000):
        shutil.copyfile( train_path +"/"+str(labels_grs[i][1][img]) + '.png', train_2_path+ "img_grps"+'/'+ str(classes[i]) +'/' +  str(labels_grs[i][1][img]) + '.png')

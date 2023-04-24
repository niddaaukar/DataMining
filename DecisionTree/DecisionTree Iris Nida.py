#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import library numpy, pandas dan scikit-learn
import numpy as np
import pandas as pd
from sklearn import tree


# In[2]:


#Membaca Dataset dari File ke Pandas dataFrame
irisDataset = pd.read_csv('Dataset Iris.csv', delimiter=';', header=0)


# In[3]:


irisDataset.head()


# In[4]:


#Mengubah kelas (kolom "Species") dari String ke Unique-Integer
irisDataset["Species"] = pd.factorize(irisDataset.Species)[0]


# In[5]:


irisDataset.head()


# In[6]:


print(irisDataset)


# In[7]:


#Menghapus kolom "Id"
irisDataset = irisDataset.drop(labels="Id", axis=1)


# In[8]:


print(irisDataset)


# In[9]:


#Mengubah dataFrame ke array Numpy
irisDataset = irisDataset.to_numpy()


# In[10]:


print(irisDataset)


# In[11]:


#Membagi Dataset => 80 baris data untuk training dan 20 baris data untuk testing
dataTraining = np.concatenate((irisDataset[0:40, :], irisDataset[50:90, :]), 
                              axis=0)
dataTesting = np.concatenate((irisDataset[40:50, :], irisDataset[90:100, :]), 
                             axis=0)


# In[12]:


print(dataTesting)
len(dataTesting)


# In[13]:


#Memecah Dataset ke Input dan Label
inputTraining = dataTraining[:, 0:4]
inputTesting = dataTesting[:, 0:4]
labelTraining = dataTraining[:, 4]
labelTesting = dataTesting[:, 4]


# In[14]:


#Mendefinisikan Decision Tree Classifier
model = tree.DecisionTreeClassifier()


# In[15]:


#Mentraining Model
model = model.fit(inputTraining, labelTraining)


# In[16]:


#Memprediksi Input Data Testing
hasilPrediksi = model.predict(inputTesting)
print("Label Sebenarnya : ", labelTesting)
print("Hasil Prediksi : ", hasilPrediksi)


# In[17]:


#Menghitung Akurasi
prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()
print("Prediksi Benar :", prediksiBenar, "data")
print("Prediksi Salah :", prediksiSalah, "data")
print("Akurasi :", prediksiBenar/(prediksiBenar+prediksiSalah) * 100, "%")


# In[ ]:





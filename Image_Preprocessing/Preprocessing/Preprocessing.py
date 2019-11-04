import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

## TRAINING DATA

pm = 'D:/Python Scripts/DR/train/mild'
b=1
path='D:/Python Scripts/DR/Training/mild/'
for i in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(path,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img =cv2.resize(img,(500,500))
    i = cv2.addWeighted ( img,5, cv2.GaussianBlur( img , (0,0) , 40) ,-5 ,128)
    a = path+str(b)+ '.png'
    b=b+1
    plt.imsave(a,i)

pm = 'D:/Python Scripts/DR/train/severe'
b=1
path='D:/Python Scripts/DR/Training/Severe/'
for i in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(path,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img =cv2.resize(img,(500,500))
    i = cv2.addWeighted ( img,5, cv2.GaussianBlur( img , (0,0) , 40) ,-5 ,128)
    a = path+str(b)+ '.png'
    b=b+1
    plt.imsave(a,i)

pm = 'D:/Python Scripts/DR/train/moderate'
b=1
path='D:/Python Scripts/DR/Training/moderate/'
for i in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(path,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img =cv2.resize(img,(500,500))
    i = cv2.addWeighted ( img,5, cv2.GaussianBlur( img , (0,0) , 40) ,-5 ,128)
    a = path+str(b)+ '.png'
    b=b+1
    plt.imsave(a,i)

pm = 'D:/Python Scripts/DR/train/No'
b=1
path='D:/Python Scripts/DR/Training/No/'
for i in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(path,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img =cv2.resize(img,(500,500))
    i = cv2.addWeighted ( img,5, cv2.GaussianBlur( img , (0,0) , 40) ,-5 ,128)
    a = path+str(b)+ '.png'
    b=b+1
    plt.imsave(a,i)
    
pm = 'D:/Python Scripts/DR/train/pdr'
b=1
path='D:/Python Scripts/DR/Training/pdr/'
for i in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(path,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img =cv2.resize(img,(500,500))
    i = cv2.addWeighted ( img,5, cv2.GaussianBlur( img , (0,0) , 40) ,-5 ,128)
    a = path+str(b)+ '.png'
    b=b+1
    plt.imsave(a,i)



## VALIDATION DATA

pm = 'D:/Python Scripts/DR/Val/mild'
b=1
path='D:/Python Scripts/DR/Validation/mild/'
for i in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(path,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img =cv2.resize(img,(500,500))
    i = cv2.addWeighted ( img,5, cv2.GaussianBlur( img , (0,0) , 40) ,-5 ,128)
    a = path+str(b)+ '.png'
    b=b+1
    plt.imsave(a,i)

pm = 'D:/Python Scripts/DR/Val/severe'
b=1
path='D:/Python Scripts/DR/Validation/Severe/'
for i in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(path,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img =cv2.resize(img,(500,500))
    i = cv2.addWeighted ( img,5, cv2.GaussianBlur( img , (0,0) , 40) ,-5 ,128)
    a = path+str(b)+ '.png'
    b=b+1
    plt.imsave(a,i)

pm = 'D:/Python Scripts/DR/Val/moderate'
b=1
path='D:/Python Scripts/DR/Validation/moderate/'
for i in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(path,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img =cv2.resize(img,(500,500))
    i = cv2.addWeighted ( img,5, cv2.GaussianBlur( img , (0,0) , 40) ,-5 ,128)
    a = path+str(b)+ '.png'
    b=b+1
    plt.imsave(a,i)

pm = 'D:/Python Scripts/DR/Val/No'
b=1
path='D:/Python Scripts/DR/Validation/No/'
for i in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(path,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img =cv2.resize(img,(500,500))
    i = cv2.addWeighted ( img,5, cv2.GaussianBlur( img , (0,0) , 40) ,-5 ,128)
    a = path+str(b)+ '.png'
    b=b+1
    plt.imsave(a,i)
    
pm = 'D:/Python Scripts/DR/Val/pdr'
b=1
path='D:/Python Scripts/DR/Validation/pdr/'
for i in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(path,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img =cv2.resize(img,(500,500))
    i = cv2.addWeighted ( img,5, cv2.GaussianBlur( img , (0,0) , 40) ,-5 ,128)
    a = path+str(b)+ '.png'
    b=b+1
    plt.imsave(a,i)

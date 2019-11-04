import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

## TRAINING DATA

pm = 'D:\Python Scripts/DR/train/mild'
b=1
path='D:\Python Scripts/DR/Training/Mild/'
for img in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(pm,img))
    img  = cv2.resize(img,(500,500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6))
    cl1 = clahe.apply(img)
    p = path+str(b)+'.jpeg'
    x = cv2.imwrite(p,cl1)
    b = b+1

pm = 'D:\Python Scripts/DR/train/severe'
b=1
path='D:\Python Scripts/DR/Training/Severe/'
for img in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(pm,img))
    img  = cv2.resize(img,(500,500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6))
    cl1 = clahe.apply(img)
    p = path+str(b)+'.jpeg'
    x = cv2.imwrite(p,cl1)
    b = b+1

pm = 'D:\Python Scripts/DR/train/moderate'
b=1
path='D:\Python Scripts/DR/Training/moderate/'
for img in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(pm,img))
    img  = cv2.resize(img,(500,500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6))
    cl1 = clahe.apply(img)
    p = path+str(b)+'.jpeg'
    x = cv2.imwrite(p,cl1)
    b = b+1

pm = 'D:\Python Scripts/DR/train/No'
b=1
path='D:\Python Scripts/DR/Training/No/'
for img in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(pm,img))
    img  = cv2.resize(img,(500,500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6))
    cl1 = clahe.apply(img)
    p = path+str(b)+'.jpeg'
    x = cv2.imwrite(p,cl1)
    b = b+1

pm = 'D:\Python Scripts/DR/train/pdr'
b=1
path='D:\Python Scripts/DR/Training/pdr/'
for img in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(pm,img))
    img  = cv2.resize(img,(500,500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6))
    cl1 = clahe.apply(img)
    p = path+str(b)+'.jpeg'
    x = cv2.imwrite(p,cl1)
    b = b+1



## VALIDATION DATA

pm = 'D:\Python Scripts/DR/Val/mild'
b=1
path='D:\Python Scripts/DR/Validation/Mild/'
for img in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(pm,img))
    img  = cv2.resize(img,(500,500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6))
    cl1 = clahe.apply(img)
    p = path+str(b)+'.jpeg'
    x = cv2.imwrite(p,cl1)
    b = b+1

pm = 'D:\Python Scripts/DR/Val/severe'
b=1
path='D:\Python Scripts/DR/Validation/Severe/'
for img in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(pm,img))
    img  = cv2.resize(img,(500,500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6))
    cl1 = clahe.apply(img)
    p = path+str(b)+'.jpeg'
    x = cv2.imwrite(p,cl1)
    b = b+1

pm = 'D:\Python Scripts/DR/Val/moderate'
b=1
path='D:\Python Scripts/DR/Validation/moderate/'
for img in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(pm,img))
    img  = cv2.resize(img,(500,500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6))
    cl1 = clahe.apply(img)
    p = path+str(b)+'.jpeg'
    x = cv2.imwrite(p,cl1)
    b = b+1

pm = 'D:\Python Scripts/DR/Val/No'
b=1
path='D:\Python Scripts/DR/Validation/No/'
for img in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(pm,img))
    img  = cv2.resize(img,(500,500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6))
    cl1 = clahe.apply(img)
    p = path+str(b)+'.jpeg'
    x = cv2.imwrite(p,cl1)
    b = b+1

pm = 'D:\Python Scripts/DR/Val/pdr'
b=1
path='D:\Python Scripts/DR/Validation/pdr/'
for img in tqdm(os.listdir(pm)):
    img = cv2.imread(os.path.join(pm,img))
    img  = cv2.resize(img,(500,500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6))
    cl1 = clahe.apply(img)
    p = path+str(b)+'.jpeg'
    x = cv2.imwrite(p,cl1)
    b = b+1

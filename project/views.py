#!/usr/bin/python
# -*- coding: utf-8 -*-
def warn(*args, **kwargs):
    pass

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.db.models import Q
from django.core.files.storage import FileSystemStorage

#from .models import *

def home(request):
    return render(request,'index.html')


def upload_classify(request):
    #default="/media/dog.jpeg"
    img_path = "/media/dog.jpeg"
    import requests
    result='Input image for image classification'
    img_des= "/static/imageClassification.webp"
    #arr=[]
    if request.method == "POST":
        uploaded_file = request.FILES['document']
        fs=FileSystemStorage()
        fs.save(uploaded_file.name,uploaded_file)
        
        #from __future__ import division, print_function
        import sys
        import os
        import glob
        import re
        import numpy as np
        import tensorflow as tf
        import shutil, os
        from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image
        model = load_model('media/ress.h5')

        #img_path = input("Entre your image : ")
        img_path = "media/"+uploaded_file.name
        img_des = "static/"+uploaded_file.name
        shutil.copy(img_path, img_des)
        
        img = image.load_img(img_path, target_size=(224, 224))
        # Preprocessing the image
        x = image.img_to_array(img)
        # x = np.true_divide(x, 255)
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        np.argmax(preds)
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result1=[]
        #result1=append(str(asfjb))
        if(str(pred_class[0][0][1])):
            result = str(pred_class[0][0][1])
        
        
        #img= "/media/"+uploaded_file.name
        
    return render(request,'upload_classify.html',{'result':result,'f5':img_des})

def upload_csvmodel(request):
    import requests
    import logging

    logger = logging.getLogger("log_hritik")
    #logger.info("Whatever to log")
    #img_path = "media/dog.jpeg"
    result='Input Image'
    if request.method == "POST":
        uploaded_file = request.FILES['document']
        fs=FileSystemStorage()
        fs.save(uploaded_file.name,uploaded_file)

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import keras
        from keras.utils import np_utils
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation,  Flatten, Conv2D, MaxPooling2D, BatchNormalization
        from keras import backend as K
        from keras.preprocessing.image import ImageDataGenerator
        from keras.models import load_model
        import os 
        import sys
        from os import listdir
        from os.path import isfile, join
        import cv2
        from glob import glob
        from keras.optimizers import Adam

        df = pd.read_csv('media/inputtype.csv') 

        #df = df.drop(['Unnamed: 0'], axis = 1)

        model= load_model('media/csvmodel.h5')

        #testing = ''
        testing = "media/"+uploaded_file.name


        import cv2
        img = cv2.imread(testing)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        img_rgb = np.array(img_rgb)
        img_rgb.shape


        img_rgb = img_rgb.reshape(1,img_rgb.shape[0],img_rgb.shape[1],3)

        img_rgb = np.array(img_rgb)/255


        pred = model.predict(img_rgb)
        pred = np.argmax(pred)
        pred
        logger.info(pred)


        #print(df.iloc[pred,1])
        result=(df.iloc[pred,1])
        logger.info(result)
        #print(result)

    return render(request,'upload_csvmodel.html',{'result':result,'f5':'/static/img.png'})
    #return render(request,'upload_csvmodel.html',{'f5':'/static/img.png')


def upload_OD(request):
    default="/static/obejctDetection.png"
    import requests
    arr=[]
    if request.method == "POST":
        uploaded_file = request.FILES['document']
        fs=FileSystemStorage()
        fs.save(uploaded_file.name,uploaded_file)
        #f = open("media/"+uploaded_file.name, "r")
        #print(f.read())
        #storage=[]
        #num=0
        import cv2 
        from imageai.Detection import ObjectDetection
        import os
        #from IPython.core.getipython import get_ipython

        execution_path = os.getcwd()

        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath( os.path.join(execution_path , "media/resnet50_coco_best_v2.0.1.h5"))
        detector.loadModel()

        #detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "/content/drive/My Drive/minor2final/dog_cycle.jpeg"), output_image_path=os.path.join("/content/drive/My Drive/minor2final" , "imagenew.jpg"))

        detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "media/"+uploaded_file.name), output_image_path=os.path.join(execution_path , "static/"+"1"+uploaded_file.name))
        
        for eachObject in detections:
            #print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
            arr.append(str(eachObject["name"]) +  " : " + str(eachObject["percentage_probability"]))
        print (arr)


        #img = cv2.imread('imagenew.jpg')
        #img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        #plt.imshow(img_rgb)



        default="/static/"+"1"+uploaded_file.name

       
            
    return render(request,'upload_OD.html',{'result':arr,'f5':default})

def upload_segmentation(request):
    image = ""
    import requests
    img_des= 'static/segement.png'
    arr='Input image for Image segemenation'

    if request.method == "POST":
        uploaded_file = request.FILES['document']
        fs=FileSystemStorage()
        fs.save(uploaded_file.name,uploaded_file)

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        #%matplotlib inline
        import cv2
        image = 'media/'+uploaded_file.name
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #plt.imshow(img)
        img.shape
        img.shape[0]
        l = img.reshape(img.shape[0]*img.shape[1], 3)
        l.shape
        from sklearn.cluster import KMeans
        #from yellowbrick.cluster import KElbowVisualizer

        #model = KMeans(random_state=0)

        #visualizer = KElbowVisualizer(model, k=(2,6), metric='silhouette', timings=False)

        #visualizer.fit(l)    
        #visualizer.poof()      # hence we will take k = 3
        kmeans = KMeans(n_clusters = 3)
        g = kmeans.fit(l)
        pic2show = g.cluster_centers_[g.labels_]
        pic2show = pic2show.reshape(img.shape[0],img.shape[1],3)
        finalpic = pic2show/255
        #plt.imshow(finalpic)
        img_des='static/'+uploaded_file.name
        cv2.imwrite(img_des, pic2show)
    return render(request,'upload_segmentation.html',{'result':arr,'f5':img_des})




def upload_foldermodel(request):
    import requests
    result= 'input image'
    testing= '/media/dog.jpeg'
    if request.method == "POST":
        uploaded_file = request.FILES['document']
        fs=FileSystemStorage()
        fs.save(uploaded_file.name,uploaded_file)

        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        import keras
        from keras.utils import np_utils
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation,  Flatten, Conv2D, MaxPooling2D, BatchNormalization
        from keras import backend as K
        from keras.models import load_model
        import os 
        import sys
        from os import listdir
        from os.path import isfile, join
        import cv2
        from glob import glob
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.models import load_model, Sequential, Model
        from tensorflow.keras.applications import InceptionV3
        from tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D, Dense, Activation, Flatten
        from tensorflow.keras.optimizers import Adam, SGD


        l = {'Apple___Apple_scab': 0,
            'Apple___Black_rot': 1,
            'Apple___Cedar_apple_rust': 2,
            'Apple___healthy': 3,
            'Blueberry___healthy': 4,
            'Cherry_(including_sour)___Powdery_mildew': 5,
            'Cherry_(including_sour)___healthy': 6,
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7,
            'Corn_(maize)__Common_rust': 8,
            'Corn_(maize)___Northern_Leaf_Blight': 9,
            'Corn_(maize)___healthy': 10,
            'Grape___Black_rot': 11,
            'Grape__Esca(Black_Measles)': 12,
            'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': 13,
            'Grape___healthy': 14,
            'Orange__Haunglongbing(Citrus_greening)': 15,
            'Peach___Bacterial_spot': 16,
            'Peach___healthy': 17,
            'Pepper,bell__Bacterial_spot': 18,
            'Pepper,bell__healthy': 19,
            'Potato___Early_blight': 20,
            'Potato___Late_blight': 21,
            'Potato___healthy': 22,
            'Raspberry___healthy': 23,
            'Soybean___healthy': 24,
            'Squash___Powdery_mildew': 25,
            'Strawberry___Leaf_scorch': 26,
            'Strawberry___healthy': 27,
            'Tomato___Bacterial_spot': 28,
            'Tomato___Early_blight': 29,
            'Tomato___Late_blight': 30,
            'Tomato___Leaf_Mold': 31,
            'Tomato___Septoria_leaf_spot': 32,
            'Tomato___Spider_mites Two-spotted_spider_mite': 33,
            'Tomato___Target_Spot': 34,
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35,
            'Tomato___Tomato_mosaic_virus': 36,
            'Tomato___healthy': 37}

        testing = 'media/'+uploaded_file.name
        #testing='/content/drive/My Drive/Colab Notebooks (1)/Minor2/Codes/Classification_model/testing/Grape___healthy.JPG'
        #testing = str(input('/content/drive/My Drive/Colab Notebooks (1)/Minor2/Codes/Classification_model/testing/Grape___healthy.JPG'))


        model1 = load_model('media/modelleaf.h5')



        import cv2
        img = cv2.imread(testing)
        #plt.imshow(img)


        img = np.array(img)/255


        img = img.reshape(1,256,256,3)


        key_list = list(l.keys())
        val_list = list(l.values()) 


        pred = model1.predict(img)
        pred = pred.argmax(axis=1)[0]
        pred


        final_output = key_list[val_list.index(pred)] 


        result=final_output

    return render(request,'upload_foldermodel.html',{'result':result,'f5':'static/dattgen.png'})


def error_404_view(request, exception):
    return render(request,'404.html')

def index(request):
    try:
        return render(request, 'index.html')
    except:
        return render(request, '404.html')


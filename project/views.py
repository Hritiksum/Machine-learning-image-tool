#!/usr/bin/python
# -*- coding: utf-8 -*-
def warn(*args, **kwargs):
    pass

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.db.models import Q
from django.core.files.storage import FileSystemStorage

#from .models import *

def upload_classify(request):
    #default="/media/dog.jpeg"
    img_path = "/media/dog.jpeg"
    import requests
    result='input image'
    img_des= "/media/dog.jpeg"
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



def upload_OD(request):
    default="/media/dog_cycle.jpeg"
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
    img_des= ''

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
    return render(request,'upload_segmentation.html',{'f5':img_des})



def error_404_view(request, exception):
    return render(request,'404.html')

def index(request):
    try:
        return render(request, 'index.html')
    except:
        return render(request, '404.html')


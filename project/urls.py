from django.urls import path
from . import views

urlpatterns = [
    path('',views.home),
    path('objectdetection',views.upload_OD,name="upload_OD"),
    path('classify',views.upload_classify,name="upload_classify"),
    path('segmentation',views.upload_segmentation,name="upload_segmentation"),
    path('csvmodel',views.upload_csvmodel,name="upload_csvmodel"),
    path('foldermodel',views.upload_foldermodel,name="upload_foldermodel"),
]


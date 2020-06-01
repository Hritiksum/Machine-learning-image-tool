from django.urls import path
from . import views

urlpatterns = [
    path('',views.upload_OD,name="upload_OD"),
    path('classify',views.upload_classify,name="upload_classify"),
    path('segmentation',views.upload_segmentation,name="upload_segmentation"),
]


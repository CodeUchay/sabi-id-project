from django.urls import path
from . import views

app_name = 'sabiapp'

urlpatterns = [
    path('', views.upload_image, name='upload_image'),
    path('result/<int:image_id>/', views.result, name='result'),
    path('download/<int:image_id>/', views.download_image, name='download_image'),
]

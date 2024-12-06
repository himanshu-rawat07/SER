from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('save_file/', views.save_file, name='save_file'),    
    path('classify/', views.classify, name='classify'),
    path('about_us/', views.about_us, name='about_us'),
    # path('project_info/', views.project_info, name='project_info'),
]
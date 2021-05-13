"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.shortcuts import HttpResponse, render
import json
from PIL import Image
import numpy as np
import joblib
import torch
from autoencoder import net_2

def PCA(request):
    if request.method == 'POST':
        name = request.POST.get('username')
        image = request.FILES.get('avatar')
        img = Image.open(image)
        img = img.resize((440, 440))
        imgnp = np.array(img)

        imgnp =imgnp / 255.0
        imgnp = imgnp.reshape(1, -1)

        model_file = "./model/pca.pkl"
        with open(model_file, 'rb') as infile:
            loaded = joblib.load(infile)

            model = loaded['model']
            pca1 = loaded['pca_fit']

            x_test1 = pca1.transform(imgnp)
            y = model.predict(x_test1)

            print(x_test1)
            result = {"probability": y.item()}
            return HttpResponse(json.dumps(result), content_type="application/json")


    return render(request, 'upload.html')

def AUTOENCODER(request):
    if request.method == 'POST':
        name = request.POST.get('username')
        image = request.FILES.get('avatar')
        img = Image.open(image)
        img = img.resize((440, 440))
        imgnp = np.array(img)
        imgnp =imgnp / 255.0

        model_file = "./model/autoEncoder_dense.pkl"
        net_dict = torch.load(model_file, map_location=torch.device('cpu'))
        net = net_2()
        net.load_state_dict(net_dict)
        imgnp = torch.from_numpy(imgnp.transpose((2, 0, 1)))
        imgnp = torch.unsqueeze(imgnp, dim=0)
        y_pred = net(imgnp.to(torch.float32))
        print(torch.max(y_pred, dim=1))
        y_pred = torch.max(y_pred, dim=1).indices
        result = {"probability": y_pred.numpy().item()}
        result = {"probability":result}
        return HttpResponse(json.dumps(result), content_type="application/json")


    return render(request, 'upload.html')

def upload(request):
    if request.method == 'POST':
        name = request.POST.get('username')
        image = request.FILES.get('avatar')
        img =Image.open(image)
        imgnp = np.array(img)
        imgnp = imgnp.reshape(-1)
        print(imgnp)
        img.show()
        return HttpResponse('ok')
    return render(request, 'upload.html')

def Performance(requset):
    result = {
        'pca': 0.54,
        'autoencoder':0.90,
        'kmeans':0.64
    }

    return HttpResponse(json.dumps(result), content_type="application/json")
urlpatterns = [
    path('pca', AUTOENCODER),
    path('autoencoder', AUTOENCODER),
    path('performance', Performance)
]

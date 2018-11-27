from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from keras.models import model_from_json
import cv2
import os
import urllib
import numpy as np
from skimage import io

# Create your views here.
from classifier.forms import ImageForm


def index(request):
    if request.method == 'POST':
        form = ImageForm(request.POST)
        if form.is_valid():
            print(form.cleaned_data['image_url'])
            print(type(form.cleaned_data['image_url']))
            image = url_to_image(form.cleaned_data['image_url'])
            image = resize_image(image)
            if predict(image)[0][0] == 1:
                return render(request, 'classifier/result.html', {
                    'type': 'Chó',
                    'image_url': form.cleaned_data['image_url']
                })
            else:
                return render(request, 'classifier/result.html', {
                    'type': 'Mèo',
                    'image_url': form.cleaned_data['image_url']
                })
    else:
        form = ImageForm()
    return render(request, 'classifier/index.html', {'form': form})


def url_to_image(url):
    image = io.imread(url)
    # return the image
    return image


def resize_image(image):
    normalize_image = cv2.resize(image, (64,64))
    normalize_image = normalize_image.reshape(1, 64, 64, 3)
    return normalize_image


def load_model():
    json_file = open('F:\Project\Machine Learning\Cat Dog\App\classifier\dlmodel\model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("F:\Project\Machine Learning\Cat Dog\App\classifier\dlmodel\model.h5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return load_model


def predict(image):
    json_file = open('F:\Project\Machine Learning\Cat Dog\App\classifier\dlmodel\model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("F:\Project\Machine Learning\Cat Dog\App\classifier\dlmodel\model.h5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return loaded_model.predict_classes(image)

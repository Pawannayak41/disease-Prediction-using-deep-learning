from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from keras.models import Sequential
from glob import glob
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import os

app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/tuberculosis",methods=['GET', 'POST'])
def tuberculosisPage():
    return render_template('tuberculosis.html')


@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img / 255.0
                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('pneumonia.html', message = message)
    return render_template('pneumonia_predict.html', pred = pred)

@app.route("/tuberculosispredict", methods = ['POST', 'GET'])
def tuberculosispredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((224,224))
                img = np.asarray(img)
                img = img.reshape((1,224,224,3))
                img = img / 255.0
               #img = np.expand_dims(img, axis = 0)
               # img_data = preprocess_input(img)
                model = load_model("models/tb.h5")
                pred = np.argmax(model.predict(img) , axis = 1)
        except:
            message = "Please upload an Image"
            return render_template('tuberculosis.html', message = message)
    return render_template('tuberculosis_predict.html', pred = pred)
            


if __name__ == '__main__':
	app.run(debug = True)

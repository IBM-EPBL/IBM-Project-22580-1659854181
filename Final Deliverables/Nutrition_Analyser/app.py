from flask import Flask,render_template,url_for,request
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img, img_to_array 
import requests
from bs4 import BeautifulSoup

print(tf.__version__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

model = load_model('fruits.h5')

class_name = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}


app = Flask(__name__)

def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?q=healthifyme+nutrition+in+' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe s3v9rd AP7Wnd").text
        print(calories)
        return calories

    except Exception as e:
        print(e)     



@app.route('/')
def index():
    return render_template('Home.html')   

@app.route('/getdata',methods = ['post'])
def data():
    return render_template('index.html')    

@app.route('/prediction', methods = ['GET','post'])   
def prediction():
    if request.method == 'POST':
        f = request.files['img']
        filename=f.filename
        target = os.path.join(APP_ROOT,'images/')
        print(target)
        des = "/".join([target,filename])
        f.save(des)
        
        test_image = load_img('images\\'+filename,target_size=(300,300))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)
        prediction = model.predict(test_image)
        print(prediction)

        prediction_class = class_name[np.argmax(prediction[0])]
        print(prediction_class)

        cal = fetch_calories(prediction_class)

        return render_template("index.html",prediction="prediction-> " +str(prediction_class),info=cal,Name = prediction_class)


    else:
        return render_template('index.html')


if __name__ ==  "__main__":
    app.run(debug=True)
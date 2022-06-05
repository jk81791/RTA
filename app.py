#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import percep
from flask import Flask, request
import numpy as np
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    sepal_length = float(request.args.get('s'))
    petal_length = float(request.args.get('p'))
    array = np.array([sepal_length, petal_length])
    with open('p.pkl', 'rb') as m:
        model = pickle.load(m)
    prediction = model.predict(array)
    prediction = prediction.tolist()
    if prediction==0:
        wynik="Iris_Setosa"
    elif prediction==1:
        wynik="Iris_Versicolor"
    else:
        wynik="Iris_Virginica"
    return wynik
    
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    
#127.0.0.1:5000/?s=4.5&p=3.2
#http://127.0.0.1:5000/?s=4.5&p=3.2

# In[2]:





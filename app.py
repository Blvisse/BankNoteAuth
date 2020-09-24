# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:04:17 2020

@author: blais
"""


import pandas as pd
import numpy as np
from flask import Flask, request
import pickle





app = Flask(__name__)
pickle_in=open('LGBMClasifier.pkl','rb')
LGBM=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return'welcome all'
    
@app.route('/predict')    
def predict_note_authentication():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=LGBM.predict([[variance,skewness,curtosis,entropy]])
    return 'The predicted value is: '+ str(prediction)

@app.route('/predict_file',methods=['POST'])    
def predict_note_file():
    testData=pd.read_csv(request.files.get('file'))
    predictions=LGBM.predict(testData)
    
    
    return 'The predicted values for the test data are as follows: '+ str(list(predictions))
    
    


if __name__ == '__main__':
    app.run()
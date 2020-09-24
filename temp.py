# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:04:17 2020

@author: blais
"""


import pandas as pd
import numpy as np
from flask import Flask, request
import pickle
import flasgger
from flasgger import Swagger 





app = Flask(__name__)
Swagger(app)


pickle_in=open('LGBMClasifier.pkl','rb')
LGBM=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return'welcome all'
    
@app.route('/predict')    
def predict_note_authentication():
    
    """
    Let's Authenticate Bank Notes
    Uisng DocString for Auhetication'
    
    ---
    parameters:
        - name:variance
          in:query
          type:number
          required:true
        - name:skewness
          in:query
          type:number
          required:true
        - name:curtosis
          in:query
          type:number
          required:true
        - name:entropy
          in:query
          type:number
          required:true
    responses:
        200:
            description: The output values
    
    
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=LGBM.predict([[variance,skewness,curtosis,entropy]])
    return 'The predicted value is: '+ str(prediction)

@app.route('/predict_file',methods=['POST'])    
def predict_note_file():
    """
    Lets's authenticate the Banks Note '
    Using docstring for specification
    
    parameters:
        - name:file
          in:formData
          type:file
          required:true
          
    responses:
        200:
            description:The output values
    
    """
    
    
    
    
    testData=pd.read_csv(request.files.get('file'))
    predictions=LGBM.predict(testData)
    
    
    return 'The predicted values for the test data are as follows: '+ str(list(predictions))
    
    


if __name__ == '__main__':
    app.run()
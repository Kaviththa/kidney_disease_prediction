from flask import Flask,render_template, request
import requests
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model/kidney_svm_model-5.pkl', 'rb'))
scaler = pickle.load(open('model/minmax_scaler-5.pkl', 'rb'))


@app.route('/kidney',methods=['GET'])
def kidney():
    return  render_template('kidney.html')



@app.route('/kidney_predict',methods=['POST'])
def kidney_predict():
    if request.method == 'POST':
          age = float(request.form['age'])
          bp = float(request.form['bp'])
          al = float(request.form['al'])
          su= float(request.form['su'])
          pcc = int(request.form['pcc'])
          ba= int(request.form['ba'])
          bgr = float(request.form['bgr'])
          bu = float(request.form['bu'])
          sc = float(request.form['sc'])
          sod = float(request.form['sod'])
          pot = float(request.form['pot'])
          hemo= float(request.form['hemo'])
          pcv = float(request.form['pcv'])
          wc = float(request.form['wc'])
          htn= int(request.form['htn'])
          dm = int(request.form['dm'])
          cad = int(request.form['cad'])
          appet= int(request.form['appet'])
          pe = int(request.form['pe'])
          ane = int(request.form['ane'])
          
          
         
          
          bp =bp/100
          
          X =[age,bp,al,su,pcc,ba,bgr,bu,sc,sod, pot,hemo,pcv,wc,htn,dm,cad,appet,pe,ane]
          y = pd.DataFrame([X],columns=['age','bp','al','su','pcc','ba','bgr','bu','sc','sod', 'pot','hemo','pcv','wc','htn','dm','cad','appet','pe','ane'])
          #dataset_to_scaled = [age,bgr,bu,sc,sod,pot,hemo,pcv,wc]
          y[['age','bgr','bu','sc','sod','pot','hemo','pcv','wc']]= pd.DataFrame(scaler.transform(y[['age','bgr','bu','sc','sod','pot','hemo','pcv','wc']]))
          
          prediction = model.predict(y)[0]
          prediction = int(prediction)
          
          
          return  render_template('kidney.html', kidney_result=prediction)

if __name__=="__main__":
    app.run(debug=True)

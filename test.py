from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
application=Flask(__name__)
import sys
import os
import pickle

app=application
# model=pickle.load(spo2.pkl)
## Route for a home page
class CustomData:
    def __init__(self,
                 age: float,
                 BMI: float,
                 diabetes: int,
                 Systolic_blood_pressure: float,
                 Diastolic_blood_pressure: float,
                 Respiratory_rate: float,
                 heart_rate: float):
        self.age = age
        self.BMI = BMI
        self.diabetes = diabetes
        self.Systolic_blood_pressure = Systolic_blood_pressure
        self.Diastolic_blood_pressure = Diastolic_blood_pressure
        self.Respiratory_rate = Respiratory_rate
        self.heart_rate = heart_rate

    def get_data_as_data_frame(self):
        # try:
        custom_data_input_dict = {
                "age": [self.age],
                "BMI": [self.BMI],
                "diabetes": [self.diabetes],
                "Systolic_blood_pressure": [self.Systolic_blood_pressure],
                "Diastolic_blood_pressure": [self.Diastolic_blood_pressure],
                "Respiratory_rate": [self.Respiratory_rate],
                "heart_rate": [self.heart_rate],
            }
        return pd.DataFrame(custom_data_input_dict)

        # except Exception as e:
        #     raise CustomException(e, sys)
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            print("Before Loading")
            model=pickle.load(open('oxy.pkl',"rb"))
            print("After Loading")
            preds=model.predict(get_data_as_data_frame)#data_scaled
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
model=pickle.load(open('oxy.pkl',"rb"))
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('predictions.html')
    # else:
    #     data = CustomData(
    #             age=int(request.form.get("age")),
    #             BMI=float(request.form.get("BMI")),
    #             diabetes=int(request.form.get("diabetes")),
    #             Systolic_blood_pressure=float(request.form.get("systolic_blood_pressure")),
    #             Diastolic_blood_pressure=float(request.form.get("diastolic_blood_pressure")),
    #             Respiratory_rate=float(request.form.get("respiratory_rate")),
    #             heart_rate=float(request.form.get("heart_rate")),
    #         )
    else:           
        data1=[[float(x) for x in request.form.values()]]
        # data1=np.array(data1).reshape(-1,1)
        print(data1)

        out=model.predict(data1)[0]
        return render_template('predictions.html',results=out)
        # print(pred_df)
        # print("Before Prediction")
        # model=pickle.load(open('oxy.pkl',"rb"))
        # # predict_pipeline=PredictPipeline()
        # print("Mid Prediction")
        # results=model.predict(pred_df)
        # print("after Prediction")
        
    

if __name__=="__main__":
    app.run(debug=True)        


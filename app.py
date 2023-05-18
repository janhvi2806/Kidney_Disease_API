from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import pandas as pd
import numpy as np
import pickle

class DataType(BaseModel):
    age: float
    bp: float
    sg: float
    al: float
    su: float
    rbc: str
    pc: str
    pcc: str
    ba: str
    bgr: float
    bu: float
    sc: float
    sod: float
    pot: float
    hemo: float
    pcv: int
    wc: int
    rc: float
    htn: str
    dm: str
    cad: str
    appet: str
    pe: str
    ane: str

app = FastAPI()

"""
Sample JSON Input:- 
{
    "radius": 23,
    "texture": 12,
    "perimeter": 151,
    "area": 954,
    "smoothness": 0.143,
    "compactness": 0.278,
    "symmetry": 0.252,
    "fractal_dimension": 0.079
}


{
    "radius": 0.062500,
    "texture": 0.812500,
    "perimeter": 0.375000,
    "area": 0.264320,
    "smoothness": 0.479452,
    "compactness": 0.485342,
    "symmetry": 0.532544,
    "fractal_dimension": 0.363636
}

{
    "radius": 0.125000,
    "texture": 0.000000,
    "perimeter": 0.233333,
    "area": 0.157518,
    "smoothness": 0.246575,
    "compactness": 0.182410,
    "symmetry": 0.343195,
    "fractal_dimension": 0.250000
}

radius               0.125000
texture              0.000000
perimeter            0.233333
area                 0.157518
smoothness           0.246575
compactness          0.182410
symmetry             0.343195
fractal_dimension    0.250000


"""


def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df= pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns    
    

def Preprocessing(data):
    print(data)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    print(type(scaled_data))
    return scaled_data

with open("kidney-disease.pkl", "rb") as f:
    model = pickle.load(f)

def scale_data(df):
    df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
    df[['rbc']] = df[['rbc']].replace(to_replace={'abnormal':1,'normal':0})
    df[['pcc','ba','pc']] = df[['pcc','ba','pc']].replace(to_replace={'present':1,'notpresent':0})
    df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
    e = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    df[e] = df[e].astype("O")
    return df

@app.post("/predict")
async def predict(item: DataType):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    print(df)
    data = scale_data(df)
    data,temp = one_hot_encoder(df, nan_as_category=True)
    print(data)
    print(data.columns)
    data.to_csv('datafile.csv')
    # ans1 = model.predict(data)
    # ans1 = list(ans1)
    # if ans1[0] == 0:
    #     return "Benign Prostatic Hyperplasia (BPH)"
    # else:
    #     return "Malignant Prostate Cancer (MPC)"

@app.get("/")
async def root():
    return {"message": "This API Only Has Get Method as of now"}



"""
{
    "radius": 9,
    "texture": 13,
    "perimeter": 133,
    "area": 1326,
    "smoothness": 0.143,
    "compactness": 0.079,
    "symmetry": 0.181,
    "fractal_dimension": 0.057
}
"""
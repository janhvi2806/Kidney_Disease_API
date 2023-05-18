# from fastapi import FastAPI
# from pydantic import BaseModel
# from sklearn.preprocessing import LabelEncoder,MinMaxScaler
# import pandas as pd
# import pickle

# class DataType(BaseModel):
#     radius: float
#     texture: float
#     perimeter: float
#     area: float
#     smoothness: float
#     compactness: float
#     symmetry: float
#     fractal_dimension: float

# app = FastAPI()

# """
# Sample JSON Input:- 
# {
#     "radius": 23,
#     "texture": 12,
#     "perimeter": 151,
#     "area": 954,
#     "smoothness": 0.143,
#     "compactness": 0.278,
#     "symmetry": 0.252,
#     "fractal_dimension": 0.079
# }


# {
#     "radius": 0.062500,
#     "texture": 0.812500,
#     "perimeter": 0.375000,
#     "area": 0.264320,
#     "smoothness": 0.479452,
#     "compactness": 0.485342,
#     "symmetry": 0.532544,
#     "fractal_dimension": 0.363636
# }

# {
#     "radius": 0.125000,
#     "texture": 0.000000,
#     "perimeter": 0.233333,
#     "area": 0.157518,
#     "smoothness": 0.246575,
#     "compactness": 0.182410,
#     "symmetry": 0.343195,
#     "fractal_dimension": 0.250000
# }

# radius               0.125000
# texture              0.000000
# perimeter            0.233333
# area                 0.157518
# smoothness           0.246575
# compactness          0.182410
# symmetry             0.343195
# fractal_dimension    0.250000


# """
# def Preprocessing(data):
#     print(data)
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(data)
#     print(type(scaled_data))
#     return scaled_data

# with open("kidney-disease.pkl", "rb") as f:
#     model = pickle.load(f)

# def scale_data(data):
#     min_max_list=[[25, 9],
#     [27, 11],
#     [172, 52],
#     [1878, 202],
#     [0.143, 0.07],
#     [0.345, 0.038],
#     [0.304, 0.135],
#     [0.097, 0.053]]
#     for i,col in zip(range(len(min_max_list)),data.columns):
#         X=data[col]
#         X_scaled = (X - min_max_list[i][1]) / (min_max_list[i][0] - min_max_list[i][1])
#         data[col]=X_scaled
#     return data

# @app.post("/predict")
# async def predict(item: DataType):
#     df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
#     data = scale_data(df)
#     ans1 = model.predict(data)
#     ans1 = list(ans1)
#     if ans1[0] == 0:
#         return "Benign Prostatic Hyperplasia (BPH)"
#     else:
#         return "Malignant Prostate Cancer (MPC)"

# @app.get("/")
# async def root():
#     return {"message": "This API Only Has Get Method as of now"}



# """
# {
#     "radius": 9,
#     "texture": 13,
#     "perimeter": 133,
#     "area": 1326,
#     "smoothness": 0.143,
#     "compactness": 0.079,
#     "symmetry": 0.181,
#     "fractal_dimension": 0.057
# }
# """

# --------------------------------------------------------------------------------------------------------
# import pickle
# with open("kidney-disease.pkl", "rb") as f:
#     model = pickle.load(f)

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
  "age": 48,
  "bp": 80,
  "sg": 1.02,
  "al": 1,
  "su": 0,
  "rbc": "normal",
  "pc": "notpresent",
  "pcc": "notpresent",
  "ba": "notpresent",
  "bgr": 121,
  "bu": 36,
  "sc": 1.2,
  "sod": 111,
  "pot": 2.5,
  "hemo": 15.4,
  "pcv": 44,
  "wc": 7800,
  "rc": 5.2,
  "htn": "yes",
  "dm": "yes",
  "cad": "no",
  "appet": "good",
  "pe": "no",
  "ane": "no"
}


"""


with open("kidney-disease.pkl", "rb") as f:
    model = pickle.load(f)



def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
    print('Number of Categorical Variables : ', len(cat_cols))
    print(cat_cols)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df= pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df

# def add_columns(df):
#     if rbc == 1:
#         df['rbc_1.0'] = 1
#         df['rbc_0.0'] = df['rbc_nan'] = 0
#     elif rbc == 0 :
#         df['rbc_0.0'] = 1
#         df['rbc_1.0'] = df['rbc_nan'] = 0
#     else :
#         df['rbc_nan'] = 1
#         df['rbc_0.0'] = df['rbc_1.0'] = 0

#     if 
    
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
    print(data)
    df = one_hot_encoder(df)
    print(df)
    print(df.shape)
    temp = [col for col in df.columns]
    print(temp)
    # ans1 = model.predict(data)
    # ans1 = list(ans1)
    # if ans1[0] == 0:
    #     return "Benign Prostatic Hyperplasia (BPH)"
    # else:
    #     return "Malignant Prostate Cancer (MPC)"

@app.get("/")
async def root():
    return {"message": "This API Only Has Get Method as of now"}





# fastapi==0.92.0
# uvicorn==0.20.0 pip install "uvicorn[standard]"
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import pandas as pd





app = FastAPI()


class Cont(BaseModel):
    income:int
    age:str
    year:int
    Credit_Score:int

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/contador')
def teste(cont:Cont):
    # Carrega o kmeans,labelencoder,scaler de normalização
    kmeans, le, scaler = load('cluster.joblib'), load('labelencoder.joblib'),load('scaler.joblib')
    df = pd.DataFrame({
        "income": [cont.income],
        "age": le.transform([cont.age]),
        "year": [cont.year],
        "Credit_Score": [cont.Credit_Score]
    })
    result = kmeans.predict(scaler.transform(df))

    return str(result)
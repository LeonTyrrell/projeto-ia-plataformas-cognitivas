# fastapi==0.92.0
# uvicorn==0.20.0 pip install "uvicorn[standard]"
from sklearn.preprocessing import LabelEncoder

from fastapi import FastAPI
from fastapi.responses import JSONResponse

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
    # Resultado da clusterização
    result = kmeans.predict(scaler.transform(df))
    # Resultado da clusterização

    # Média por cluster 
    media = (0.238315,0.279129,0.238814,0.274846,0.231080,0.362069)
    # Média por cluster 
    
    response = {
        'propensao_media': media[result[0]],
        'grupo': int(result[0])
    }
    return JSONResponse(content=response)
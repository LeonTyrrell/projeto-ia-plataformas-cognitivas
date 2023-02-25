''' ?Exemplo de post
{
	"income": 4980,
	"age": "55-64",
	"year": 2019,
	"gender": "Female"
}
'''



# fastapi==0.92.0
# uvicorn==0.20.0 pip install "uvicorn[standard]"

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import pandas as pd


app = FastAPI()


class Cont(BaseModel):
    income:int
    age:str # ['25-34', '35-44', '45-54', '55-64', '65-74', '<25', '>74']
    year:int
    # Credit_Score:int
    gender:str # ['Female', 'Joint', 'Male', 'Sex Not Available']

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/')
def teste(cont:Cont):
    # Carrega o kmeans,labelencoder,scaler de normalização
    kmeans, le_age,le_gender, scaler = load('cluster.joblib'), load('labelencoder_age.joblib'),load('labelencoder_gender.joblib'),load('scaler.joblib')
    df = pd.DataFrame({
        "income": [cont.income],
        "age": le_age.transform([cont.age]),
        "year": [cont.year],
        # "Credit_Score": [cont.Credit_Score],
        "Gender": le_gender.transform([cont.gender])
    })
    # Resultado da clusterização
    result = kmeans.predict(scaler.transform(df))
    # Resultado da clusterização

    # Média por cluster 
    media = (0.264404,0.208608,0.246458,0.372881,0.304962,0.224107)
    # Média por cluster 
    classi = (
                "Masculino e LGBT com risco de não pagamento",     # 0
                "Feminino e LGBT de alta renda",                   # 1
                "Feminino e LGBT de media renda de 3° idade",      # 2
                "Homens 2° idade de alta renda",                   # 3
                "Sem definição de sexo oui homens de baixa renda", # 4
                "Risco de não pagamento"                           # 5
            )
    
    response = {
        'propensao_media': media[result[0]],
        'grupo': int(result[0]),
        "classificacao": classi[result[0]] 
    }
    return JSONResponse(content=response)
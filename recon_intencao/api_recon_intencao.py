
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import os

from transformers import pipeline
import numpy as np
from starlette.responses import JSONResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_html():
    html_path = os.path.join("static", "index.html")
    return FileResponse(html_path)

@app.get("/recon_intencao/model_qa")
def call_model_qa(text, question):
    """
    Essa função recebe uma frase (str) e uma pergunta (str) e retorna um dicionário:

    Em caso de SUCESSO:
    1- sucesso: True
    2- resposta: resposta da pergunta

    Em caso de FALHA:
    1- sucesso: False
    2- erro: erro
    """
    try:
        pipe_qa = pipeline("question-answering", model='deepset/roberta-base-squad2') #mover para outra parte para não carregar toda vez
        QA_input = {'question': question, 'context': text}
        res = pipe_qa(QA_input)
        print(res)
        return {"sucesso":True, "resposta": res['answer']}
    except Exception as e:
        return {"sucesso":False, "erro": str(e)}

@app.get("/recon_intencao/model_zsc")
def call_model_zsc(text, question):
    """
    Essa função recebe uma frase (str) e os labels candidatos (list), e retorna um label e uma confiança:

    Em caso de SUCESSO:
    1- sucesso: True
    2- label: label mais provável
    3- confidence: confiança do label mais provável

    Em caso de FALHA:
    1- sucesso: False
    2- erro: erro
    """
    try:
        pipe_zsc = pipeline('zero-shot-classification', model='facebook/bart-large-mnli') #mover para outra parte para não carregar toda vez
        ret = pipe_zsc(text, question)
        label = ret['labels'][np.argmax(ret['scores'])]
        confidence = ret['scores'][np.argmax(ret['scores'])]
        return {"sucesso": True, "label": label, "confidence": confidence}
    except Exception as e:
        return {"sucesso": False, "erro": str(e)}


import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from transformers import pipeline
import os
import numpy as np
from pydantic import BaseModel
import transformers
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
import requests

# Carregamento dos modelos de forma global
transformers.logging.set_verbosity_error()
pipe_qa = pipeline("question-answering", model='deepset/roberta-base-squad2', device=-1)
pipe_zsc = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=-1)

# Importações específicas de suas classes personalizadas
from transcription_models.jonatasgrosman_wav2vec2 import jonatasgrosman_wav2vec2
#from transcription_models.pierreguillou_whisper import pierreguillou_whisper
from sentence_classification.faiss import TextClassifier
from sentence_classification.recon_intencao import Recon_Itencao

# Instância do FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Modelos de Transcrição
wav2vec2 = jonatasgrosman_wav2vec2()
#whisper = pierreguillou_whisper()
faiss = TextClassifier()

# Classes Pydantic para validação de entrada
class QAInput(BaseModel):
    text: str
    question: str

class ZSCInput(BaseModel):
    text: str
    labels: list[str]

@app.get("/")
async def serve_html():
    return FileResponse(os.path.join("static", "index.html"))

@app.post("/transcription/{model}")
def run_transcription(model: str, audio: UploadFile = File(...)):
    if model not in ["whisper", "wav2vec2"]:
        raise HTTPException(status_code=400, detail="Modelo não suportado")
    
    if model == "whisper":
        output = whisper.transcript(audio.file)
    else:  # wav2vec2
        output = wav2vec2.transcript(audio.file)
    
    print(output)
    return output

@app.post("/rawtranscription/{model}")
async def run_rawtranscription(model: str, url: str = Form(...)):
    if model not in ["whisper", "wav2vec2"]:
        raise HTTPException(status_code=400, detail="Modelo não suportado")

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Erro ao baixar o arquivo de áudio: {e}")

    with NamedTemporaryFile(delete=False, suffix=".oga") as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name
    if model == "whisper":
        #output = whisper.raw_transcript(audio.file)
        output = wav2vec2.raw_transcript(audio.file)
    else:  # wav2vec2
        output = wav2vec2.raw_transcript(audio.file)
    
    print(output)
    return output

@app.post("/rawtranscription_file/{model}")
def run_rawtranscription(model: str, audio: UploadFile = File(...)):
    if model not in ["whisper", "wav2vec2"]:
        raise HTTPException(status_code=400, detail="Modelo não suportado")

    if model == "whisper":
        #output = whisper.raw_transcript(audio.file)
        output = wav2vec2.raw_transcript(audio.file)
    else:  # wav2vec2
        output = wav2vec2.raw_transcript(audio.file)
    
    print(output)
    return output

@app.post("/recon_intencao/model_qa")
def call_model_qa(input: QAInput):
    try:
        QA_input = {'question': input.question, 'context': input.text}
        res = pipe_qa(QA_input)
        return {"sucesso": True, "resposta": res['answer']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recon_intencao/model_zsc")
def call_model_zsc(input: ZSCInput):
    try:
        ret = pipe_zsc(input.text, input.labels)
        label = ret['labels'][np.argmax(ret['scores'])]
        confidence = ret['scores'][np.argmax(ret['scores'])]
        return {"sucesso": True, "label": label, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ReconItenInput(BaseModel):
    text: str

@app.post("/recon_intencao")
def run_recon_iten(input: ReconItenInput):
    retorno = faiss.classifier(input.text)
    return retorno


@app.get("/health_check")
async def run_health_check():
    return {'res': True}

methods_info = {
    "/transcription/{model}": {
        "description": "Realiza a transcrição de um arquivo de áudio.",
        "models": {
            "whisper": "Usa o modelo Whisper para transcrição de áudio.",
            "wav2vec2": "Usa o modelo Wav2Vec 2.0 para transcrição de áudio."
        },
        "function": "run_transcription",
        "method": "POST",
        "content_type": "multipart/form-data"
    },
    "/rawtranscription/{model}": {
        "description": "Realiza a transcrição de um arquivo de áudio e retorna apenas a transcrição e o modelo usado.",
        "models": {
            "whisper": "Usa o modelo Whisper para transcrição de áudio.",
            "wav2vec2": "Usa o modelo Wav2Vec 2.0 para transcrição de áudio."
        },
        "function": "run_rawtranscription",
        "method": "POST",
        "content_type": "multipart/form-data"
    },
    "/recon_intencao": {
        "description": "Retorna um TOKEN com base em um contexto fornecido.",
        "model": "FAISS",
        "function": "run_recon_iten",
        "method": "POST",
        "content_type": "application/json"
    },
    "/recon_intencao/model_qa": {
        "description": "Responde a uma pergunta com base em um contexto fornecido.",
        "model": "deepset/roberta-base-squad2",
        "function": "call_model_qa",
        "method": "POST",
        "content_type": "application/json"
    },
    "/recon_intencao/model_zsc": {
        "description": "Classifica um texto em uma das categorias fornecidas.",
        "model": "facebook/bart-large-mnli",
        "function": "call_model_zsc",
        "method": "POST",
        "content_type": "application/json"
    },
    # Inclua outras rotas conforme necessário
}

@app.get("/methods")
def list_methods():
    return JSONResponse(content=methods_info)
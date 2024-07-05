
from fastapi import FastAPI
from transformers import pipeline
import uvicorn

app = FastAPI()

@app.get("/") #Mensaje al iniciar el aplicativo.
def read_root():
    return {"message": "Welcome to the FastAPI and Hugging Face pipeline practice BD13- RA"}

@app.get("/sentiment") #Como el visto en clase.
def analyze_sentiment(text: str):
  sentiment_analysis_pipeline = pipeline('sentiment-analysis')
  result = sentiment_analysis_pipeline(text)
  return {"sentiment": result}

@app.get("/summarize")
def summarize_text(text: str):
  summarization_pipeline = pipeline("summarization")
  result = summarization_pipeline(text)
  return {"summary": result}

@app.get("/translate")
def translate_text(text: str):
  translation_pipeline = pipeline("translation_en_to_fr")
  result = translation_pipeline(text)
  return {"translation": result}

@app.get("/add")
def adding(n1: float, n2: float):
  return {"Result": n1+n2}

@app.get("/square")
def square(number: float):
  return {"Result": number**2}



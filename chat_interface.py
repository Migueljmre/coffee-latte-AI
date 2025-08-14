import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL")

# Función para enviar mensajes al modelo
def chat_with_lm(message, history):
    # Construir el historial de mensajes
    messages = [{"role": "system", "content": "Eres un asistente útil. Genera respuestas cortas y concretas al usuario. Resumir todas las peticiones"}]
    messages.extend(history) 
    messages.append({"role": "user", "content": message})


    # Solicitud POST a la API local
    response = requests.post(API_URL, json={
        "model": "local-model",  # nombre del modelo local
        "messages": messages,
        "temperature": 0.7
    })

    # Procesar respuesta
    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        return reply
    else:
        return f"Error: {response.status_code} - {response.text}"
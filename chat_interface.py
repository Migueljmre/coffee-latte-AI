import requests

# URL de la API local de LM Studio
API_URL = "http://127.0.0.1:1234/v1/chat/completions"

# Función para enviar mensajes al modelo
def chat_with_lm(message, history):
    # Construir el historial de mensajes
    messages = [{"role": "system", "content": "Eres un asistente útil. Genera respuestas cortas y concretas al usuario. Resumir todas las peticiones"}]
    messages.extend(history)  # formato [{"role": ..., "content": ...}]
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
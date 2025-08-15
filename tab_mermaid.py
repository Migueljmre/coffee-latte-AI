import gradio as gr
import requests
import base64
from PIL import Image
import io
import os
import shutil
from dotenv import load_dotenv
from mermaid import Mermaid
# Cargar variables de entorno
load_dotenv()
API_URL = os.getenv("API_URL")


# Ruta de imagen fija TEST.png
test_image_path = "temp.png"
test_image = Image.open(test_image_path) if os.path.exists(test_image_path) else None

# Estado para mantener contexto
chat_history = {"last_response": None, "first_message": True}

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def generar_mermaid(prompt_text, image):
    if chat_history["first_message"] and image is None:
        return "⚠️ El primer mensaje debe incluir una imagen."

    messages = []
    if chat_history["first_message"]:
        messages.append({"role": "system", "content": "Eres un asistente que convierte diagramas en código Mermaid. no uses el formato de markdow en tu respuesra evita ```"})
        user_content = [{"type": "text", "text": prompt_text}]
    else:
        messages.append({"role": "system", "content": "Eres un asistente que convierte diagramas en código Mermaid. Usa el contexto anterior para continuar."})
        user_content = [{"type": "text", "text": chat_history["last_response"] +"de este diagrama cambia: "+ prompt_text+"no me expliques nada solo dame el codigo"}]
    
    if image is not None:
        imagen_b64 = image_to_base64(image)
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{imagen_b64}"}})

    messages.append({"role": "user", "content": user_content})

    response = requests.post(API_URL, json={
        "model": "local-model",
        "messages": messages,
        "temperature": 0.3
    })

    
    respuesta = response.json()["choices"][0]["message"]["content"]
    respuestalimpia = respuesta.replace("```mermaid","").replace("```","").strip()
    print(respuestalimpia)
    chat_history["last_response"] = respuesta
    chat_history["first_message"] = False
    diagram = Mermaid(respuestalimpia)
    diagram.to_png("temp.png")
    return respuesta

def generar_explicacion_detallada(directorio_destino):
    if chat_history["last_response"] is None:
        return "⚠️ No hay respuesta previa para generar explicación."

    prompt = chat_history["last_response"] + "\nGenerame una explicación general de este diagrama."
    messages = [
        {"role": "system", "content": "Eres un asistente que explica diagramas en detalle."},
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]

    response = requests.post(API_URL, json={
        "model": "local-model",
        "messages": messages,
        "temperature": 0.3
    })

    explicacion = response.json()["choices"][0]["message"]["content"]

    # Copiar imagen TEST.png al directorio destino
    if test_image and os.path.isdir(directorio_destino):
        destino = os.path.join(directorio_destino, "TEST.png")
        shutil.copy(test_image_path, destino)

    return explicacion

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt del usuario", lines=2)
        with gr.Column():
            respuesta_output = gr.Textbox(label="Respuesta del modelo", lines=10)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Subir imagen", type="pil")
        with gr.Column():
            gr.Image(value=test_image, label="Imagen fija TEST.png")

    submit_btn = gr.Button("Enviar")
    submit_btn.click(fn=generar_mermaid, inputs=[prompt_input, image_input], outputs=respuesta_output)

    with gr.Row():
        with gr.Column():
            directorio_input = gr.Textbox(label="Directorio para guardar copia de TEST.png")
        with gr.Column():
            explicacion_output = gr.Textbox(label="Explicación detallada", lines=10)

    explicacion_btn = gr.Button("Generar explicación detallada")
    explicacion_btn.click(fn=generar_explicacion_detallada, inputs=directorio_input, outputs=explicacion_output)

demo.launch()

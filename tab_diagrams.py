import gradio as gr
import requests
import base64
import io
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL")

# Prompt para generar código Mermaid desde imagen UML
PROMPT = (
    "Genera un diagrama en codigo Mermaid y presentalo visualmente con una pequeña explicación de lo que representa el diagrama."
)

# Convertir imagen a base64
def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Función principal para enviar imagen a LM Studio y obtener código Mermaid
def generar_mermaid(imagen):
    if imagen is None:
        return "No se ha proporcionado una imagen."

    imagen_b64 = image_to_base64(imagen)

    messages = [
        {"role": "system", "content": "Eres un asistente que convierte diagramas en código Mermaid."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{imagen_b64}"}}
            ]
        }
    ]

    response = requests.post(API_URL, json={
        "model": "gemma-3-4b-it",
        "messages": messages,
        "temperature": 0.4
    })

    if response.status_code != 200:
        return f"Error al contactar LLM: {response.status_code}"

    content = response.json()["choices"][0]["message"]["content"]
    return content

def tab_diagrams():
    # Interfaz Gradio
    with gr.TabItem("🗂️ Generador de diagramas"):
        gr.Markdown("### 🚀 Dos pasos para tu diagrama")
        gr.Markdown(
            "**Step 1:** Sube boceto o fotografia en formado, jpeg, jpg o png, → **Step 2:** Generará tu diagrama editable"
        )
        imagen_input = gr.Image(type="pil", label="Sube tu imagen boceto o fotografía")
        mermaid_output = gr.Code(label="Código Mermaid generado", language="markdown")

        boton = gr.Button("Generar Diagrama")

        boton.click(fn=generar_mermaid, inputs=imagen_input, outputs=mermaid_output)
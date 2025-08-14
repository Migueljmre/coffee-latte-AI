import gradio as gr
import requests
import base64
import io
from PIL import Image

# URL de la API local de LM Studio
API_URL = "http://127.0.0.1:1234/v1/chat/completions"

# Prompt para LM Studio
PROMPT = (
    "Analiza esta imagen que contiene un boceto o fotografía de un diagrama de clases UML. "
    "Extrae las clases, atributos, métodos y relaciones (herencia, asociación, composición, etc.). "
    "Genera un archivo XML compatible con draw.io que represente este diagrama de forma editable. "
    "El resultado debe ser un archivo .drawio con sintaxis mxfile que pueda abrirse directamente en draw.io."
)

# Función para convertir imagen a base64
def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Función principal
def generar_xml(imagen):
    if imagen is None:
        return None, "No se ha proporcionado una imagen."

    imagen_b64 = image_to_base64(imagen)

    messages = [
        {"role": "system", "content": "Eres un asistente que convierte diagramas UML en XML para draw.io."},
        {"role": "user", "content": PROMPT},
        {"role": "user", "content": f"Imagen codificada en base64:\n{imagen_b64}"}
    ]

    response = requests.post(API_URL, json={
        "model": "local-model",
        "messages": messages,
        "temperature": 0.7
    })

    if response.status_code != 200:
        return None, f"Error al contactar LM Studio: {response.status_code}"

    content = response.json()["choices"][0]["message"]["content"]
    print(content)
    if "<mxfile" not in content:
        return None, "La respuesta no contiene XML válido para draw.io."

    start = content.find("<mxfile")
    end = content.rfind("</mxfile>") + len("</mxfile>")
    xml_content = content[start:end]

    xml_path = "uml_diagrama.drawio"
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    return xml_path, "Archivo XML generado correctamente."
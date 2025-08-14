import gradio as gr
import os
import docx
import requests
import fitz  # PyMuPDF

# Herramienta: Leer archivo PDF
def leer_pdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        texto = "\n".join([page.get_text() for page in doc])
        resumen = resumir_texto(texto)
        return resumen
    except Exception as e:
        return f"Error al leer PDF: {e}"

# Herramienta: Leer archivo Word
def leer_word(path: str) -> str:
    try:
        doc = docx.Document(path)
        texto = "\n".join([p.text for p in doc.paragraphs])
        resumen = resumir_texto(texto)
        return resumen
    except Exception as e:
        return f"Error al leer Word: {e}"

# Herramienta: Leer archivo Excel
def leer_excel(path: str) -> str:
    try:
        df = pd.read_excel(path)
        texto = df.to_string()
        resumen = resumir_texto(texto)
        return resumen
    except Exception as e:
        return f"Error al leer Excel: {e}"

# Herramienta: Leer archivo de texto
def leer_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            texto = f.read()
        resumen = resumir_texto(texto)
        return resumen
    except Exception as e:
        return f"Error al leer TXT: {e}"

# Herramienta: Escribir archivo
def escribir_archivo(path: str, contenido: str) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(contenido)
        return f"Archivo escrito correctamente en {path}"
    except Exception as e:
        return f"Error al escribir archivo: {e}"

# Función auxiliar para resumir texto usando LM Studio
def resumir_texto(texto: str) -> str:
     # Solicitud POST a la API local
    response = requests.post(API_URL, json={
        "model": "local-model",  # nombre del modelo local
        "messages": [
            {"role": "system", "content": "Resumir el contenido del siguiente texto:"},
            {"role": "user", "content": texto}
        ],
        "temperature": 0.7
    })

    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        return reply
    else:
        return f"Error: {response.status_code} - {response.text}"

def procesar_archivo(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".pdf":
        return leer_pdf(file.name)
    elif ext in [".docx", ".doc"]:
        return leer_word(file.name)
    elif ext in [".xls", ".xlsx"]:
        return leer_excel(file.name)
    elif ext in [".txt"]:
        return leer_txt(file.name)
    else:
        return "Formato no soportado."

def tab_agent():
# Tab para resumen de archivos
    with gr.TabItem("📥 Agentes"):
        gr.Markdown("### 🚀 Dos pasos para obtener un resumen")
        gr.Markdown(
            "**Step 1:** Sube el archivo en formado .docx, pdf, xlsx, txt → **Step 2:** Genera un URLresumen del contenido del archivo"
        )

        archivo = gr.File(label="Sube un archivo")
        salida = gr.Textbox(label="Resumen generado", lines=10)
        boton = gr.Button("Procesar archivo")
        boton.click(fn=procesar_archivo, inputs=archivo, outputs=salida)
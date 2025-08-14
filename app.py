import gradio as gr
import os
import pandas as pd
from dotenv import load_dotenv
from assistant import assistant
from tab_diagrams import tab_diagrams
from tab_agent import tab_agent
from tab_config import tab_config

load_dotenv()
API_URL = os.getenv("API_URL")

def mostrar_imagen(imagen):
    return imagen

# Interfaz Gradio
with gr.Blocks(title="Hackaton 2025") as demo:
    gr.Markdown("# 📚 Café con Leche: AI")
    gr.Markdown(
        "AI Agents"
    )
    
    with gr.Tabs():
        assistant()
        tab_diagrams()
        tab_agent()
        tab_config()
if __name__ == "__main__":
    demo.launch(share=False)

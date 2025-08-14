import gradio as gr
from chat_interface import chat_with_lm

# Inicializar
chat = chat_with_lm

def assistant(): 
    # Tab para chat privado 
    with gr.TabItem("🤖 AI Assistant"):
        # Interfaz Gradio
        gr.ChatInterface(
            chat,
            type="messages",
            flagging_mode="manual",
            # title="# Chat privado",
            # stream=True,
            save_history = True,
            )
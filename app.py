import gradio as gr
import requests
import os
#import fitz  # PyMuPDF
import docx
import pandas as pd
from chat_interface import chat_with_lm

# Configuración de API local (LM Studio u Ollama)
API_URL = "http://127.0.0.1:1234/v1/chat/completions"  # Cambia el puerto si usas Ollama

# Inicializar
chat = chat_with_lm

# Herramienta: Leer archivo PDF
def leer_pdf(path: str) -> str:
    try:
        # doc = fitz.open(path)
        # texto = "\n".join([page.get_text() for page in doc])
        resumen = resumir_texto("")
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
# Interfaz Gradio
with gr.Blocks(title="Hackaton 2025") as demo:
    gr.Markdown("# 📚 Café con Leche: AI")
    gr.Markdown(
        "AI Agents"
    )
    
    with gr.Tabs():
        # Tab para resumen de archivos
        with gr.TabItem("📥 Resumen de archivos"):
            gr.Markdown("### 🚀 Dos pasos para obtener un resumen")
            gr.Markdown(
                "**Step 1:** Sube el archivo en formado .docx, pdf, xlsx, txt → **Step 2:** Genera un resumen del contenido del archivo"
            )

            archivo = gr.File(label="Sube un archivo")
            salida = gr.Textbox(label="Resumen generado", lines=10)
            boton = gr.Button("Procesar archivo")
            boton.click(fn=procesar_archivo, inputs=archivo, outputs=salida)
       
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
        # ================================
        # Tab 5: About & MCP Configuration
        # ================================
        with gr.TabItem("ℹ️ About & MCP Setup"):
            gr.Markdown("# 📚 Doc-MCP: Documentation RAG System")
            gr.Markdown(
                "**Transform GitHub documentation repositories into accessible MCP servers for AI agents.**"
            )

            with gr.Row():
                with gr.Column(scale=2):
                    # Project Overview
                    with gr.Accordion("🎯 What is Doc-MCP?", open=True):
                        gr.Markdown("""
                        **Doc-MCP** converts GitHub documentation into AI-queryable knowledge bases via the Model Context Protocol.
                        
                        **🔑 Key Features:**
                        - 📥 **GitHub Integration** - Automatic markdown file extraction
                        - 🧠 **AI Embeddings** - Nebius AI-powered vector search  
                        - 🔍 **Smart Search** - Semantic, keyword & hybrid modes
                        - 🤖 **MCP Server** - Direct AI agent integration
                        - ⚡ **Real-time** - Live processing progress
                        """)

                    # Quick Start Guide
                    with gr.Accordion("🚀 Quick Start", open=False):
                        gr.Markdown("""
                        **1. Ingest Documentation** → Enter GitHub repo URL → Select files → Run 2-step pipeline
                        
                        **2. Query with AI** → Select repository → Ask questions → Get answers with sources
                        
                        **3. Manage Repos** → View stats → Delete old repositories
                        
                        **4. Use MCP Tools** → Configure your AI agent → Query docs directly from IDE
                        """)

                with gr.Column(scale=2):
                    # MCP Server Configuration
                    with gr.Accordion("🔧 MCP Server Setup", open=True):
                        gr.Markdown("### 🌐 Server URL")

                        # Server URL
                        gr.Textbox(
                            value="https://agents-mcp-hackathon-doc-mcp.hf.space/gradio_api/mcp/sse",
                            label="MCP Endpoint",
                            interactive=False,
                            info="Copy this URL for your MCP client configuration",
                        )

                        gr.Markdown("### ⚙️ Configuration")

                        # SSE Configuration
                        with gr.Accordion("For Cursor, Windsurf, Cline", open=False):
                            sse_config = """{
  "mcpServers": {
    "doc-mcp": {
      "url": "https://agents-mcp-hackathon-doc-mcp.hf.space/gradio_api/mcp/sse"
    }
  }
}"""
                            gr.Code(
                                value=sse_config,
                                label="SSE Configuration",
                                language="json",
                                interactive=False,
                            )

                        # STDIO Configuration
                        with gr.Accordion(
                            "For STDIO Clients (Experimental)", open=False
                        ):
                            stdio_config = """{
  "mcpServers": {
    "doc-mcp": {
      "command": "npx",
      "args": ["mcp-remote", "https://agents-mcp-hackathon-doc-mcp.hf.space/gradio_api/mcp/sse", "--transport", "sse-only"]
    }
  }
}"""
                            gr.Code(
                                value=stdio_config,
                                label="STDIO Configuration",
                                language="json",
                                interactive=False,
                            )

            # MCP Tools Overview
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🛠️ Available MCP Tools")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**🔍 Documentation Query Tools**")
                            gr.Markdown(
                                "• `get_available_docs_repo` - List repositories"
                            )
                            gr.Markdown("• `make_query` - Search documentation with AI")

                        with gr.Column():
                            gr.Markdown("**📁 GitHub File Tools**")
                            gr.Markdown("• `list_repository_files` - Scan repo files")
                            gr.Markdown("• `get_single_file` - Fetch one file")
                            gr.Markdown("• `get_multiple_files` - Fetch multiple files")

            # Technology Stack & Project Info
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("⚙️ Technology Stack", open=False):
                        gr.Markdown("**🖥️ Frontend & API**")
                        gr.Markdown("• **Gradio** - Web interface & API framework")
                        gr.Markdown("• **Hugging Face Spaces** - Cloud hosting")

                        gr.Markdown("**🤖 AI & ML**")
                        gr.Markdown("• **Nebius AI** - LLM & embedding models")
                        gr.Markdown("• **LlamaIndex** - RAG framework")

                        gr.Markdown("**💾 Database & Storage**")
                        gr.Markdown("• **MongoDB Atlas** - Vector database")
                        gr.Markdown("• **GitHub API** - Source file access")

                        gr.Markdown("**🔌 Integration**")
                        gr.Markdown("• **Model Context Protocol** - AI agent standard")
                        gr.Markdown(
                            "• **Server-Sent Events** - Real-time communication"
                        )

                with gr.Column():
                    with gr.Accordion("👥 Project Information", open=False):
                        gr.Markdown("**🏆 MCP Hackathon Project**")
                        gr.Markdown(
                            "Created to showcase AI agent integration with documentation systems."
                        )

                        gr.Markdown("**💡 Inspiration**")
                        gr.Markdown("• Making Gradio docs easily searchable")
                        gr.Markdown("• Leveraging Hugging Face AI ecosystem")
                        gr.Markdown(
                            "• Improving developer experience with AI assistants"
                        )

                        gr.Markdown("**🔮 Future Plans**")
                        gr.Markdown("• Support for PDF, HTML files")
                        gr.Markdown("• Multi-language documentation")
                        gr.Markdown("• Custom embedding fine-tuning")

                        gr.Markdown("**📄 License:** MIT - Free to use and modify")

            # Usage Examples
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("💡 Usage Examples", open=False):
                        gr.Markdown("### Example Workflow")

                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**📥 Step 1: Ingest Docs**")
                                gr.Code(
                                    value="1. Enter: gradio-app/gradio\n2. Select markdown files\n3. Run ingestion pipeline",
                                    label="Ingestion Process",
                                    interactive=False,
                                )

                            with gr.Column():
                                gr.Markdown("**🤖 Step 2: Query with AI**")
                                gr.Code(
                                    value='Query: "How to create custom components?"\nResponse: Detailed answer with source links',
                                    label="AI Query Example",
                                    interactive=False,
                                )

                        gr.Markdown("### MCP Tool Usage")
                        gr.Code(
                            value="""# In your AI agent:
1. Call: get_available_docs_repo() -> ["gradio-app/gradio", ...]  
2. Call: make_query("gradio-app/gradio", "default", "custom components")
3. Get: AI response + source citations""",
                            label="MCP Integration Example",
                            language="python",
                            interactive=False,
                        )
if __name__ == "__main__":
    demo.launch(share=False)

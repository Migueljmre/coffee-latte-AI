import gradio as gr 
from upload_img import generar_xml

def tab_diagrams():
# Tab para resumen de archivos
        with gr.TabItem("🗂️ Generador de diagramas"):
            gr.Markdown("### 🚀 Dos pasos para tu diagrama uml")
            gr.Markdown(
                "**Step 1:** Sube boceto o fotografia en formado, jpeg, jpg o png, → **Step 2:** Generará tu diagrama editable"
            )
            
            imagen_input = gr.Image(type="pil", label="Sube tu imagen UML (boceto o fotografía)")
            archivo_output = gr.File(label="Descargar archivo .drawio")

            mensaje_estado = gr.Textbox(label="Estado", interactive=False)
            boton = gr.Button("Generar XML para draw.io")
            boton.click(fn=generar_xml, inputs=imagen_input, outputs=[archivo_output, mensaje_estado])
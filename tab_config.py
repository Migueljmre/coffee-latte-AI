import gradio as gr
from dotenv import load_dotenv
import json
import os

load_dotenv()

def guardar_config(local, local_on, azure, azure_on, aws, aws_on, gcp, gcp_on, sharepoint, sharepoint_on):
    config = {
        "rutas": {
            "local": local,
            "azure": azure,
            "aws": aws,
            "gcp": gcp,
            "sharepoint": sharepoint
        },
        "activos": {
            "local": local_on,
            "azure": azure_on,
            "aws": aws_on,
            "gcp": gcp_on,
            "sharepoint": sharepoint_on
        }
    }
    with open(os.getenv("json"), "w") as f:
        json.dump(config, f, indent=4)
    return "✅ Configuración guardada exitosamente"

def cargar_config():
    if os.path.exists(os.getenv("json")):
        with open(os.getenv("json"), "r") as f:
            config = json.load(f)
            rutas = config.get("rutas", {})
            activos = config.get("activos", {})
            return (
                rutas.get("local", ""), activos.get("local", False),
                rutas.get("azure", ""), activos.get("azure", False),
                rutas.get("aws", ""), activos.get("aws", False),
                rutas.get("gcp", ""), activos.get("gcp", False),
                rutas.get("sharepoint", ""), activos.get("sharepoint", False)
            )
    else:
        return "", False, "", False, "", False, "", False, "", False

def fila_ruta(label, valor_ruta, valor_switch):
    with gr.Row():
        ruta = gr.Textbox(label=label, value=valor_ruta, scale=5)
        switch = gr.Checkbox(label="Activo", value=valor_switch, scale=1)
    return ruta, switch

def tab_config():
    with gr.Tab("⚙️ settings"):
        gr.Markdown("# ☕ Configuración de rutas de guardado:")

        ruta_local, local_on, ruta_azure, azure_on, ruta_aws, aws_on, ruta_gcp, gcp_on, ruta_sharepoint, sharepoint_on = cargar_config()

        local_input, local_switch = fila_ruta("Ruta Local", ruta_local, local_on)
        azure_input, azure_switch = fila_ruta("Ruta Azure", ruta_azure, azure_on)
        aws_input, aws_switch = fila_ruta("Ruta AWS", ruta_aws, aws_on)
        gcp_input, gcp_switch = fila_ruta("Ruta GCP", ruta_gcp, gcp_on)
        sharepoint_input, sharepoint_switch = fila_ruta("Ruta SharePoint", ruta_sharepoint, sharepoint_on)

        boton_guardar = gr.Button("Guardar configuración")
        mensaje_salida = gr.Textbox(label="Estado")

        boton_guardar.click(
            fn=guardar_config,
            inputs=[
                local_input, local_switch,
                azure_input, azure_switch,
                aws_input, aws_switch,
                gcp_input, gcp_switch,
                sharepoint_input, sharepoint_switch
            ],
            outputs=mensaje_salida
        )

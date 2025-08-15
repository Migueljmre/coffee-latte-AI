import tkinter as tk
from tkinter import messagebox, simpledialog
import requests
from bs4 import BeautifulSoup
import urllib.parse

# Función que se ejecuta al hacer clic en el botón
def realizar_scrapping():
    # Obtener URL desde cuadro de diálogo
    url = simpledialog.askstring("URL", "Ingresa la URL (por ejemplo: https://example.com):")
    
    if not url:
        messagebox.showwarning("Error", "No ingresaste una URL.")
        return
    
    # Obtener texto para buscar (opcional)
    texto_buscar = simpledialog.askstring("Texto de búsqueda", "Escribe el texto a buscar (ej. python, AI, etc.):")
    
    if not texto_buscar:
        messagebox.showwarning("Error", "No ingresaste un término de búsqueda.")
        return
    
    # Construir la URL con parámetro de búsqueda
    url_completa = f"{url}?q={urllib.parse.quote(texto_buscar)}"
    
    try:
        # Hacer petición HTTP (GET)
        response = requests.get(url_completa, timeout=10)
        response.raise_for_status()  # Lanza error si hay problema
        
        # Parsear el contenido HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        print(soup)
        # Extraer contenido (por ejemplo, títulos o párrafos)
        titulos = []
        for titulo in soup.find_all('a')[:50]:  # Tomamos los primeros 50 <a>
            titulos.append(titulo.get_text(strip=True))

        if not titulos:
            resultado = "No se encontraron títulos en la página."
        else:
            resultado = "\n".join(f"• {t}" for t in titulos)
        
        # Mostrar resultado en ventana de mensaje
        messagebox.showinfo("Resultados del scraping", f"URL: {url_completa}")
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Error", f"No se pudo acceder a la URL. Error: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error inesperado: {e}")

# Crear ventana principal
root = tk.Tk()
root.title("Herramienta de Scraping")
root.geometry("500x300")

# Etiqueta y botón
etiqueta = tk.Label(root, text="Haz clic en 'Buscar' para hacer scraping", font=("Arial", 12))
etiqueta.pack(pady=20)

boton = tk.Button(root, text="🔍 Hacer Scraping", command=realizar_scrapping, bg="#4CAF50", fg="white", font=("Arial", 12), height=2, width=30)
boton.pack(pady=20)

# Iniciar la ventana
root.mainloop()

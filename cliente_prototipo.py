import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import webbrowser
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import time

# --- CONFIGURACIÓN Y RECURSOS ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")
WINDOW_SIZE = 256 # El tamaño de la ventana (ticks) que el modelo espera

class NeuralAntiCheatApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CS2 Neural Anti-Cheat - Forensic Engine")
        self.geometry("1200x800")
        
        # Cargar recursos de IA al iniciar
        self.load_ia_resources()

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- PANEL LATERAL ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo = ctk.CTkLabel(self.sidebar, text="🛡️ NEURAL\nANTI-CHEAT", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo.grid(row=0, column=0, padx=20, pady=(40, 20))
        
        self.btn_load = ctk.CTkButton(self.sidebar, text="ANALIZAR DEMO (.dem)", height=50,
                                      fg_color="#c0392b", hover_color="#e74c3c", font=ctk.CTkFont(weight="bold"),
                                      command=self.iniciar_analisis)
        self.btn_load.grid(row=1, column=0, padx=20, pady=20)

        self.lbl_status = ctk.CTkLabel(self.sidebar, text="SISTEMA LISTO", text_color="#2ecc71")
        self.lbl_status.grid(row=2, column=0, padx=20, pady=10)

        # --- PANEL PRINCIPAL ---
        self.main_view = ctk.CTkScrollableFrame(self, label_text="REPORTE DE DETECCIÓN")
        self.main_view.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

    def load_ia_resources(self):
        """Carga el modelo, escalador y lista de columnas."""
        try:
            # Archivos clave
            model_path = 'anticheat_model_kill_focused.keras'
            scaler_path = 'scaler_entrenado.pkl'
            cols_path = 'columnas_numericas.pkl'
            
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.numeric_cols = joblib.load(cols_path)
            
            self.lbl_status.configure(text="✅ MODELO CARGADO", text_color="#2ecc71")
            print("Recursos cargados correctamente.")
        except Exception as e:
            messagebox.showerror("Error de Carga", f"Faltan archivos clave: {e}. Asegúrate de tener {model_path}, {scaler_path} y {cols_path} en esta carpeta.")
            self.model = None

    def iniciar_analisis(self):
        if not self.model:
            messagebox.showwarning("Advertencia", "Modelo no cargado. No se puede analizar.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("CS2 Demos", "*.dem")])
        if file_path:
            self.lbl_status.configure(text="🟡 PROCESANDO DEMO...", text_color="#f39c12")
            self.btn_load.configure(state="disabled")
            threading.Thread(target=self.pipeline_analisis, args=(file_path,)).start()

    def procesar_demo_real(self, file_path):
        """
        [ATENCIÓN: FUNCIÓN PUENTE]
        Esta función simula la salida de un parser real.
        El usuario debe integrar aquí el código de awpy/demoparser para devolver
        un DataFrame con la telemetría de cada tick.
        """
        print(f"Iniciando parser para: {file_path}")
        time.sleep(1.0) 
        
        # --- ESTRUCTURA DE DATOS REQUERIDA ---
        # Este DataFrame debe ser la salida del parser, donde cada fila es un kill.
        # En una demo real, el parser nos daría:
        
        # Simulación de la Salida de un Parser (para fines de ejecución de la App)
        datos_simulados = []
        jugadores = [
            {"steamid": "76561198000000002", "nombre": "SpinBot_User_1337", "kills": 42},
            {"steamid": "76561198000000004", "nombre": "Closet_Cheater_X", "kills": 28},
            {"steamid": "76561198000000001", "nombre": "S1mple_Wannabe", "kills": 25},
        ]
        
        for jugador in jugadores:
            # Simular 3 ventanas de contexto por jugador (3 kills sospechosas)
            for i in range(3):
                 datos_simulados.append({
                    'steamid': jugador['steamid'],
                    'nombre': jugador['nombre'],
                    'kills_totales_partida': jugador['kills'],
                    # Simular tensor de 256 ticks, con valores aleatorios para que la IA prediga.
                    # Estos valores DEBEN ser reales y sin escalar en producción.
                    'window_data': np.random.rand(WINDOW_SIZE, len(self.numeric_cols)) 
                 })

        return pd.DataFrame(datos_simulados)


    def pipeline_analisis(self, file_path):
        """Ejecuta el flujo completo de Inferencia."""
        try:
            self.lbl_status.configure(text="🟡 PARSING y EXTRACCIÓN...", text_color="#f39c12")
            
            # 1. PARSING REAL (Función a reemplazar por el usuario)
            df_raw_windows = self.procesar_demo_real(file_path) 
            
            # 2. PREPROCESAMIENTO -> Obtener tensores listos para la IA
            self.lbl_status.configure(text="🟠 ESCALANDO Y FORMATEANDO...", text_color="#e67e22")
            X_windows_scaled, meta_info = self.preparar_datos_para_modelo(df_raw_windows)

            # 3. INFERENCIA
            self.lbl_status.configure(text="🔴 EJECUTANDO NEURAL ENGINE...", text_color="#c0392b")
            predicciones = self.model.predict(X_windows_scaled)
            
            # 4. AGRUPAR RESULTADOS Y VEREDICTO
            self.lbl_status.configure(text="🟣 AGREGANDO RESULTADOS...", text_color="#8e44ad")
            resultados_finales = self.agrupar_resultados(predicciones, meta_info)
            
            # 5. ACTUALIZAR UI
            self.after(0, lambda: self.mostrar_resultados(resultados_finales))
            self.after(0, lambda: self.reset_ui())

        except Exception as e:
            print(f"Error en pipeline: {e}")
            self.after(0, lambda: self.lbl_status.configure(text=f"❌ ERROR: {e}", text_color="red"))
            self.after(0, lambda: self.btn_load.configure(state="normal"))

    def preparar_datos_para_modelo(self, df_raw_windows):
        """Escala y prepara los tensores para Keras."""
        X_list = []
        meta_info = []

        for index, row in df_raw_windows.iterrows():
            # Simulación: En producción, aquí aplicaríamos One-Hot Encoding si fuera necesario
            
            # 1. Escalar (CRÍTICO: Usar el scaler entrenado)
            # Reestructuramos la ventana para que sea 2D para el scaler, y luego 3D para el modelo
            window_2d = row['window_data'].reshape(-1, len(self.numeric_cols))
            window_scaled_2d = self.scaler.transform(window_2d)
            window_scaled_3d = window_scaled_2d.reshape(1, WINDOW_SIZE, len(self.numeric_cols))
            
            X_list.append(window_scaled_3d[0])
            meta_info.append({"steamid": row['steamid'], "nombre": row['nombre'], "kills_totales_partida": row['kills_totales_partida']})

        return np.array(X_list), meta_info

    def agrupar_resultados(self, predicciones, meta_info):
        """Agrega las predicciones por jugador y calcula el 'CSWatch' (Score Estadístico)."""
        jugadores = {}
        UMBRAL_STRICTO = 0.80 # Umbral para el veredicto final

        for i, pred in enumerate(predicciones):
            info = meta_info[i]
            sid = info['steamid']
            prob = pred[0]
            
            if sid not in jugadores:
                jugadores[sid] = {"nombre": info['nombre'], "probs": [], "kills_totales_partida": info['kills_totales_partida']}
            jugadores[sid]["probs"].append(prob)

        reporte = []
        for sid, data in jugadores.items():
            max_prob = np.max(data["probs"])
            avg_prob = np.mean(data["probs"])
            
            # Calculo del CSWatch Score (Ejemplo: Promedio ponderado de probabilidad y kills)
            # Un score simple que el usuario puede refinar
            score_ponderado_cswatch = (max_prob * 0.7) + (avg_prob * 0.3) 
            
            reporte.append({
                "steamid": sid,
                "nombre": data["nombre"],
                "probabilidad_max": max_prob,
                "veredicto": max_prob >= UMBRAL_STRICTO, # Veredicto con umbral estricto
                "cswatch_score": score_ponderado_cswatch,
                "kills_analizadas": len(data["probs"]),
                "kills_totales_partida": data["kills_totales_partida"]
            })
            
        return reporte

    def reset_ui(self):
        self.lbl_status.configure(text="✅ ANÁLISIS COMPLETADO", text_color="#2ecc71")
        self.btn_load.configure(state="normal")

    def mostrar_resultados(self, datos):
        for widget in self.main_view.winfo_children():
            widget.destroy()

        datos_ordenados = sorted(datos, key=lambda x: x['probabilidad_max'], reverse=True)

        for jugador in datos_ordenados:
            self.crear_tarjeta(jugador)

    def crear_tarjeta(self, data):
        color = "#e74c3c" if data["veredicto"] else "#27ae60"
        texto = "⚠️ DETECTADO" if data["veredicto"] else "✓ LIMPIO"
        
        card = ctk.CTkFrame(self.main_view, border_width=2, border_color=color)
        card.pack(fill="x", padx=10, pady=5)

        # Columna 1: Identidad
        ctk.CTkLabel(card, text=data["nombre"], font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=(10,0))
        ctk.CTkLabel(card, text=f"ID: {data['steamid']}", font=("Arial", 10)).pack(anchor="w", padx=10)
        
        # Columna 2: Métricas del Modelo
        stats = ctk.CTkFrame(card, fg_color="transparent")
        stats.pack(side="left", padx=40)
        
        prob_pct = data["probabilidad_max"] * 100
        
        ctk.CTkLabel(stats, text=texto, text_color=color, font=("Arial", 16, "bold")).pack()
        
        # Barra de progreso
        progress = ctk.CTkProgressBar(stats, width=150, height=10, progress_color=color)
        progress.set(data["probabilidad_max"])
        progress.pack(pady=5)
        
        ctk.CTkLabel(stats, text=f"Confianza IA: {prob_pct:.1f}% | Kills Analizadas: {data['kills_analizadas']}").pack()
        ctk.CTkLabel(stats, text=f"Score Ponderado (CSWatch): {data['cswatch_score']:.2f}", text_color="#3498db").pack()

        # Columna 3: Herramientas Externas
        links = ctk.CTkFrame(card, fg_color="transparent")
        links.pack(side="right", padx=20, pady=10)
        
        sid = data['steamid']
        
        self.btn_link(links, "Leetify", "#e812f3", f"https://leetify.com/public/profile/{sid}")
        self.btn_link(links, "CSStats", "#3498db", f"https://csstats.gg/player/{sid}")
        self.btn_link(links, "Steam", "#95a5a6", f"https://steamcommunity.com/profiles/{sid}")

    def btn_link(self, parent, txt, color, url):
        ctk.CTkButton(parent, text=txt, width=80, height=24, fg_color="transparent", 
                      border_width=1, border_color=color, text_color=color,
                      command=lambda: webbrowser.open(url)).pack(side="left", padx=5)

if __name__ == "__main__":
    app = NeuralAntiCheatApp()
    app.mainloop()
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import webbrowser
import threading
import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from demoparser2 import DemoParser

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

COLOR_BG_MAIN = "#121212"
COLOR_SIDEBAR = "#181818"
COLOR_ROW_BG = "#1e1e1e"
COLOR_CT = "#5d79ae"
COLOR_T = "#de9b35"
COLOR_RED_HIGHLIGHT = "#4a1a1a"
COLOR_ACCENT_RED = "#ff5555"
COLOR_ACCENT_GREEN = "#2ecc71"
COLOR_TEXT_GRAY = "#888888"
COLOR_BTN_CSWATCH = "#a30404"
COLOR_PENDING = "#444444"

W_RANK = 40
W_NAME = 200
W_STAT = 50
W_HS = 60
W_IA = 80
W_BTNS = 160

class CS2UltimateAuditor(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("CS2 Anticheat Overwatch")
        self.geometry("1280x960")
        
        self.sidebar_expanded = True
        self.current_folder = ""
        self.current_match_data = None
        self.model = None
        self.scaler = None
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.topbar = ctk.CTkFrame(self, height=50, corner_radius=0, fg_color=COLOR_SIDEBAR)
        self.topbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        
        self.btn_toggle = ctk.CTkButton(self.topbar, text="☰", width=40, height=40, 
                                        fg_color="transparent", font=("Arial", 20),
                                        command=self.toggle_sidebar)
        self.btn_toggle.pack(side="left", padx=10)
        
        ctk.CTkLabel(self.topbar, text="Demo Analyzer", font=("Arial", 16, "bold")).pack(side="left", padx=10)
        self.lbl_status = ctk.CTkLabel(self.topbar, text="Sistema Listo", text_color="gray")
        self.lbl_status.pack(side="right", padx=20)

        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color=COLOR_SIDEBAR)
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        self.btn_folder = ctk.CTkButton(self.sidebar, text="ABRIR CARPETA DEMOS", fg_color="#333", command=self.seleccionar_carpeta)
        self.btn_folder.pack(pady=(20,10), padx=10, fill="x")

        self.file_scroll = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.file_scroll.pack(fill="both", expand=True, padx=5)
        
        self.btn_run = ctk.CTkButton(self.sidebar, text="EJECUTAR ANÁLISIS", height=45, 
                                     fg_color="#c0392b", hover_color="#e74c3c",
                                     state="disabled", font=("Arial", 12, "bold"),
                                     command=self.ejecutar_analisis_ia)
        self.btn_run.pack(pady=20, padx=10, fill="x", side="bottom")

        self.main_panel = ctk.CTkFrame(self, fg_color=COLOR_BG_MAIN)
        self.main_panel.grid(row=1, column=1, sticky="nsew", padx=20, pady=20)
        self.main_panel.grid_rowconfigure(1, weight=1)
        self.main_panel.grid_columnconfigure(0, weight=1)

        self.setup_match_header()
        self.board_scroll = ctk.CTkScrollableFrame(self.main_panel, fg_color="transparent")
        self.board_scroll.grid(row=1, column=0, sticky="nsew")

        self.load_ia_resources()

    def toggle_sidebar(self):
        if self.sidebar_expanded:
            self.sidebar.grid_forget()
        else:
            self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.sidebar_expanded = not self.sidebar_expanded

    def setup_match_header(self):
        self.match_header_frame = ctk.CTkFrame(self.main_panel, fg_color="transparent", height=80)
        self.match_header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.info_left = ctk.CTkFrame(self.match_header_frame, fg_color="transparent")
        self.info_left.pack(side="left")
        self.lbl_map = ctk.CTkLabel(self.info_left, text="MAPA: --", font=("Arial", 14, "bold"), text_color="gray")
        self.lbl_map.pack(anchor="w")
        self.lbl_server = ctk.CTkLabel(self.info_left, text="SERVER: --", font=("Arial", 11), text_color="gray")
        self.lbl_server.pack(anchor="w")

        self.score_box = ctk.CTkFrame(self.match_header_frame, fg_color="#1f1f1f", corner_radius=10)
        self.score_box.pack(side="left", expand=True, padx=20)
        
        self.lbl_name_ct = ctk.CTkLabel(self.score_box, text="CT", font=("Arial", 12, "bold"), text_color=COLOR_CT)
        self.lbl_name_ct.pack(side="left", padx=(20, 5))
        self.lbl_score_ct = ctk.CTkLabel(self.score_box, text="--", font=("Arial", 40, "bold"), text_color=COLOR_CT)
        self.lbl_score_ct.pack(side="left", padx=5)
        
        ctk.CTkLabel(self.score_box, text="-", font=("Arial", 30, "bold"), text_color="gray").pack(side="left", padx=10)
        
        self.lbl_score_t = ctk.CTkLabel(self.score_box, text="--", font=("Arial", 40, "bold"), text_color=COLOR_T)
        self.lbl_score_t.pack(side="left", padx=5)
        self.lbl_name_t = ctk.CTkLabel(self.score_box, text="T", font=("Arial", 12, "bold"), text_color=COLOR_T)
        self.lbl_name_t.pack(side="left", padx=(5, 20))

    def load_ia_resources(self):
        self.numeric_cols = [
            "X", "Y", "Z", "pitch", "yaw", "usercmd_mouse_dx", "usercmd_mouse_dy", 
            "is_walking", "is_scoped", "is_airborne", "active_weapon", "health", "armor_value"
        ]      
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'anticheat_model_kill_focused.keras')
            scaler_path = os.path.join(script_dir, 'scaler_entrenado.pkl')
            columns_path = os.path.join(script_dir, 'columnas_numericas.pkl')
            
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                if os.path.exists(scaler_path):
                    try:
                        self.scaler = joblib.load(scaler_path)
                    except ImportError:
                        self.scaler = None
                
                if os.path.exists(columns_path):
                    self.numeric_cols = joblib.load(columns_path)
                
                print(f"IA Cargada desde: {model_path}")
            else:
                print(f"No se encontró '{model_path}'. Modo visualización activado.")
                self.lbl_status.configure(text="Solo visualización", text_color="orange")
        except Exception as e:
            print(f"Error cargando IA: {e}")
            self.model = None
            self.scaler = None

    def seleccionar_carpeta(self):
        folder = filedialog.askdirectory()
        if folder:
            self.current_folder = folder
            self.listar_archivos()

    def listar_archivos(self):
        for w in self.file_scroll.winfo_children(): w.destroy()
        try:
            demos = [f for f in os.listdir(self.current_folder) if f.endswith(".dem")]
            if not demos:
                ctk.CTkLabel(self.file_scroll, text="No hay .dem", text_color="gray").pack()
                return
            for d in demos:
                btn = ctk.CTkButton(self.file_scroll, text=d, fg_color="transparent", 
                                    border_width=1, border_color="#333", anchor="w", height=25,
                                    font=("Arial", 11),
                                    command=lambda f=d: self.cargar_demo_real(f))
                btn.pack(fill="x", pady=1)
        except: pass

    def cargar_demo_real(self, filename):
        self.lbl_status.configure(text=f"Leyendo {filename}...", text_color="#3498db")
        self.btn_run.configure(state="disabled", text="CARGANDO...")
        
        for w in self.board_scroll.winfo_children(): w.destroy()
        self.lbl_score_ct.configure(text="-")
        self.lbl_score_t.configure(text="-")
        
        threading.Thread(target=self.thread_parser, args=(filename,)).start()

    def thread_parser(self, filename):
        try:
            full_path = os.path.join(self.current_folder, filename)
            print(f"Iniciando parsing de: {full_path}")

            parser = DemoParser(full_path)
            
            try:
                header = parser.parse_header()
                map_name = header.get("map_name", "Unknown")
                server_name = header.get("server_name", "Unknown")
            except:
                map_name = "Unknown"
                server_name = "Unknown"

            print("Extrayendo GameState (Score y Nombres)...")
            try:
                df_gamestate = parser.parse_ticks(["team_rounds_total", "team_num", "team_name"])
                
                score_t, score_ct = 0, 0
                name_t, name_ct = "TERRORISTS", "COUNTER-TERRORISTS"

                if not df_gamestate.empty:
                    t_data = df_gamestate[df_gamestate['team_num'] == 2]
                    ct_data = df_gamestate[df_gamestate['team_num'] == 3]

                    if not t_data.empty:
                        score_t = t_data['team_rounds_total'].max()
                        last_name = t_data['team_name'].iloc[-1]
                        if last_name and str(last_name) != "nan": name_t = str(last_name)

                    if not ct_data.empty:
                        score_ct = ct_data['team_rounds_total'].max()
                        last_name = ct_data['team_name'].iloc[-1]
                        if last_name and str(last_name) != "nan": name_ct = str(last_name)
            except Exception as e:
                print(f"Advertencia extrayendo GameState: {e}")
                score_t, score_ct = 0, 0
                name_t, name_ct = "T", "CT"

            fields = ["player_name", "player_steamid", "team_num", "kills_total", "deaths_total", "assists_total", "headshot_kills_total", "rank"]
            
            print("Extrayendo stats jugadores...")
            df_stats = parser.parse_ticks(fields)

            if df_stats is None or df_stats.empty:
                raw_events = parser.parse_events(["player_death"], other=["total_rounds_played"])
                if isinstance(raw_events, list): raw_events = pd.DataFrame(raw_events)
                if raw_events.empty: raise ValueError("Demo vacía o ilegible.")
                 
            if not df_stats.empty:
                df_final_stats = df_stats.sort_values("tick").groupby("player_steamid").tail(1)
            else:
                df_final_stats = pd.DataFrame()

            team_a = []
            team_b = []
            
            for _, row in df_final_stats.iterrows():
                if pd.isna(row['player_steamid']) or row['player_steamid'] == 0: continue
                
                p_data = {
                    "name": str(row.get("player_name", "Unknown")),
                    "steamid": str(int(row.get("player_steamid", 0))),
                    "rank": str(int(row.get("rank", 0))),
                    "k": int(row.get("kills_total", 0)),
                    "d": int(row.get("deaths_total", 0)),
                    "a": int(row.get("assists_total", 0)),
                    "hs": int(row.get("headshot_kills_total", 0)),
                    "team": int(row.get("team_num", 0)),
                    "prob": None
                }
                
                if p_data["team"] == 2:
                    team_b.append(p_data)
                elif p_data["team"] == 3:
                    team_a.append(p_data)

            team_a.sort(key=lambda x: x['k'], reverse=True)
            team_b.sort(key=lambda x: x['k'], reverse=True)

            meta = {
                "mapa": map_name, 
                "server": server_name, 
                "score_ct": int(score_ct), 
                "score_t": int(score_t),
                "name_ct": name_ct,
                "name_t": name_t
            }
            
            self.current_match_data = {
                "meta": meta, 
                "team_a": team_a, 
                "team_b": team_b,
                "parser": parser, 
                "full_path": full_path
            }

            self.after(0, lambda: self.render_scoreboard(meta, team_a, team_b, analyzed=False))

        except Exception as e:
            print(f"Error crítico: {e}")
            self.after(0, lambda: self.lbl_status.configure(text="Error lectura demo", text_color="red"))
            self.after(0, lambda: self.btn_run.configure(state="normal", text="ERROR"))

    def ejecutar_analisis_ia(self):
        if not self.current_match_data: return
        if not self.model:
            messagebox.showwarning("Error", "Modelo no cargado.")
            return
        
        self.btn_run.configure(state="disabled", text="PROCESANDO IA...")
        self.lbl_status.configure(text="Extrayendo telemetría...", text_color="#f1c40f")
        threading.Thread(target=self.thread_ia_inference_real).start()

    def thread_ia_inference_real(self):
        try:
            parser = self.current_match_data['parser']
            
            print("Extrayendo eventos de muerte desde ticks...")
            try:
                kill_fields = [
                    "tick",
                    "player_steamid",
                    "player_name",
                    "kills_total",
                    "team_num",
                    "X", "Y", "Z", "pitch", "yaw", 
                    "usercmd_mouse_dx", "usercmd_mouse_dy",
                    "is_walking", "is_scoped", "is_airborne",
                    "active_weapon", "health", "armor_value",

                    "aim_punch_angle", "FIRE",
                    "spotted", "approximate_spotted_by"
                ]
                
                print("Extrayendo todos los ticks del juego...")
                df_all_data = parser.parse_ticks(kill_fields)
                
                if df_all_data is None or df_all_data.empty:
                    raise ValueError("No se pudieron extraer datos de ticks")
                
                print(f"Extraídos {len(df_all_data)} ticks")
                

                print("Aplicando feature engineering...")
                df_all_data['variance_pre_kill'] = 0.0
                df_all_data['suma_abs_pre_kill'] = 0.0
                
                if 'aim_punch_angle' in df_all_data.columns:
                    df_all_data['aim_punch_angle'] = df_all_data['aim_punch_angle'].astype(str).fillna('[0 0]')
                    split_df = df_all_data['aim_punch_angle'].str.strip('[]').str.split(expand=True)
                    df_all_data[['aim_punch_x', 'aim_punch_y']] = split_df.iloc[:, :2]
                    df_all_data[['aim_punch_x', 'aim_punch_y']] = df_all_data[['aim_punch_x', 'aim_punch_y']].apply(pd.to_numeric, errors='coerce').fillna(0)
                    df_all_data = df_all_data.drop(columns=['aim_punch_angle'])
                else:
                    df_all_data['aim_punch_x'] = 0.0
                    df_all_data['aim_punch_y'] = 0.0

                for col in ['is_walking', 'FIRE', 'is_scoped']:
                    if col in df_all_data.columns:
                        df_all_data[col] = (df_all_data[col].notna() & (df_all_data[col] != 'False')).astype(int)
                
                df_all_data = df_all_data.drop(columns=['spotted', 'approximate_spotted_by'], errors='ignore')

                columnas_numericas = df_all_data.select_dtypes(include=np.number).columns
                for col in columnas_numericas:
                    if df_all_data[col].isnull().any():
                        df_all_data[col] = df_all_data[col].fillna(df_all_data[col].median())
                
                print(f"Feature engineering completado")
                

                print("Detectando kills...")
                df_all_data = df_all_data.sort_values(['player_steamid', 'tick'])
                
                def normalize_steamid(sid):
                    if pd.isna(sid) or sid is None:
                        return '0'
                    sid_str = str(sid).split('.')[0]
                    return sid_str if sid_str != 'nan' else '0'
                
                for col in ['is_walking', 'FIRE', 'is_scoped']:
                    if col in df_all_data.columns:
                        df_all_data[col] = (df_all_data[col].notna() & (df_all_data[col] != False) & (df_all_data[col] != 'False')).astype(int)
                
                if 'aim_punch_angle' in df_all_data.columns:
                    df_all_data['aim_punch_angle'] = df_all_data['aim_punch_angle'].astype(str).fillna('[0 0]')
                    split_df = df_all_data['aim_punch_angle'].str.strip('[]').str.split(expand=True)
                    df_all_data[['aim_punch_x', 'aim_punch_y']] = split_df.iloc[:, :2]
                    df_all_data[['aim_punch_x', 'aim_punch_y']] = df_all_data[['aim_punch_x', 'aim_punch_y']].apply(pd.to_numeric, errors='coerce').fillna(0)
                    df_all_data = df_all_data.drop(columns=['aim_punch_angle'])
                
                df_all_data['player_steamid'] = df_all_data['player_steamid'].apply(normalize_steamid)
                
                df_all_data['kill_diff'] = df_all_data.groupby('player_steamid')['kills_total'].diff()
                
                kills_ticks = df_all_data[df_all_data['kill_diff'] > 0].copy()
                
                print("Calculando features de varianza pre-kill...")
                for idx, kill_event in kills_ticks.iterrows():
                    steam_id = kill_event['player_steamid']
                    kill_tick = kill_event['tick']
                    start_tick = kill_tick - 192
                    
                    pre_kill_window = df_all_data[
                        (df_all_data['player_steamid'] == steam_id) & 
                        (df_all_data['tick'] >= start_tick) & 
                        (df_all_data['tick'] < kill_tick)
                    ]
                    
                    if not pre_kill_window.empty:
                        variance = np.nansum([
                            pre_kill_window['usercmd_mouse_dx'].var(),
                            pre_kill_window['usercmd_mouse_dy'].var()
                        ])
                        abs_sum = (
                            pre_kill_window['usercmd_mouse_dx'].abs().sum() + 
                            pre_kill_window['usercmd_mouse_dy'].abs().sum()
                        )
                        df_all_data.loc[idx, 'variance_pre_kill'] = variance
                        df_all_data.loc[idx, 'suma_abs_pre_kill'] = abs_sum
                
                df_all_data = df_all_data.drop(columns=['kill_diff'])
                
                print(f"Total kills detectados: {len(kills_ticks)}")
                
                if kills_ticks.empty:
                    raise ValueError("No se detectaron kills en la demo")
                
            except Exception as e:
                print(f"Error extrayendo datos: {e}")
                self.after(0, lambda: self.lbl_status.configure(text="Error: Demo sin kills", text_color="red"))
                self.after(0, lambda: self.btn_run.configure(state="normal", text="ERROR"))
                return

            player_windows = {}
            
            print("Preparando telemetría para análisis...")

            df_telemetry = df_all_data
            
            columnas_excluir = ['player_steamid', 'player_name', 'kills_total', 'team_num', 'tick']
            available_numeric_cols = [col for col in df_telemetry.select_dtypes(include=np.number).columns 
                                     if col not in columnas_excluir]

            if hasattr(self, 'numeric_cols') and self.numeric_cols:
                print(f"Usando columnas del modelo: {len(self.numeric_cols)} features esperadas")
                print(f"Columnas disponibles en demo: {len(available_numeric_cols)}")
                model_expected_cols = self.numeric_cols
            else:
                print(f"Modo dinámico: usando {len(available_numeric_cols)} features disponibles")
                model_expected_cols = available_numeric_cols
            
            print(f"SteamIDs únicos en telemetría: {df_telemetry['player_steamid'].nunique()}")
            
            kills_processed = 0
            kills_skipped = 0
            
            for idx, kill_event in kills_ticks.iterrows():
                attacker_id = kill_event['player_steamid']
                death_tick = kill_event['tick']

                if idx < 3 or kills_processed < 3:
                    print(f"Kill #{kills_processed}: attacker={attacker_id}, tick={death_tick}")
                
                if attacker_id == '0' or attacker_id == 'None':
                    kills_skipped += 1
                    continue

                start_tick = death_tick - 224
                end_tick = death_tick + 32
                
                window = df_telemetry[
                    (df_telemetry['tick'] >= start_tick) & 
                    (df_telemetry['tick'] <= end_tick) & 
                    (df_telemetry['player_steamid'] == attacker_id)
                ].sort_values('tick')
                
                if kills_processed < 3:
                    print(f"Ventana: {len(window)} ticks encontrados")

                window_aligned = pd.DataFrame()
                
                for col in model_expected_cols:
                    if col in window.columns:
                        col_data = pd.to_numeric(window[col], errors='coerce').fillna(0.0)
                        window_aligned[col] = col_data
                    else:
                        window_aligned[col] = 0.0
                
                if kills_processed < 1:
                    non_numeric = window_aligned.select_dtypes(include=['object']).columns.tolist()
                    if non_numeric:
                        print(f"Columnas NO numéricas detectadas: {non_numeric}")
                        for col in non_numeric:
                            print(f"{col}: muestra = {window_aligned[col].head(3).tolist()}")
                            window_aligned[col] = pd.to_numeric(window_aligned[col], errors='coerce').fillna(0.0)
                
                window_aligned = window_aligned.astype('float32')
                
                if len(window_aligned.columns) == 0:
                    if kills_processed < 3:
                        print(f"Sin features después de alineación")
                    kills_skipped += 1
                    continue
                
                window_features = window_aligned
                
                if kills_processed < 3:
                    print(f"Features: {len(window_features.columns)} columnas × {len(window_features)} ticks") 

                window_len = len(window_features)
                num_features = len(window_features.columns)
                
                if window_len > 0:
                    kills_processed += 1
                    
                    if window_len < 256:
                        padding_needed = 256 - window_len
                        padding = np.zeros((padding_needed, num_features))
                        window_array = np.vstack([window_features.to_numpy(), padding])
                    elif window_len > 256:
                        window_array = window_features.to_numpy()[:256]
                    else:
                        window_array = window_features.to_numpy()
                    
                    if attacker_id not in player_windows: 
                        player_windows[attacker_id] = []
                    
                    player_windows[attacker_id].append(window_array)
                else:
                    kills_skipped += 1

            print(f"Kills procesados: {kills_processed}")
            print(f"Kills descartados: {kills_skipped}")
            print(f"Jugadores con datos: {len(player_windows)}")

            print("Prediciendo...")

            all_feature_stats = {}
            
            for team in [self.current_match_data['team_a'], self.current_match_data['team_b']]:
                for player in team:
                    sid = player['steamid']
                    if sid in player_windows and len(player_windows[sid]) > 0:
                        batch = np.array(player_windows[sid])
                        preds = self.model.predict(batch, verbose=0)
                        
                        player['individual_probs'] = [float(p[0]) for p in preds]
                        player['prob'] = float(np.mean(preds))
                        player['prob_max'] = float(np.max(preds))
                        player['prob_min'] = float(np.min(preds))
                        player['prob_std'] = float(np.std(preds))
                        
                        feature_means = np.mean(batch, axis=(0, 1))
                        all_feature_stats[sid] = feature_means
                        
                        print(f"{player['name']}: {len(player_windows[sid])} kills analizados")
                        print(f"Promedio: {int(player['prob']*100)}% | Max: {int(player['prob_max']*100)}% | Min: {int(player['prob_min']*100)}% | Std: {player['prob_std']:.2f}")
                        
                        sorted_probs = sorted(player['individual_probs'], reverse=True)
                        top_3 = sorted_probs[:min(3, len(sorted_probs))]
                        if len(top_3) > 0:
                            print(f"Top 3 kills sospechosos: {', '.join([f'{int(p*100)}%' for p in top_3])}")
                    else:
                        player['prob'] = 0.0
                        player['individual_probs'] = []
                        print(f"{player['name']}: SIN DATOS (0%)")
            
            print("\nANÁLISIS COMPARATIVO DE FEATURES:")
            if len(all_feature_stats) >= 2:
                suspicious_sids = []
                legit_sids = []
                for team in [self.current_match_data['team_a'], self.current_match_data['team_b']]:
                    for player in team:
                        prob_avg = player.get('prob', 0)
                        prob_max = player.get('prob_max', 0)
                        # Criterio mixto: promedio > 0.38 OR max > 0.50
                        if (prob_avg > 0.38 or prob_max > 0.50) and player['steamid'] in all_feature_stats:
                            suspicious_sids.append(player['steamid'])
                        elif prob_avg > 0 and player['steamid'] in all_feature_stats:
                            legit_sids.append(player['steamid'])
                
                if len(suspicious_sids) > 0 and len(legit_sids) > 0:
                    sus_features = np.mean([all_feature_stats[sid] for sid in suspicious_sids], axis=0)
                    legit_features = np.mean([all_feature_stats[sid] for sid in legit_sids], axis=0)
                    
                    diffs = np.abs(sus_features - legit_features)
                    top_indices = np.argsort(diffs)[-5:][::-1]
                    
                    if hasattr(self, 'columnas_numericas') and self.columnas_numericas:
                        feature_names = self.columnas_numericas
                    else:
                        feature_names = [f"Feature_{i}" for i in range(len(diffs))]
                    
                    print(f"Jugadores sospechosos: {len(suspicious_sids)} | Legítimos: {len(legit_sids)}")
                    print(f"Top 5 features más discriminativas:")
                    for idx in top_indices:
                        fname = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                        print(f"{fname}: Sospechosos={sus_features[idx]:.3f} vs Legítimos={legit_features[idx]:.3f} (diff={diffs[idx]:.3f})")
                else:
                    print(f"No hay suficientes jugadores en ambos grupos para comparar")
                    print(f"Sospechosos: {len(suspicious_sids)}, Legítimos: {len(legit_sids)}")
            else:
                print(f"Insuficientes datos para análisis comparativo")

            self.after(0, lambda: self.render_scoreboard(
                self.current_match_data['meta'], 
                self.current_match_data['team_a'], 
                self.current_match_data['team_b'], 
                analyzed=True
            ))
            
        except Exception as e:
            print(f"Error de Análisis: {e}")
            self.after(0, lambda: self.lbl_status.configure(text="Error de Análisis", text_color="red"))
            self.after(0, lambda: self.btn_run.configure(state="normal"))

    def render_scoreboard(self, meta, team_a, team_b, analyzed):
        self.lbl_map.configure(text=f"MAPA: {meta['mapa']}")
        self.lbl_server.configure(text=f"SERVER: {meta['server']}")

        self.lbl_score_ct.configure(text=str(meta['score_ct']))
        self.lbl_score_t.configure(text=str(meta['score_t']))
        self.lbl_name_ct.configure(text=meta['name_ct'])
        self.lbl_name_t.configure(text=meta['name_t'])

        for w in self.board_scroll.winfo_children(): w.destroy()

        self.crear_tabla_equipo(COLOR_CT, meta['name_ct'], team_a, analyzed)
        ctk.CTkFrame(self.board_scroll, height=20, fg_color="transparent").pack()
        self.crear_tabla_equipo(COLOR_T, meta['name_t'], team_b, analyzed)

        if not analyzed:
            if self.model:
                self.btn_run.configure(state="normal", text="▶ EJECUTAR ANÁLISIS")
                self.lbl_status.configure(text="Datos cargados. Listo para el Análisis.", text_color="white")
            else:
                self.btn_run.configure(state="disabled", text="NO HAY MODELO")
        else:
            self.btn_run.configure(state="disabled", text="ANÁLISIS COMPLETO")
            self.lbl_status.configure(text="Análisis Finalizado", text_color=COLOR_ACCENT_GREEN)

    def crear_tabla_equipo(self, team_color, team_name, players, analyzed):
        header_frame = ctk.CTkFrame(self.board_scroll, fg_color=team_color, height=30, corner_radius=3)
        header_frame.pack(fill="x", pady=(0,0))
        ctk.CTkLabel(header_frame, text=f"  {team_name}", font=("Arial", 12, "bold"), text_color="white").pack(side="left", pady=2)

        cols = ctk.CTkFrame(self.board_scroll, fg_color="#1a1a1a", height=30)
        cols.pack(fill="x")
        self.add_col(cols, "", W_RANK, "center")
        self.add_col(cols, "JUGADOR", W_NAME, "w")
        self.add_col(cols, "K", W_STAT, "center")
        self.add_col(cols, "D", W_STAT, "center")
        self.add_col(cols, "A", W_STAT, "center")
        self.add_col(cols, "+/-", W_STAT, "center")
        self.add_col(cols, "HS%", W_HS, "center")
        self.add_col(cols, "IA PROB", W_IA, "center")
        self.add_col(cols, "ENLACES", W_BTNS, "e")

        for p in players:
            self.crear_fila_jugador(p, analyzed)

    def crear_fila_jugador(self, data, analyzed):
        is_cheater = False
        bg_color = COLOR_ROW_BG
        
        if not analyzed:
            prob_text = "?"
            badge_bg = COLOR_PENDING
        else:
            prob = data.get('prob', 0)
            prob_max = data.get('prob_max', 0)
            prob_text = f"{int(prob*100)}%"
            if (prob and prob > 0.38) or (prob_max and prob_max > 0.50):
                is_cheater = True
                bg_color = COLOR_RED_HIGHLIGHT
                badge_bg = COLOR_ACCENT_RED
            else:
                badge_bg = COLOR_ACCENT_GREEN

        row = ctk.CTkFrame(self.board_scroll, fg_color=bg_color, height=45, corner_radius=4)
        row.pack(fill="x", pady=2, padx=5)
        row.pack_propagate(False)

        name_color = COLOR_ACCENT_RED if is_cheater else "white"
        icon = "⚠️ " if is_cheater else ""
        
        diff = data['k'] - data['d']
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        diff_color = COLOR_ACCENT_GREEN if diff > 0 else (COLOR_ACCENT_RED if diff < 0 else "gray")
        hs_pct = int((data['hs'] / data['k']) * 100) if data['k'] > 0 else 0

        self.add_cell(row, str(data['rank']), W_RANK, "center", "gray", font_size=9)
        self.add_cell(row, f"{icon}{data['name']}", W_NAME, "w", name_color, font_size=13, bold=True)
        self.add_cell(row, str(data['k']), W_STAT, "center")
        self.add_cell(row, str(data['d']), W_STAT, "center")
        self.add_cell(row, str(data['a']), W_STAT, "center")
        self.add_cell(row, diff_str, W_STAT, "center", diff_color)
        self.add_cell(row, f"{hs_pct}%", W_HS, "center")

        cell_ia = ctk.CTkFrame(row, fg_color="transparent", width=W_IA, height=40)
        cell_ia.pack(side="left")
        ctk.CTkLabel(cell_ia, text=prob_text, fg_color=badge_bg, corner_radius=6, 
                     font=("Arial", 11, "bold"), width=50).place(relx=0.5, rely=0.5, anchor="center")

        cell_btns = ctk.CTkFrame(row, fg_color="transparent", width=W_BTNS, height=40)
        cell_btns.pack(side="right")
        sid = data.get('steamid', '0')
        self.mini_btn(cell_btns, "C", "#3498db", f"https://csstats.gg/player/{sid}")
        self.mini_btn(cell_btns, "L", "#f312e8", f"https://leetify.com/public/profile/{sid}")
        self.mini_btn(cell_btns, "S", "#030a8d", f"https://steamcommunity.com/profiles/{sid}")
        self.mini_btn(cell_btns, "W", COLOR_BTN_CSWATCH, f"https://wsteamcommunity.com/profiles/{sid}", True)

    def add_col(self, parent, text, width, anchor):
        f = ctk.CTkFrame(parent, fg_color="transparent", width=width, height=25)
        f.pack(side="left" if anchor != "e" else "right")
        f.pack_propagate(False)
        ctk.CTkLabel(f, text=text, font=("Arial", 9, "bold"), text_color="gray", anchor="w" if anchor=="w" else "center").place(relx=0.5, rely=0.5, anchor="center")

    def add_cell(self, parent, text, width, anchor, color="white", font_size=11, bold=False):
        f = ctk.CTkFrame(parent, fg_color="transparent", width=width, height=35)
        f.pack(side="left")
        f.pack_propagate(False)
        font_weight = "bold" if bold else "normal"
        
        if anchor == "w":
            align = "w"
            x_pos = 5
            relx_pos = 0.0
        elif anchor == "e":
            align = "e"
            x_pos = -5
            relx_pos = 1.0
        else:
            align = "center"
            x_pos = 0
            relx_pos = 0.5
            
        ctk.CTkLabel(f, text=text, font=("Arial", font_size, font_weight), text_color=color, anchor=align).place(relx=relx_pos, rely=0.5, anchor=align, x=x_pos)

    def mini_btn(self, parent, text, color, url, bold=False):
        font_w = "bold" if bold else "normal"
        btn = ctk.CTkButton(parent, text=text, width=28, height=22, fg_color="transparent", 
                            border_width=1, border_color=color, text_color=color, hover_color="#333",
                            font=("Arial", 9, font_w), command=lambda: webbrowser.open(url))
        btn.pack(side="right", padx=2, pady=7)

if __name__ == "__main__":
    app = CS2UltimateAuditor()
    app.mainloop()
import psutil
import time
import json
import requests
import threading
import win32gui
import win32process
from pynput import mouse, keyboard
from fastapi import FastAPI, Request
import uvicorn

estado_del_juego = {
    "proceso_activo": False,
    "ventana_activa": False,
    "partida_en_vivo": False
}
last_net_stats = None
mouse_movements = []
estado_jugador = { 'is_firing': False, 'is_scoped': False, 'is_walking': False }

gsi_app = FastAPI()

@gsi_app.post("/")
async def recibir_datos_gsi(request: Request):
    try:
        data = await request.json()
        is_live_now = False
        if 'round' in data and 'phase' in data['round']:
            if data['round']['phase'] == 'live':
                is_live_now = True
        
        if is_live_now != estado_del_juego["partida_en_vivo"]:
            estado_del_juego["partida_en_vivo"] = is_live_now
            status_msg = "INICIANDO" if is_live_now else "PAUSANDO"
            print(f"\n--- Partida {'EN VIVO' if is_live_now else 'NO ACTIVA'} detectada por GSI! {status_msg} recolección. ---", flush=True)
            
    except (json.JSONDecodeError, KeyError):
        pass
    return {"status": "ok"}

def iniciar_servidor_gsi():
    uvicorn.run(gsi_app, host="127.0.0.1", port=3000, log_level="warning")

def on_move(x, y):
    if not all(estado_del_juego.values()): return
    if hasattr(on_move, 'last_pos') and on_move.last_pos is not None:
        dx = x - on_move.last_pos['x']
        dy = y - on_move.last_pos['y']
        mouse_movements.append({'dx': dx, 'dy': dy, 'ts': time.perf_counter()})
    on_move.last_pos = {'x': x, 'y': y}
on_move.last_pos = None

def on_click(x, y, button, pressed):
    if not all(estado_del_juego.values()): return
    if button == mouse.Button.left:
        estado_jugador['is_firing'] = pressed
    elif button == mouse.Button.right:
        estado_jugador['is_scoped'] = pressed

def on_press(key):
    if not all(estado_del_juego.values()): return
    if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
        estado_jugador['is_walking'] = True

def on_release(key):
    if not all(estado_del_juego.values()): return
    if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
        estado_jugador['is_walking'] = False

def actualizar_estado_ventana():
    proceso_encontrado = "cs2.exe" in (p.name() for p in psutil.process_iter())
    ventana_activa = False
    if proceso_encontrado:
        try:
            hwnd = win32gui.GetForegroundWindow()
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid > 0 and psutil.Process(pid).name() == "cs2.exe":
                ventana_activa = True
        except (psutil.NoSuchProcess, KeyError):
            pass
    
    estado_del_juego["proceso_activo"] = proceso_encontrado
    estado_del_juego["ventana_activa"] = ventana_activa

def obtener_datos_de_red(intervalo):
    global last_net_stats
    datos_red = {
        'ip_servidor': None, 'puerto_servidor': None, 'ubicacion_servidor': 'Desconocida',
        'kb_enviados_s': 0.0, 'kb_recibidos_s': 0.0
    }
    for proc in psutil.process_iter(['pid', 'name']):
        if 'cs2.exe' == proc.info['name']:
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.status == psutil.CONN_ESTABLISHED and conn.raddr:
                        datos_red['ip_servidor'] = conn.raddr.ip
                        datos_red['puerto_servidor'] = conn.raddr.port
                        break
                if datos_red['ip_servidor']:
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    if datos_red['ip_servidor']:
        try:
            response = requests.get(f"https://ipinfo.io/{datos_red['ip_servidor']}/json", timeout=1)
            if response.status_code == 200:
                geo_data = response.json()
                datos_red['ubicacion_servidor'] = f"{geo_data.get('city', '')}, {geo_data.get('country', '')}"
        except requests.exceptions.RequestException:
            datos_red['ubicacion_servidor'] = 'Fallo en geolocalización'

    current_net_stats = psutil.net_io_counters()
    if last_net_stats is not None:
        bytes_enviados = current_net_stats.bytes_sent - last_net_stats.bytes_sent
        bytes_recibidos = current_net_stats.bytes_recv - last_net_stats.bytes_recv
        datos_red['kb_enviados_s'] = round((bytes_enviados / 1024) / intervalo, 2)
        datos_red['kb_recibidos_s'] = round((bytes_recibidos / 1024) / intervalo, 2)
    last_net_stats = current_net_stats
    return datos_red

if __name__ == "__main__":
    print("Cliente anti-cheat final iniciado. Presiona Ctrl+C para detener.", flush=True)

    gsi_thread = threading.Thread(target=iniciar_servidor_gsi, daemon=True)
    gsi_thread.start()
    print("Servidor GSI escuchando en http://127.0.0.1:3000...", flush=True)

    mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    mouse_listener.start()
    keyboard_listener.start()
    
    INTERVALO_SEGUNDOS = 5
    
    try:
        while True:
            actualizar_estado_ventana()
            
            if all(estado_del_juego.values()):
                print("\n" + "="*80, flush=True)
                
                datos_red_intervalo = obtener_datos_de_red(INTERVALO_SEGUNDOS)
                datos_raton_intervalo = list(mouse_movements)
                mouse_movements.clear()
                
                paquete_de_datos = {
                    "timestamp_utc": time.time(),
                    "network_data": datos_red_intervalo,
                    "player_state": estado_jugador.copy(),
                    "mouse_data": datos_raton_intervalo
                }
                
                json_para_procesar = json.dumps(paquete_de_datos, indent=4)
                print("--- PAQUETE DE DATOS UNIFICADO LISTO PARA EL MOTOR DE IA ---")
                print(json_para_procesar, flush=True)
                print("="*80, flush=True)

            else:
                print(f"\rEn espera: Proceso={estado_del_juego['proceso_activo']}, Ventana={estado_del_juego['ventana_activa']}, Partida en Vivo={estado_del_juego['partida_en_vivo']} ({time.strftime('%H:%M:%S')})", end="", flush=True)
            
            time.sleep(INTERVALO_SEGUNDOS)

    except KeyboardInterrupt:
        print("\nDeteniendo cliente...", flush=True)
        mouse_listener.stop()
        keyboard_listener.stop()
        print("Cliente detenido por el usuario.", flush=True)
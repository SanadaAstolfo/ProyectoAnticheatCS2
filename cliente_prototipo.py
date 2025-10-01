import psutil
import time
import json
import requests
from pynput import mouse, keyboard
import win32gui
import win32process

juego_en_ejecucion = False
last_net_stats = None
mouse_movements = []
estado_jugador = {
    'is_firing': False,
    'is_scoped': False,
    'is_walking': False,
}

def on_move(x, y):
    if not juego_en_ejecucion: return
    if hasattr(on_move, 'last_pos') and on_move.last_pos is not None:
        dx = x - on_move.last_pos['x']
        dy = y - on_move.last_pos['y']
        mouse_movements.append({'dx': dx, 'dy': dy, 'ts': time.perf_counter()})
    on_move.last_pos = {'x': x, 'y': y}
on_move.last_pos = None

def on_click(x, y, button, pressed):
    if not juego_en_ejecucion: return
    if button == mouse.Button.left:
        estado_jugador['is_firing'] = pressed
    elif button == mouse.Button.right:
        estado_jugador['is_scoped'] = pressed

def on_press(key):
    if not juego_en_ejecucion: return
    if isinstance(key, keyboard.Key) and key.name.startswith('shift'):
        estado_jugador['is_walking'] = True

def on_release(key):
    if not juego_en_ejecucion: return
    if isinstance(key, keyboard.Key) and key.name.startswith('shift'):
        estado_jugador['is_walking'] = False

mouse_move_listener = mouse.Listener(on_move=on_move, on_click=on_click)
keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
mouse_move_listener.start()
keyboard_listener.start()

def actualizar_estado_juego():
    global juego_en_ejecucion
    proceso_encontrado = "cs2.exe" in (p.name() for p in psutil.process_iter())
    ventana_activa = False
    if proceso_encontrado:
        try:
            pid = win32process.GetWindowThreadProcessId(win32gui.GetForegroundWindow())
            proceso_activo = psutil.Process(pid[-1]).name()
            if proceso_activo == "cs2.exe":
                ventana_activa = True
        except (psutil.NoSuchProcess, KeyError):
            pass
    estado_actual = proceso_encontrado and ventana_activa
    if estado_actual and not juego_en_ejecucion:
        print("¡Juego CS2 detectado y en primer plano! Iniciando recolección de datos...")
    elif not estado_actual and juego_en_ejecucion:
        print("El juego CS2 se ha cerrado o ya no está en primer plano. Pausando recolección.")
    juego_en_ejecucion = estado_actual

def obtener_datos_de_red(intervalo):
    global last_net_stats
    datos_red = {
        'ip_servidor': None,
        'puerto_servidor': None,
        'ubicacion_servidor': 'Desconocida',
        'kb_enviados_s': 0,
        'kb_recibidos_s': 0
    }
    for proc in psutil.process_iter(['pid', 'name']):
        if 'cs2.exe' in proc.info['name']:
            try:
                conexiones = proc.net_connections(kind='inet')
                for conn in conexiones:
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
                city = geo_data.get('city', '')
                country = geo_data.get('country', '')
                datos_red['ubicacion_servidor'] = f"{city}, {country}"
        except requests.exceptions.RequestException:
            datos_red['ubicacion_servidor'] = 'Fallo en geolocalización'

    current_net_stats = psutil.net_io_counters()
    if last_net_stats is not None:
        bytes_enviados = current_net_stats.bytes_sent - last_net_stats.bytes_sent
        bytes_recibidos = current_net_stats.bytes_recv - last_net_stats.bytes_recv
        datos_red['kb_enviados_s'] = (bytes_enviados / 1024) / intervalo
        datos_red['kb_recibidos_s'] = (bytes_recibidos / 1024) / intervalo
    last_net_stats = current_net_stats
    return datos_red

print("Cliente anti-cheat unificado iniciado. Presiona Ctrl+C para detener.")
INTERVALO_SEGUNDOS = 5

try:
    while True:
        actualizar_estado_juego()
        if juego_en_ejecucion:
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
            print(json_para_procesar)
        else:
            print(f"CS2 no detectado en primer plano. Esperando...")
        time.sleep(INTERVALO_SEGUNDOS)
except KeyboardInterrupt:
    mouse_move_listener.stop()
    keyboard_listener.stop()
    print("\nCliente detenido por el usuario.")
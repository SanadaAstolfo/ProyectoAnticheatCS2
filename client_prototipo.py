import pandas as pd
import numpy as np
import time
import json
import psutil
from pynput import mouse, keyboard

juego_en_ejecucion = False
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

# NUEVO: Listener para los clics del ratón
def on_click(x, y, button, pressed):
    if not juego_en_ejecucion: return
    if button == mouse.Button.left:
        estado_jugador['is_firing'] = pressed
    elif button == mouse.Button.right:
        estado_jugador['is_scoped'] = pressed

# NUEVO: Listeners para el teclado
def on_press(key):
    if not juego_en_ejecucion: return
    # Usamos key.shift para detectar la tecla Shift izquierda o derecha
    if isinstance(key, keyboard.Key) and key.name.startswith('shift'):
        estado_jugador['is_walking'] = True

def on_release(key):
    if not juego_en_ejecucion: return
    if isinstance(key, keyboard.Key) and key.name.startswith('shift'):
        estado_jugador['is_walking'] = False

# Iniciamos todos los listeners en hilos separados
mouse_move_listener = mouse.Listener(on_move=on_move)
mouse_click_listener = mouse.Listener(on_click=on_click)
keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)

mouse_move_listener.start()
mouse_click_listener.start()
keyboard_listener.start()


# --- Módulo de Estado del Sistema ---
def actualizar_estado_juego():
    global juego_en_ejecucion
    if "cs2.exe" in (p.name() for p in psutil.process_iter()):
        if not juego_en_ejecucion: print("¡Juego CS2 detectado! Iniciando recolección de datos...")
        juego_en_ejecucion = True
    else:
        if juego_en_ejecucion: print("El juego CS2 se ha cerrado. Deteniendo recolección de datos.")
        juego_en_ejecucion = False


# --- Bucle Principal de Recolección ---
print("Cliente anti-cheat iniciado. Presiona Ctrl+C para detener.")
INTERVALO_SEGUNDOS = 5

try:
    while True:
        actualizar_estado_juego()

        if juego_en_ejecucion:
            datos_de_raton_intervalo = list(mouse_movements)
            mouse_movements.clear()

            # Creamos el paquete de datos, ahora incluyendo el estado del jugador
            paquete_de_datos = {
                "timestamp_utc": time.time(),
                "intervalo_s": INTERVALO_SEGUNDOS,
                "game_active": True,
                "player_state": estado_jugador.copy(), # Copiamos el estado actual
                "mouse_data": datos_de_raton_intervalo
            }

            json_para_enviar = json.dumps(paquete_de_datos, indent=4)
            
            print("--- PAQUETE DE DATOS JSON LISTO PARA ENVIAR ---")
            print(json_para_enviar)
        else:
            print(f"CS2 no está en ejecución. Esperando...")

        time.sleep(INTERVALO_SEGUNDOS)

except KeyboardInterrupt:
    # Detenemos todos los listeners antes de salir
    mouse_move_listener.stop()
    mouse_click_listener.stop()
    keyboard_listener.stop()
    print("\nCliente detenido por el usuario.")
import psutil
import socket
import time
from pynput import mouse

juego_en_ejecucion = False
mouse_movements = []

def on_move(x, y):
    """Esta función se ejecuta con cada movimiento del ratón."""
    if not juego_en_ejecucion:
        return
    mouse_movements.append({'x': x, 'y': y, 'time': time.perf_counter()})

mouse_listener = mouse.Listener(on_move=on_move)
mouse_listener.start()

def actualizar_estado_juego():
    """Revisa si cs2.exe está en ejecución y actualiza el interruptor global."""
    global juego_en_ejecucion
    if "cs2.exe" in (p.name() for p in psutil.process_iter()):
        if not juego_en_ejecucion:
            print("¡Juego CS2 detectado! Iniciando recolección de datos...")
        juego_en_ejecucion = True
    else:
        if juego_en_ejecucion:
            print("El juego CS2 se ha cerrado. Deteniendo recolección de datos.")
        juego_en_ejecucion = False

print("Cliente anti-cheat iniciado. Presiona Ctrl+C para detener.")
INTERVALO_SEGUNDOS = 5

while True:
    try:
        actualizar_estado_juego()
        if juego_en_ejecucion:
            datos_de_raton_intervalo = list(mouse_movements)
            mouse_movements.clear()
            print(f"--- Reporte del Intervalo ({INTERVALO_SEGUNDOS}s) ---")
            print(f"  [Estado] Juego: Activo")
            print(f"  [Comportamiento] Movimientos de ratón capturados: {len(datos_de_raton_intervalo)}")
        else:
            print(f"CS2 no está en ejecución. Esperando... (próxima revisión en {INTERVALO_SEGUNDOS}s)")
        time.sleep(INTERVALO_SEGUNDOS)

    except KeyboardInterrupt:
        mouse_listener.stop()
        print("\nCliente detenido por el usuario.")
        break